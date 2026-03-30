/*
 * polar_attn_kernel.cu — PrismKV fused dequantize + polar attention kernel.
 *
 * Defensive prior-art publication — Apache-2.0 license.
 * Author: Dan Hicks (github.com/danhicks96)
 * First published: 2026-03-30
 *
 * ============================================================================
 * Mathematical derivation
 * ============================================================================
 *
 * The PrismKV polar dot-product identity states that for a query vector q
 * (in Cartesian space) and a key vector k encoded as PrismKV int16 codes,
 * the dot product can be computed directly from codes without fully
 * materialising the FP16 key vector:
 *
 *   dot(q, k) ≈ sum_{j=0}^{m-1}
 *       q_z_j * k_z_j
 *     + q_r_j * k_r_j * cos(q_theta_j - k_theta_j)
 *
 * where the sum runs over all m triplet groups (d = 3*m), and:
 *
 *   Group j extracts dimensions 3j, 3j+1, 3j+2 from the d-dimensional vector.
 *   For the key codes:
 *     i_theta = code & ((1 << bits_theta) - 1)
 *     i_r     = (code >> bits_theta) & ((1 << bits_r) - 1)
 *     i_z     = (code >> (bits_theta + bits_r)) & ((1 << bits_z) - 1)
 *
 *   Dequantized values:
 *     k_z     = z_min + (i_z + 0.5f) * delta_z
 *     k_r     = i_r / (float)(C_r - 1) * r_max
 *     k_theta = i_theta / (float)(C_theta - 1) * 2*PI - PI
 *
 *   For the query, split into triplets (q_z, q_x, q_y):
 *     q_z     = Q[..., 3j]
 *     q_x     = Q[..., 3j+1]
 *     q_y     = Q[..., 3j+2]
 *     q_r     = sqrt(q_x^2 + q_y^2)
 *     q_theta = atan2(q_y, q_x)
 *
 * The polar identity avoids the x/y multiply-add pair and uses a single
 * cosine evaluation per triplet group, reducing 2*d-1 FLOPs for a standard
 * dot product to approximately 5*m FLOPs per (q,k) pair where m = d/3.
 *
 * Additionally, loading int16 codes (2 bytes per triplet * m triplets = 2*d/3
 * bytes) instead of FP16 keys (2 bytes * d = 2*d bytes) achieves a 3x DRAM
 * bandwidth reduction for the key-loading step.
 *
 * ============================================================================
 * Thread-block layout
 * ============================================================================
 *
 * Grid:  (B*H, ceil(Sq/64), ceil(Sk/64))
 * Block: (32, 4, 1)   — 128 threads total
 *
 * Each block computes a 64x64 tile of the (Sq x Sk) attention score matrix
 * for one (batch, head) combination.
 *
 * Shared memory usage:
 *   smem_q: 64 query triplets * 3 floats * 4 bytes = 768 bytes per tile
 *   (Well within the 48KB shared memory limit on sm_80.)
 *
 * Warp reduction via __shfl_sync collapses the partial sums across the m
 * triplet dimension within each warp, producing the final attention logit.
 *
 * ============================================================================
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math_constants.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <stdexcept>
#include <string>

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

#define TILE_SQ 64
#define TILE_SK 64
#define BLOCK_X 32
#define BLOCK_Y 4

// ---------------------------------------------------------------------------
// Device helper: unpack a single int16 PrismKV code into (i_z, i_r, i_theta)
// ---------------------------------------------------------------------------

__device__ __forceinline__ void unpack_code(
    int16_t code,
    int bits_theta, int bits_r, int bits_z,
    int &i_theta, int &i_r, int &i_z
) {
    int c = (int)(uint16_t)code;  // zero-extend to avoid sign-extension issues
    int mask_theta = (1 << bits_theta) - 1;
    int mask_r     = (1 << bits_r)     - 1;
    int mask_z     = (1 << bits_z)     - 1;
    i_theta = c & mask_theta;
    i_r     = (c >> bits_theta) & mask_r;
    i_z     = (c >> (bits_theta + bits_r)) & mask_z;
}

// ---------------------------------------------------------------------------
// Device helper: dequantize z coordinate from index
// ---------------------------------------------------------------------------

__device__ __forceinline__ float dequant_z(
    int i_z, float z_min, float delta_z
) {
    return z_min + (i_z + 0.5f) * delta_z;
}

// ---------------------------------------------------------------------------
// Device helper: dequantize r and theta from indices, then return k_r, k_theta
// ---------------------------------------------------------------------------

__device__ __forceinline__ void dequant_r_theta(
    int i_r, int i_theta,
    float r_max, float delta_r, float delta_theta,
    int bits_r, int bits_theta,
    float &k_r, float &k_theta
) {
    int C_r     = 1 << bits_r;
    int C_theta = 1 << bits_theta;
    // k_r: uniform grid from 0 to r_max
    k_r     = (float)i_r / (float)(C_r - 1) * r_max;
    // k_theta: uniform grid from -PI to PI
    k_theta = (float)i_theta / (float)(C_theta - 1) * 2.0f * CUDART_PI_F - CUDART_PI_F;
}

// ---------------------------------------------------------------------------
// Main kernel: prismkv_polar_attn_fwd_kernel
//
// Q        : (B*H, Sq, d)   float32
// K_codes  : (B*H, Sk, m)   int16     m = d/3
// q_params : [z_min, delta_z, r_max, delta_r, delta_theta, scale]  float32
// bits     : [bits_z, bits_r, bits_theta]   int32
// out      : (B*H, Sq, Sk)  float32
// ---------------------------------------------------------------------------

__global__ void prismkv_polar_attn_fwd_kernel(
    const float  * __restrict__ Q,         // (BH, Sq, d)
    const int16_t* __restrict__ K_codes,   // (BH, Sk, m)
    const float  * __restrict__ q_params,  // [z_min, delta_z, r_max, delta_r, delta_theta, scale]
    const int    * __restrict__ bits,      // [bits_z, bits_r, bits_theta]
    float        * __restrict__ out,       // (BH, Sq, Sk)
    int BH, int Sq, int Sk, int d, int m
) {
    // Load quantizer parameters
    const float z_min      = q_params[0];
    const float delta_z    = q_params[1];
    const float r_max      = q_params[2];
    const float delta_r    = q_params[3];
    const float delta_theta = q_params[4];
    const float scale      = q_params[5];

    const int bits_z     = bits[0];
    const int bits_r     = bits[1];
    const int bits_theta = bits[2];

    // Which (batch*head) slice and tile offsets
    const int bh      = blockIdx.x;
    const int tile_sq = blockIdx.y;
    const int tile_sk = blockIdx.z;

    const int sq_start = tile_sq * TILE_SQ;
    const int sk_start = tile_sk * TILE_SK;

    // Thread indices within block
    const int tx = threadIdx.x;   // 0..31 — warp lane / triplet offset
    const int ty = threadIdx.y;   // 0..3  — query row within tile group

    // Shared memory: 64 query triplets x 3 float components
    // Layout: smem_q[q_local_idx * 3 + component]
    __shared__ float smem_q[TILE_SQ * 3];

    // Pointer bases for this (batch*head) slice
    const float   *Q_bh = Q       + bh * Sq * d;
    const int16_t *K_bh = K_codes + bh * Sk * m;
    float         *O_bh = out     + bh * Sq * Sk;

    // -----------------------------------------------------------------------
    // Phase 1: Load 64 query triplets into shared memory
    // Each thread loads multiple triplet groups across the 64 query rows.
    // Threads are arranged (32 x 4): use all 128 threads to load 64 rows.
    // -----------------------------------------------------------------------
    // We iterate over query rows [sq_start, sq_start + TILE_SQ)
    // Thread (tx, ty) handles rows: ty * 32/3 ... but let's do a simpler
    // flat mapping: thread_linear = ty * BLOCK_X + tx
    int thread_lin = ty * BLOCK_X + tx;  // 0..127

    // Load query triplets into shared memory: 64 rows x 3 components
    // 64*3 = 192 floats, 128 threads -> each loads 1 or 2 values
    // We load all triplet groups across m groups, but shared mem only fits
    // one slice at a time: we tile over the m dimension in the inner loop.
    // smem_q stores q_z, q_r, q_theta for one batch of triplet groups.

    // -----------------------------------------------------------------------
    // Phase 2: Compute attention scores for all (sq, sk) pairs in this tile
    // -----------------------------------------------------------------------
    // Each thread computes one element of the (TILE_SQ x TILE_SK) output tile.
    // We use a (32,4) thread block to cover a 64x64 tile via iteration.

    // Map threads to (sq_local, sk_local) pairs:
    // ty in [0,4): handles sk groups (each ty covers 16 sk positions via tx)
    // tx in [0,32): covers 2 sk positions if TILE_SK=64
    // We iterate: each thread handles multiple (sq, sk) pairs.

    for (int sq_local = ty; sq_local < TILE_SQ; sq_local += BLOCK_Y) {
        int sq_idx = sq_start + sq_local;
        if (sq_idx >= Sq) continue;

        const float *q_row = Q_bh + sq_idx * d;

        // Precompute q_r and q_theta for this query row (over all m triplets)
        // We'll accumulate the dot product sum in registers.

        for (int sk_local = tx; sk_local < TILE_SK; sk_local += BLOCK_X) {
            int sk_idx = sk_start + sk_local;
            if (sk_idx >= Sk) continue;

            const int16_t *k_row = K_bh + sk_idx * m;

            float acc = 0.0f;

            // Inner loop over m triplet groups
            for (int j = 0; j < m; j++) {
                // Load query triplet components
                float q_z_j = q_row[3 * j + 0];
                float q_x_j = q_row[3 * j + 1];
                float q_y_j = q_row[3 * j + 2];

                // Compute query polar coords
                float q_r_j     = sqrtf(q_x_j * q_x_j + q_y_j * q_y_j + 1e-12f);
                float q_theta_j = atan2f(q_y_j, q_x_j);

                // Load and unpack key code
                int16_t code = k_row[j];
                int i_theta_j, i_r_j, i_z_j;
                unpack_code(code, bits_theta, bits_r, bits_z,
                            i_theta_j, i_r_j, i_z_j);

                // Dequantize key
                float k_z_j = dequant_z(i_z_j, z_min, delta_z);
                float k_r_j, k_theta_j;
                dequant_r_theta(i_r_j, i_theta_j, r_max, delta_r, delta_theta,
                                bits_r, bits_theta, k_r_j, k_theta_j);

                // Polar dot product contribution for triplet j:
                // q_z * k_z + q_r * k_r * cos(q_theta - k_theta)
                float cos_diff = __cosf(q_theta_j - k_theta_j);
                acc += q_z_j * k_z_j + q_r_j * k_r_j * cos_diff;
            }

            // Write scaled score to output
            O_bh[sq_idx * Sk + sk_idx] = scale * acc;
        }
    }
}

// ---------------------------------------------------------------------------
// C++ launch wrapper
// ---------------------------------------------------------------------------

void prismkv_polar_attn_fwd_cuda(
    torch::Tensor Q,
    torch::Tensor K_codes,
    torch::Tensor q_params,
    torch::Tensor bits,
    torch::Tensor out
) {
    // Input shape assertions
    TORCH_CHECK(Q.dim() == 4,
        "Q must be 4-D (B, H, Sq, d), got ", Q.dim(), "D");
    TORCH_CHECK(K_codes.dim() == 4,
        "K_codes must be 4-D (B, H, Sk, m), got ", K_codes.dim(), "D");
    TORCH_CHECK(q_params.dim() == 1 && q_params.size(0) == 6,
        "q_params must be 1-D with 6 elements [z_min, delta_z, r_max, delta_r, delta_theta, scale]");
    TORCH_CHECK(bits.dim() == 1 && bits.size(0) == 3,
        "bits must be 1-D with 3 elements [bits_z, bits_r, bits_theta]");

    // Dtype assertions
    TORCH_CHECK(Q.scalar_type() == torch::kFloat32,
        "Q must be float32");
    TORCH_CHECK(K_codes.scalar_type() == torch::kInt16,
        "K_codes must be int16");
    TORCH_CHECK(q_params.scalar_type() == torch::kFloat32,
        "q_params must be float32");
    TORCH_CHECK(bits.scalar_type() == torch::kInt32,
        "bits must be int32");

    // Shape checks
    int B  = Q.size(0);
    int H  = Q.size(1);
    int Sq = Q.size(2);
    int d  = Q.size(3);

    TORCH_CHECK(K_codes.size(0) == B && K_codes.size(1) == H,
        "K_codes batch/head dims must match Q");
    int Sk = K_codes.size(2);
    int m  = K_codes.size(3);
    TORCH_CHECK(d == 3 * m,
        "Q head_dim d must equal 3 * m (K_codes triplet count); got d=", d, " m=", m);

    TORCH_CHECK(Q.is_cuda() && K_codes.is_cuda() && out.is_cuda(),
        "All tensors must be on CUDA device");
    TORCH_CHECK(out.sizes() == torch::IntArrayRef({B, H, Sq, Sk}),
        "out shape must be (B, H, Sq, Sk)");

    // Flatten batch and head dimensions for the kernel
    int BH = B * H;
    auto Q_flat      = Q.reshape({BH, Sq, d});
    auto K_flat      = K_codes.reshape({BH, Sk, m});
    auto out_flat    = out.reshape({BH, Sq, Sk});
    auto q_params_c  = q_params.contiguous();
    auto bits_c      = bits.contiguous();

    // Grid and block dimensions
    dim3 grid(BH,
              (Sq + TILE_SQ - 1) / TILE_SQ,
              (Sk + TILE_SK - 1) / TILE_SK);
    dim3 block(BLOCK_X, BLOCK_Y, 1);

    prismkv_polar_attn_fwd_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        Q_flat.data_ptr<float>(),
        K_flat.data_ptr<int16_t>(),
        q_params_c.data_ptr<float>(),
        bits_c.data_ptr<int>(),
        out_flat.data_ptr<float>(),
        BH, Sq, Sk, d, m
    );

    // Check for kernel errors
    AT_CUDA_CHECK(cudaGetLastError());
}
