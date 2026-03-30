# Design Document – 3‑D Stacked‑Plane RAG Framework

## 1. High‑Level Goal
Create a **general‑purpose Retrieval‑Augmented Generation (RAG) framework** that ships with a **3‑D stacked‑plane KV‑cache quantizer**. The framework must be:
- Model‑agnostic (any LLM backend can be plugged in).
- Language‑agnostic at the API level (Python core, optional bindings).
- Publicly released to serve as defensive prior‑art without revealing a writing‑specific use‑case.

The quantizer is an extension of Google’s TurboQuant (2‑D polar quantization) that captures cross‑plane relationships via a conditional 3‑D cell structure.

---

## 2. System Architecture
```
+-------------------+      +-------------------+      +-------------------+
|   Data Sources    | ---> |   Adapters Layer  | ---> |   Ingestion Engine |
+-------------------+      +-------------------+      +-------------------+
                                                        |
                                                        v
+-------------------+      +-------------------+      +-------------------+
|   Graph Index     | <---|   RAG Engine      | ---> |   Generation API   |
+-------------------+      +-------------------+      +-------------------+
                                                        |
                                                        v
+-------------------+      +-------------------+      +-------------------+
|   KV‑Cache Layer  | <---|   Quantizer       | ---> |   Model Backend   |
+-------------------+      +-------------------+      +-------------------+
```

### 2.1. Adapters Layer
- **BaseAdapter** – abstract interface `load()` → returns a list of documents/embeddings.
- Implementations: `TextAdapter` (plain text, markdown), `EmbeddingAdapter` (FAISS/Chroma), `APIAdapter` (REST endpoints).

### 2.2. Ingestion Engine
1. Split each document into **chunks** (configurable size, e.g., 512 tokens).
2. Compute embeddings (user‑provided function) for each chunk.
3. Store chunks as **nodes** in a graph index (bidirectional edges based on semantic similarity).
4. Persist the node embeddings in a vector store; the graph is saved as adjacency lists.

### 2.3. RAG Engine
- **Retrieval**: Given a query, embed it, perform nearest‑neighbor search → retrieve top‑K nodes.
- **Context Assembly**: Concatenate retrieved chunks, optionally reorder using graph traversal (e.g., shortest‑path covering).
- **Generation Hook**: Call user‑provided `generate(prompt, context)` which returns the LLM output.

### 2.4. KV‑Cache Layer & Quantizer
The KV‑cache sits between the **model backend** and the **RAG Engine**. For each transformer layer:
1. The model produces a KV matrix `K ∈ ℝ^{N×d}` and `V ∈ ℝ^{N×d}` where `N` is the token count and `d` the hidden dimension per head.
2. Before writing to GPU memory, the matrix is **rotated** by a global orthogonal matrix `R ∈ ℝ^{d×d}` (same as TurboQuant). This decorrelates dimensions.
3. The rotated vectors are partitioned into overlapping **triplets** `(x_i, y_i, z_i)` where `z_i` is the first coordinate of the next pair, creating a stacked‑plane structure.
4. **Quantization pipeline** (see Section 3) compresses each triplet into a compact code.
5. The compressed representation is stored in GPU memory; during attention computation the quantized values are de‑quantized on‑the‑fly (or used directly if the model can operate on quantized logits).
6. The quantized KV cache is then used for subsequent attention steps.

---

## 3. 3‑D Stacked‑Plane Quantizer Mathematics
### 3.1. Notation
- `v ∈ ℝ^{d}` – a rotated KV vector (after step 2 above).
- `d = 3·m` for some integer `m` (we group dimensions in triples).
- `v = [p_1, p_2, …, p_m]` where each `p_k = (x_k, y_k, z_k)`.
- `B_r, B_θ, B_φ` – number of bits allocated to radius, azimuth, and polar angles respectively.
- `C_r = 2^{B_r}`, `C_θ = 2^{B_θ}`, `C_φ = 2^{B_φ}` – cardinalities of the respective codebooks.

### 3.2. Polar Quantization (baseline 2‑D)
For a 2‑D pair `(x, y)`:
1. Convert to polar coordinates:
   - `r = sqrt(x² + y²)`
   - `θ = atan2(y, x)`
2. Quantize `r` and `θ` uniformly:
   - `r_q = round(r / Δ_r) * Δ_r` where `Δ_r = r_max / C_r`
   - `θ_q = round((θ + π) / Δ_θ) * Δ_θ` where `Δ_θ = 2π / C_θ`
3. Store indices `(i_r, i_θ)` (each fits in `B_r` and `B_θ` bits).

### 3.3. Extending to 3‑D – Conditional Stacked‑Plane
We keep the **coarse 1‑D index** for the shared coordinate `z` and then apply **plane‑specific 2‑D polar tables** conditioned on that index.
1. **Coarse quantization of `z`**
   - Define `B_z` bits → `C_z = 2^{B_z}` bins.
   - Uniform bin width `Δ_z = (z_max - z_min) / C_z`.
   - Compute `i_z = floor((z - z_min) / Δ_z)`.
2. **Conditional 2‑D tables**
   - For each possible `i_z` we pre‑compute a dedicated polar codebook `(C_r(i_z), C_θ(i_z))`. The tables can be learned offline (e.g., k‑means on rotated vectors restricted to that `z` slice) or simply use the same uniform quantization but with **different offsets** to reduce bias.
3. **Quantize `(x, y)` conditioned on `i_z`**
   - Retrieve the table for `i_z`.
   - Perform the polar quantization steps from §3.2 using that table, yielding indices `(i_r, i_θ)`.
4. **Packed representation**
   - Final code for the triplet: `code = (i_z << (B_r + B_θ)) | (i_r << B_θ) | i_θ`
   - Total bits per triplet = `B_z + B_r + B_θ`.
   - Typical configuration: `B_z = 4`, `B_r = 4`, `B_θ = 4` → **12 bits** per KV vector (vs. 8 bits for a pure 2‑D 4‑bit polar scheme).

### 3.4. De‑quantization
Given `code`:
1. Extract `i_z`, `i_r`, `i_θ` via bit masks.
2. Recover `z_q = z_min + (i_z + 0.5) * Δ_z`.
3. Using the same conditional table, compute `r_q` and `θ_q` from `i_r`, `i_θ`.
4. Convert back to Cartesian:
   - `x_q = r_q * cos(θ_q)`
   - `y_q = r_q * sin(θ_q)`
5. Re‑assemble the triplet `(x_q, y_q, z_q)` and apply the inverse rotation `Rᵀ` to obtain the approximate original KV vector.

### 3.5. Error Bounds (informal)
- **Radius error** ≤ `Δ_r / 2`.
- **Angle error** ≤ `Δ_θ / 2`.
- **Coarse `z` error** ≤ `Δ_z / 2`.
- The total Euclidean error for a triplet can be bounded by:
  ```
  ||v - v̂|| ≤ sqrt( (Δ_r/2)² + (r_max·Δ_θ/2)² + (Δ_z/2)² )
  ```
- Because `Δ_z` is typically much larger than `Δ_r` and `Δ_θ`, the dominant error comes from the 2‑D polar part; the coarse `z` index mainly provides **cross‑plane context**.

---

## 4. Execution Flow (per token)
1. **Model forward pass** → produces KV vectors for the new token.
2. **Rotate** each vector with `R` (pre‑computed, stored on GPU).
3. **Partition** rotated vector into overlapping triplets.
4. **Quantize** each triplet using the conditional stacked‑plane algorithm.
5. **Store** the compact codes in the KV cache memory.
6. **Attention**: when computing attention scores, de‑quantize the needed KV entries on‑the‑fly (vectorized kernel) or use the quantized representation directly if the attention kernel is adapted to operate on polar coordinates.
7. **Cache update**: repeat for subsequent tokens.

---

## 5. Integration Points for the 24/7 LLM Agent
| Component | What the agent must implement | Expected Input/Output |
|-----------|------------------------------|-----------------------|
| **Rotation matrix generator** (`utils.py`) | Compute an orthogonal matrix `R` (e.g., via QR decomposition of a random Gaussian matrix). Store as a GPU tensor. | `R` (d×d) |
| **Coarse `z` table builder** (`build_quantizer.py`) | Scan a large corpus of rotated KV vectors, compute min/max of each `z` dimension, decide `B_z`, `Δ_z`. Optionally cluster per‑`z` slice to produce refined 2‑D tables. | `z_bins`, per‑bin `(C_r, C_θ)` tables |
| **Conditional polar quantizer** (`stacked_plane.py`) | Functions `quantize(triplet)` and `dequantize(code)`. Must be fully vectorized (batch size = number of KV entries). | `code` (int) ↔ `(x, y, z)` approximations |
| **KV‑Cache wrapper** (`kv_cache.py`) | Replace the raw tensor cache with a structure that stores the compact codes and provides `get_dequantized()` for the attention kernel. | `store(code)`, `retrieve_dequantized()` |
| **Attention kernel adaptation** (optional) | Either modify the existing attention implementation to accept de‑quantized vectors on‑the‑fly, or rewrite it to work directly with polar coordinates (requires rewriting the dot‑product as `r_i·r_j·cos(θ_i-θ_j)`). | `attention(Q, K_codes, V_codes)` |
| **Benchmark suite** (`benchmark.py`) | Measure memory usage, latency, and retrieval quality for both the baseline 2‑D quantizer and the new 3‑D version on synthetic long‑context data. | CSV/JSON reports |

The 24/7 LLM should:
1. **Read this design doc** to understand the data flow and math.
2. **Generate concrete code** for each component listed above, respecting the repository layout defined in the implementation plan.
3. **Validate** the quantizer by running unit tests that check round‑trip error against the bound in §3.5.
4. **Iterate**: if the error is larger than expected, adjust `B_z`, `B_r`, `B_θ` or refine the conditional tables (e.g., k‑means per‑slice).

---

## 6. Deliverables for the Agent
- Fully functional `src/core/quantizer/stacked_plane.py` implementing the conditional algorithm.
- `scripts/build_quantizer.py` that produces the rotation matrix and per‑slice tables and writes them to `src/core/quantizer/tables/`.
- Updated `kv_cache.py` that stores compact codes and provides a de‑quantization API.
- Unit tests (`tests/unit/test_quantizer.py`) covering:
  * Random vector → quantize → de‑quantize → error < bound.
  * Consistency of conditional table lookup.
- Integration test (`tests/integration/test_end_to_end.py`) that runs a short transformer model with the KV‑cache layer on a synthetic 10k‑token sequence and reports memory savings.
- Benchmark script (`src/eval/benchmark.py`) that produces a comparative table for 2‑D vs 3‑D.

---

## 7. CUDA Kernel Specification (M15)

### Overview

The fused dequantize + polar attention kernel (`polar_attn_kernel.cu`) eliminates one global memory round-trip per attention step by computing attention scores directly from int16 PrismKV codes without materialising FP16 key tensors.

The kernel is written as defensive prior art and is compilable on CUDA >= 11.8.
It is **not compiled in this distribution** (CPU-only CI); build with `python setup_cuda.py build_ext --inplace` on a CUDA host.

### Thread-Block Layout and Occupancy

```
Grid:  (B*H, ceil(Sq/64), ceil(Sk/64))
Block: (32, 4, 1)   -- 128 threads = 4 warps
```

Each block computes a 64x64 tile of the (Sq x Sk) attention score matrix for one (batch x head) slice.

**Shared memory usage per block:**
- `smem_q`: 64 query triplets x 3 float components x 4 bytes = 768 bytes
- Total shared memory per block: 768 bytes
- sm_80 shared memory limit: 48 KB, so occupancy is limited by register count, not shared memory
- At 128 threads per block, theoretical occupancy on sm_80 allows up to 32 blocks per SM (4096 threads / 128 = 32).

### DRAM Bandwidth Savings

Loading int16 codes instead of FP16 key vectors:

| Format       | Bytes per key vector | Formula |
|--------------|---------------------|---------|
| FP16 keys    | 2 * d bytes         | 2d      |
| int16 codes  | 2 * m bytes         | 2d/3    |

**3x DRAM bandwidth reduction** for key loading (m = d/3).

At d=128 (LLaMA-style): FP16 costs 256 bytes per key vector; int16 codes cost 85 bytes.

### FLOP Count per (q, k) Pair

| Operation              | FLOPs   | Notes                                  |
|------------------------|--------|----------------------------------------|
| Standard Cartesian dot | 2d - 1 | d multiply-adds                        |
| PrismKV polar dot      | ~5m    | per triplet: 2 (r^2), 1 (cos), 2 (fma)|

For d=128, m=42: standard = 255 FLOPs, polar ~210 FLOPs (~1.2x FLOP reduction), plus the 3x bandwidth reduction from loading smaller codes.

### Hardware Targets

| GPU       | Architecture  | Build flag |
|-----------|--------------|------------|
| A100      | Ampere       | `sm_80`    |
| RTX 3090  | Ampere       | `sm_86`    |
| RTX 4090  | Ada Lovelace | `sm_89`    |
| H100 SXM5 | Hopper       | `sm_90`    |

The default build flag is `sm_80`. For multi-architecture PTX:
`-gencode arch=compute_80,code=sm_80 -gencode arch=compute_90,code=sm_90`.

The kernel uses `__cosf()` (hardware fast-path single-precision cosine, available on all supported architectures). FP32 accumulation is used throughout for numerical stability.

### llama.cpp / Consumer Inference Integration: C++ Memory Layout

This section specifies how PrismKV codes map to the 32-bit-aligned memory layout required
by `llama.cpp` (GGML), ExLlamaV2, and similar consumer inference engines. It constitutes
prior art for integrating PrismKV with the local LLM ecosystem.

**Core insight from external SME review (2026-03-30):** Consumer GPU users are VRAM-bound.
A 100k-token code RAG context at FP16 costs ~50 GB of KV cache on LLaMA-3-70B. PrismKV at
4 bits/dim compresses this to ~8 GB — fitting a 100k context window into a single A100.
The integration path below enables this without Python overhead.

#### Packed Code Layout (32-bit register)

Each triplet group produces 3 indices: `i_z` (bits_z bits), `i_r` (bits_r bits), `i_theta`
(bits_theta bits). For the equal-split 4+4+4 configuration (12 bits/triplet), two triplets
fit into one 32-bit integer with 8 bits spare (usable for alignment or future extension):

```c
/* PrismKV code word — one uint32 holds 2 triplet groups at 4+4+4 bits each */
typedef uint32_t prismkv_codeword_t;

/* Pack two triplet groups into one uint32:
 *   bits [0..3]   = i_theta of group 0
 *   bits [4..7]   = i_r     of group 0
 *   bits [8..11]  = i_z     of group 0
 *   bits [12..15] = i_theta of group 1
 *   bits [16..19] = i_r     of group 1
 *   bits [20..23] = i_z     of group 1
 *   bits [24..31] = reserved / alignment
 */
static inline uint32_t prismkv_pack2(
    uint8_t iz0, uint8_t ir0, uint8_t it0,
    uint8_t iz1, uint8_t ir1, uint8_t it1)
{
    return ((uint32_t)iz1 << 20) | ((uint32_t)ir1 << 16) | ((uint32_t)it1 << 12)
         | ((uint32_t)iz0 <<  8) | ((uint32_t)ir0 <<  4) | (uint32_t)it0;
}

static inline void prismkv_unpack2(
    uint32_t cw,
    uint8_t *iz0, uint8_t *ir0, uint8_t *it0,
    uint8_t *iz1, uint8_t *ir1, uint8_t *it1)
{
    *it0 = (cw >>  0) & 0xF;
    *ir0 = (cw >>  4) & 0xF;
    *iz0 = (cw >>  8) & 0xF;
    *it1 = (cw >> 12) & 0xF;
    *ir1 = (cw >> 16) & 0xF;
    *iz1 = (cw >> 20) & 0xF;
}
```

For `head_dim=128` (LLaMA-family, m=42 triplet groups after DimAligner padding to 126):
`ceil(42/2) = 21` uint32 words per KV vector — **84 bytes vs 256 bytes for FP16**.

#### Per-Layer Quantization Metadata

```c
/* Uniform-z quantization parameters (loaded once per layer into SRAM) */
typedef struct {
    float z_min;        /* z-axis range minimum                         */
    float delta_z;      /* (z_max - z_min) / bins_z                     */
    float r_max;        /* maximum radius for (x,y) polar quantization  */
    float delta_r;      /* r_max / (bins_r - 1)                         */
    float delta_theta;  /* 2*pi / (bins_theta - 1)                      */
    float scale;        /* attention logit scale factor (1/sqrt(head_d)) */
    int32_t bits_z;     /* bits for z quantization (typically 4)        */
    int32_t bits_r;     /* bits for r quantization (typically 4)        */
    int32_t bits_theta; /* bits for theta quantization (typically 4)    */
    int32_t n_groups;   /* head_dim / 3 = m                             */
} prismkv_layer_params_t;  /* 40 bytes — fits in 2 cache lines */
```

#### Static Codebook Layout (Lloyd-Max variant)

When Lloyd-Max z quantization is active (M13), the uniform `delta_z` formula is replaced
by a centroid lookup table. The static codebook loads into SRAM before the generation loop:

```c
#define PRISMKV_MAX_Z_BINS 16

/* Static z codebook — 16 centroids, loaded once into __shared__ SRAM     */
typedef struct {
    float z_centroids[PRISMKV_MAX_Z_BINS];  /* optimal reconstruction values */
    float z_boundaries[PRISMKV_MAX_Z_BINS + 1]; /* bin boundaries for encode */
} prismkv_z_codebook_t;  /* 132 bytes — fits in 3 cache lines */

/* Usage in CUDA kernel (decode path):
 *   __shared__ prismkv_z_codebook_t s_zbook;
 *   if (threadIdx.x == 0) s_zbook = *g_zbook;  // load from global memory once
 *   __syncthreads();
 *   float z_q = s_zbook.z_centroids[i_z];       // SRAM lookup, no divergence
 */
```

#### GGML Type Registration (llama.cpp integration path)

```c
/* New quantization type for llama.cpp — registers with ggml_type system */
#define GGML_TYPE_PRISMKV_4   (GGML_TYPE_COUNT + 1)  /* 4 bits/dim, 4+4+4 split */

static const ggml_type_traits_t ggml_type_prismkv_4 = {
    .type_name         = "prismkv_q4_0",
    .blck_size         = 6,        /* 6 dims per packed uint32 (2 triplets) */
    .type_size         = 4,        /* 4 bytes per 6-dim block               */
    .is_quantized      = true,
    .vec_dot           = prismkv_vec_dot_q4_f32,
    .vec_dot_type      = GGML_TYPE_F32,
};

/* vec_dot implementation: calls prismkv_polar_attn_fwd for the head slice */
```

#### Consumer Hardware Adoption Path

| Engine        | Integration point                  | Status                    |
|---------------|------------------------------------|---------------------------|
| llama.cpp     | Register `GGML_TYPE_PRISMKV_4`     | Spec above; needs C PR    |
| ExLlamaV2     | `ext_c` custom quantization type   | Compatible with layout     |
| MLX (Apple)   | Metal kernel, same bit layout      | Needs Metal shader port    |
| vLLM          | `VLLMSwapCompressor` (M12, done)   | CPU swap boundary, done   |

The static codebook approach (one 132-byte SRAM load per layer) eliminates the dynamic
per-token codebook lookup that would otherwise cause warp divergence — directly addressing
the hardware-sympathy concern identified in external SME review.

### Note on Distribution

`src/prismkv/cuda/polar_attn_kernel.cu` is included as syntactically-valid CUDA C++ source. The Python interface (`src/prismkv/cuda/__init__.py`) falls back silently to the pure-Python `polar_attention.py` implementation when the CUDA extension is not available, so all CPU-mode tests pass without building the extension.

---

## 8. References (for the agent)
- TurboQuant paper (ICLR 2026) – 2‑D polar quantization and KV‑cache compression.
- “PolarQuant + QJL” analysis – two‑stage pipeline that inspired the conditional approach.
- Product Quantization literature – explains why independent sub‑spaces waste cross‑dimensional information.
- Existing open‑source KV‑cache compressors (e.g., `kvcompress` on GitHub) – useful for code‑style reference.

---

**End of Design Document**
