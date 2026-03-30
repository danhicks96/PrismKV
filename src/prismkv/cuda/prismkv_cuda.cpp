#include <torch/extension.h>

void prismkv_polar_attn_fwd_cuda(
    torch::Tensor Q, torch::Tensor K_codes,
    torch::Tensor q_params, torch::Tensor bits,
    torch::Tensor out
);

torch::Tensor polar_attn_fwd(
    torch::Tensor Q, torch::Tensor K_codes,
    torch::Tensor q_params, torch::Tensor bits
) {
    auto B = Q.size(0), H = Q.size(1), Sq = Q.size(2);
    auto Sk = K_codes.size(2);
    auto out = torch::zeros({B, H, Sq, Sk}, Q.options().dtype(torch::kFloat32));
    prismkv_polar_attn_fwd_cuda(Q, K_codes, q_params, bits, out);
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("polar_attn_fwd", &polar_attn_fwd,
          "PrismKV fused dequantize + polar attention forward (CUDA)");
}
