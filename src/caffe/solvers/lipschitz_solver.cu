#include <string>

#include "caffe/util/gpu_math_functions.cuh"
#include "caffe/util/math_functions.hpp"

#pragma clang diagnostic push
#pragma ide diagnostic ignored "CannotResolve"
namespace caffe {

template<typename Dtype>
__global__ void LipschitzRegUpdateAllAndClear(int N,
		Dtype* g, Dtype *w,
		Dtype* h_g, Dtype* h_w,
    float rate,  float decay,
    bool reg_L2, bool clear_grads) {
  CUDA_KERNEL_LOOP(i, N) {
    h_g[i] = g[i];
    h_w[i] = w[i];
    float reg = reg_L2 ? (float)w[i] : float((Dtype(0.F) < w[i]) - (w[i] < Dtype(0.F)));
    float gr = float(g[i]) + reg * decay;
    w[i] -=  (Dtype) (gr * rate);
    g[i] = clear_grads ? Dtype(0) : g[i];
  }
}

#pragma clang diagnostic pop

/*
template<>
__global__ void LipschitzRegUpdateAllAndClear<half, half>(int N,
  half* g, half *w, half* m, half* v,
    float beta1, float beta2, float eps_hat, float local_rate, float local_decay,
    bool reg_L2,  bool clear_grads) {
  half hz;
  CUDA_KERNEL_LOOP(i, N) {
    float wf = __half2float(w[i]);
    float gf = __half2float(g[i]);
    float mf = __half2float(m[i]);
    float vf = __half2float(v[i]);

    float reg = reg_L2 ? wf : float((0.F < wf)-(wf < 0.F));
    gf += reg * local_decay;
    mf = beta1 * mf + (1.F - beta1)*gf;
    vf = beta2 * vf + (1.F - beta2)*gf*gf;
    gf = local_rate * mf / sqrt(vf + eps_hat);
    wf -= gf;

    w[i] = float2half_clip(wf);
    m[i] = float2half_clip(mf);
    v[i] = float2half_clip(vf);
    g[i] = clear_grads ? hz : float2half_clip(gf);
  }
}
*/

template<typename Dtype>
void Lipschitz_reg_update_and_clear_gpu(int N,
  Dtype* g,    Dtype *w,  Dtype* h_g,  Dtype* h_w,
  float rate, const std::string& reg_type, float decay,
  void *handle, bool clear_grads) {
  cublasHandle_t cublas_handle =
      handle == nullptr ? Caffe::cublas_handle(0) : reinterpret_cast<cublasHandle_t>(handle);
  cudaStream_t stream;
  CUBLAS_CHECK(cublasGetStream(cublas_handle, &stream));
  LipschitzRegUpdateAllAndClear  // NOLINT_NEXT_LINE(whitespace/operators)
      <<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0, stream>>>(N,
      g, w, h_g, h_w, rate, decay, reg_type == "L2",
      clear_grads);
  CUDA_POST_KERNEL_CHECK;
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

template void Lipschitz_reg_update_and_clear_gpu<float16>(int, float16*, float16*, float16*, float16*, float,
    const std::string&, float, void*, bool);
template void Lipschitz_reg_update_and_clear_gpu<float>(int,   float*,   float*, float*, float*, float,
    const std::string&, float, void*, bool);
template void Lipschitz_reg_update_and_clear_gpu<double>(int,  double*,  double*, double*, double*, float,
    const std::string&, float, void*, bool);

}  // namespace caffe
