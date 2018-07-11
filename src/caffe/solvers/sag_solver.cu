#include <string>

#include "caffe/util/gpu_math_functions.cuh"
#include "caffe/util/math_functions.hpp"

#pragma clang diagnostic push
#pragma ide diagnostic ignored "CannotResolve"
namespace caffe {

template<typename Dtype>
__global__ void SAGRegUpdateAllAndClear(int N,
    Dtype* g, Dtype *w, Dtype* h,
    float momentum, float rate,  float decay, bool clear_grads) {
    float m1 = 1.F - momentum;
    Dtype dz = Dtype(0.);

  CUDA_KERNEL_LOOP(i, N) {
    /*
    float hf = momentum * float(h[i]) + m1 * (float(g[i]) + decay*float(w[i]));
 //   h[i] -=  (1.F- momentum) * (g[i] - h[i]);
    h[i]=Dtype(hf);
    w[i] -= Dtype(rate * hf);
    if (clear_grads)
      g[i] = Dtype(0.);
    */
    float wf = w[i];
    float gf = g[i];
    float hf = h[i];
    hf = momentum * hf + m1 * (gf + decay * wf);
    wf = wf - rate * hf;
   // h[i] -= (1.F - momentum) * (g[i] + decay*w[i] - h[i]);

    h[i] = Dtype(hf);
    w[i] = Dtype(wf);
    if (clear_grads) {
      g[i] = dz;
    }
  }
}

template<>
__global__ void SAGRegUpdateAllAndClear<half>(int N,
    half* g, half* w, half* h,
    float momentum, float rate,  float decay, bool clear_grads) {
  float m1 = 1.F - momentum;
  half hz=__float2half(0.F);

  CUDA_KERNEL_LOOP(i, N) {
    float wf = __half2float(w[i]);
    float gf = __half2float(g[i]);
    float hf = __half2float(h[i]);

    hf = momentum * hf + m1 * (gf + decay * wf);
   // hf -= (1.F - momentum) * (gf + decay*wf - hf);
    wf -= rate * hf;

    h[i] = float2half_clip(hf);
    w[i] = float2half_clip(wf);
    if (clear_grads) {
         g[i] = hz;
    }
  }
}


template<typename Dtype>
__global__ void SAGWdUpdateAllAndClear(int N, Dtype* g, Dtype *w, Dtype* h,
		float momentum, float rate,  float decay, bool clear_grads) {
  float m1 = 1.F - momentum;
  CUDA_KERNEL_LOOP(i, N) {
    h[i] = Dtype(momentum * h[i] + m1 * g[i]);
 //   h[i] -=  (1.F- momentum) * (g[i] - h[i]);
    w[i] -= Dtype(rate * (float(h[i]) + decay*float(w[i])));
    if (clear_grads) {
      g[i] = Dtype(0.);
    }
  }
}

template<>
__global__ void SAGWdUpdateAllAndClear<half>(int N,
    half* g, half* w, half* h,
    float momentum, float rate,  float decay, bool clear_grads) {
  half hz=float2half_clip(0.F);
  CUDA_KERNEL_LOOP(i, N) {
    float wf = __half2float(w[i]);
    float gf = __half2float(g[i]);
    float hf = __half2float(h[i]);

    hf = momentum * hf + (1.F- momentum) * gf;
   // hf -= (1.F - momentum) * (gf - hf);
    wf-= rate * (hf + decay*wf);

    h[i] = float2half_clip(hf);
    w[i] = float2half_clip(wf);
    g[i] = clear_grads ? hz : float2half_clip(gf);
  }
}

#pragma clang diagnostic pop



template<typename Dtype>
void SAG_reg_update_and_clear_gpu(int N,
  Dtype* g, Dtype *w,  Dtype* h,
  float momentum, float rate,
  const std::string& reg_type, float decay,
  void *handle, bool clear_grads) {
  cublasHandle_t cublas_handle =
      handle == nullptr ? Caffe::cublas_handle(0) : reinterpret_cast<cublasHandle_t>(handle);
  cudaStream_t stream;
  CUBLAS_CHECK(cublasGetStream(cublas_handle, &stream));
  if (reg_type == "L2") {
    SAGRegUpdateAllAndClear  // NOLINT_NEXT_LINE(whitespace/operators)
      <<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0, stream>>>(N,
      g, w, h, momentum, rate, decay, clear_grads);
  } else if (reg_type == "WD") {
    SAGWdUpdateAllAndClear  // NOLINT_NEXT_LINE(whitespace/operators)
         <<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0, stream>>>(N,
         g, w, h, momentum, rate, decay, clear_grads);
  } else {
    LOG(FATAL) << "Unknown regularization mode: " << reg_type;
  }

  CUDA_POST_KERNEL_CHECK;
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

template<>
void SAG_reg_update_and_clear_gpu<float16>(int N,
  float16* g, float16 *w,  float16* h,
  float momentum, float rate,
  const std::string& reg_type, float decay,
  void *handle, bool clear_grads) {
  cublasHandle_t cublas_handle =
      handle == nullptr ? Caffe::cublas_handle(0) : reinterpret_cast<cublasHandle_t>(handle);
  cudaStream_t stream;
  CUBLAS_CHECK(cublasGetStream(cublas_handle, &stream));
  if (reg_type == "L2") {
    SAGRegUpdateAllAndClear  // NOLINT_NEXT_LINE(whitespace/operators)
      <<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0, stream>>>(N,
      g, w, h, momentum, rate, decay, clear_grads);
  } else { //if (reg_type == "WD") {
    SAGWdUpdateAllAndClear  // NOLINT_NEXT_LINE(whitespace/operators)
         <<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0, stream>>>(N,
             reinterpret_cast<half*>(g), reinterpret_cast<half*>(w), reinterpret_cast<half*>(h),
             momentum, rate, decay, clear_grads);
  }
  CUDA_POST_KERNEL_CHECK;
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

template void SAG_reg_update_and_clear_gpu<float>(int,   float*,    float*,  float*,  float, float,
    const std::string&, float, void*, bool);
template void SAG_reg_update_and_clear_gpu<double>(int,   double*,   double*, double*, float, float,
    const std::string&, float, void*, bool);

}  // namespace caffe
