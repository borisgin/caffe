#include <string>

#include "caffe/util/gpu_math_functions.cuh"
#include "caffe/util/math_functions.hpp"

#pragma clang diagnostic push
#pragma ide diagnostic ignored "CannotResolve"
namespace caffe {


template<typename Gtype, typename Wtype, typename Htype>
__global__ void SAGRegUpdateAllAndClear(int N,
    Gtype* g, Wtype* w, Htype* h,
    float momentum, float rate,  float decay, bool clear_grads) {
  float m1 = 1.F - momentum;
  Gtype gz = Gtype(0);
  CUDA_KERNEL_LOOP(i, N) {
    float wf = float(w[i]);
    float gf = float(g[i]);
    float hf = float(h[i]);
    hf = momentum * hf + m1 * (gf + decay * wf);
    wf -= rate * hf;
    w[i] = Wtype(wf);
    h[i] = Htype(hf);
    if (clear_grads) {
      g[i] = gz;
    }
  }
}

template<>
__global__ void SAGRegUpdateAllAndClear<half, half, half>(int N,
    half* g, half* w, half* h,
    float momentum, float rate,  float decay, bool clear_grads) {
  float m1 = 1.F - momentum;
  half hz;
  CUDA_KERNEL_LOOP(i, N) {
    float wf = __half2float(w[i]);
    float gf = __half2float(g[i]);
    float hf = __half2float(h[i]);
    hf = momentum * hf + m1 * (gf + decay * wf);
    wf -= rate * hf;
    w[i] = float2half_clip(wf);
    h[i] = float2half_clip(hf);
    if (clear_grads) {
      g[i] = hz;
    }
  }
}

template<>
__global__ void SAGRegUpdateAllAndClear<float, float,  half>(int N,
    float* g, float* w, half* h,
    float momentum, float rate,  float decay, bool clear_grads) {
  float m1 = 1.F - momentum;
  CUDA_KERNEL_LOOP(i, N) {
    float wf = w[i];
    float hf = __half2float(h[i]);
    hf = momentum * hf + m1 * (g[i] + decay * wf);
    wf -= rate * hf;
    w[i] = wf;
    h[i] = float2half_clip(hf);
    if (clear_grads) {
      g[i] = 0.F;
    }
  }
}

template<>
__global__ void SAGRegUpdateAllAndClear<half, float, float>(int N,
    half* g, float* w, float* h,
    float momentum, float rate,  float decay, bool clear_grads) {
  float m1 = 1.F - momentum;
  half hz;
  CUDA_KERNEL_LOOP(i, N) {
    float wf = w[i];
    float hf = momentum * h[i] + m1 * (__half2float(g[i]) + decay * wf);
    wf -= rate * hf;
    w[i] = wf;
    h[i] = hf;
    if (clear_grads) {
      g[i] = hz;
    }
  }
}

template<typename Gtype, typename Wtype, typename Htype>
__global__ void SAGWdUpdateAllAndClear(int N,
    Gtype* g, Wtype *w, Htype* h,
		float momentum, float rate,  float decay, bool clear_grads) {
  float m1 = 1.F - momentum;
  CUDA_KERNEL_LOOP(i, N) {
    float wf = float(w[i]);
    float gf = float(g[i]);
    float hf = float(h[i]);
    hf = momentum * hf + m1 * gf;
    wf -= rate * (hf + decay * wf);
    w[i] = Wtype(wf);
    h[i] = Htype(hf);
    if (clear_grads) {
      g[i] = Gtype(0);
    }
  }
}

template<>
__global__ void SAGWdUpdateAllAndClear<half, half, half>(int N,
    half* g, half* w, half* h,
    float momentum, float rate,  float decay, bool clear_grads) {
  float m1= 1.F - momentum;
  half hz;
  CUDA_KERNEL_LOOP(i, N) {
    float wf = __half2float(w[i]);
    float gf = __half2float(g[i]);
    float hf = __half2float(h[i]);
    hf = momentum * hf + m1 * gf;
    wf-= rate * (hf + decay*wf);
    h[i] = float2half_clip(hf);
    w[i] = float2half_clip(wf);
    if (clear_grads) {
      g[i] = hz;
    }
  }
}

template<>
__global__ void SAGWdUpdateAllAndClear<float, float, half>(int N,
    float* g, float *w, half* h,
    float momentum, float rate,  float decay, bool clear_grads) {
  float m1 = 1.F - momentum;
  CUDA_KERNEL_LOOP(i, N) {
    float wf = w[i];
    float hf = __half2float(h[i]);
    hf = momentum * hf + m1 * g[i];
    wf -= rate * (hf + decay * wf);
    w[i] = wf;
    h[i] = float2half_clip(hf);
    if (clear_grads) {
      g[i] = 0.F;
    }
  }
}

template<>
__global__ void SAGWdUpdateAllAndClear<half, float, float>(int N,
    half* g, float *w, float* h,
    float momentum, float rate,  float decay, bool clear_grads) {
  float m1 = 1.F - momentum;
  half hz;
  CUDA_KERNEL_LOOP(i, N) {
    float wf = w[i];
    float hf = momentum * h[i] + m1 * __half2float(g[i]);
    wf -=  rate * (hf + decay * wf);
    w[i] = wf;
    h[i] = hf;
    if (clear_grads) {
      g[i] = hz;
    }
  }
}

#pragma clang diagnostic pop

template<typename Gtype, typename Wtype, typename Htype>
void SAG_reg_update_and_clear_gpu(int N,
  Gtype* g, Wtype *w,  Htype* h,
  float momentum, float rate, const std::string& reg_type, float decay,
  void *handle, bool clear_grads) {
  cublasHandle_t cublas_handle =
      handle == nullptr ? Caffe::cublas_handle(0) : reinterpret_cast<cublasHandle_t>(handle);
  cudaStream_t stream;
  CUBLAS_CHECK(cublasGetStream(cublas_handle, &stream));
  if (reg_type == "L2") {
    // NOLINT_NEXT_LINE(whitespace/operators)
    SAGRegUpdateAllAndClear  // NOLINT_NEXT_LINE(whitespace/operators)
         <<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0, stream>>>(N,
           g, w, h, momentum, rate, decay, clear_grads);
  } else if (reg_type == "WD") {
    // NOLINT_NEXT_LINE(whitespace/operators)
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
void SAG_reg_update_and_clear_gpu<float16, float16, float16>(int N,
  float16* g, float16 *w,  float16* h,
  float momentum, float rate, const std::string& reg_type, float decay,
  void *handle, bool clear_grads) {
  cublasHandle_t cublas_handle =
      handle == nullptr ? Caffe::cublas_handle(0) : reinterpret_cast<cublasHandle_t>(handle);
  cudaStream_t stream;
  CUBLAS_CHECK(cublasGetStream(cublas_handle, &stream));
  if (reg_type == "L2") {
    SAGRegUpdateAllAndClear  // NOLINT_NEXT_LINE(whitespace/operators)
      <<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0, stream>>>(N,
          reinterpret_cast<half*>(g), reinterpret_cast<half*>(w), reinterpret_cast<half*>(h),
          momentum, rate, decay, clear_grads);
  } else { //if (reg_type == "WD") {
    SAGWdUpdateAllAndClear  // NOLINT_NEXT_LINE(whitespace/operators)
      <<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0, stream>>>(N,
          reinterpret_cast<half*>(g), reinterpret_cast<half*>(w), reinterpret_cast<half*>(h),
          momentum, rate, decay, clear_grads);
  }
  CUDA_POST_KERNEL_CHECK;
  CUDA_CHECK(cudaStreamSynchronize(stream));
}


template<>
void SAG_reg_update_and_clear_gpu<float, float, float16>(int N,
  float* g, float *w,  float16* h,
  float momentum, float rate, const std::string& reg_type, float decay,
  void *handle, bool clear_grads) {
  cublasHandle_t cublas_handle =
      handle == nullptr ? Caffe::cublas_handle(0) : reinterpret_cast<cublasHandle_t>(handle);
  cudaStream_t stream;
  CUBLAS_CHECK(cublasGetStream(cublas_handle, &stream));
  if (reg_type == "L2") {
    SAGRegUpdateAllAndClear  // NOLINT_NEXT_LINE(whitespace/operators)
      <<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0, stream>>>(N,
          g,w, reinterpret_cast<half*>(h),
          momentum, rate, decay, clear_grads);
  } else { //if (reg_type == "WD") {
    SAGWdUpdateAllAndClear  // NOLINT_NEXT_LINE(whitespace/operators)
      <<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0, stream>>>(N,
          g,w, reinterpret_cast<half*>(h),
          momentum, rate, decay, clear_grads);
  }
  CUDA_POST_KERNEL_CHECK;
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

template<>
void SAG_reg_update_and_clear_gpu<float16, float, float>(int N,
    float16* g, float *w,  float* h,
  float momentum, float rate, const std::string& reg_type, float decay,
  void *handle, bool clear_grads) {
  cublasHandle_t cublas_handle =
      handle == nullptr ? Caffe::cublas_handle(0) : reinterpret_cast<cublasHandle_t>(handle);
  cudaStream_t stream;
  CUBLAS_CHECK(cublasGetStream(cublas_handle, &stream));
  if (reg_type == "L2") {
    SAGRegUpdateAllAndClear  // NOLINT_NEXT_LINE(whitespace/operators)
      <<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0, stream>>>(N,
          reinterpret_cast<half*>(g), w, h,
          momentum, rate, decay, clear_grads);
  } else { //if (reg_type == "WD") {
    SAGWdUpdateAllAndClear  // NOLINT_NEXT_LINE(whitespace/operators)
      <<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0, stream>>>(N,
          reinterpret_cast<half*>(g), w, h,
          momentum, rate, decay, clear_grads);
  }
  CUDA_POST_KERNEL_CHECK;
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

template void SAG_reg_update_and_clear_gpu<float,float, float>(int,
    float*, float*, float*,
    float, float, const std::string&, float, void*, bool);
template void SAG_reg_update_and_clear_gpu<double, double, double>(int,
    double*, double*, double*,
    float, float, const std::string&, float, void*, bool);

template void SAG_reg_update_and_clear_gpu<float,float16, float16>(int,
    float*, float16*, float16*,
    float, float, const std::string&, float, void*, bool);
template void SAG_reg_update_and_clear_gpu<double,float16, float16>(int,
    double*, float16*, float16*,
    float, float, const std::string&, float, void*, bool);
template void SAG_reg_update_and_clear_gpu<float,float, double>(int,
    float*, float*, double*,
    float, float, const std::string&, float, void*, bool);
template void SAG_reg_update_and_clear_gpu<double,float, float>(int,
    double*, float*, float*,
    float, float, const std::string&, float, void*, bool);
template void SAG_reg_update_and_clear_gpu<float16, double, double>(int,
    float16*, double*, double*,
    float, float, const std::string&, float, void*, bool);
template void SAG_reg_update_and_clear_gpu<float, double, double>(int,
    float*, double*, double*,
    float, float, const std::string&, float, void*, bool);



}  // namespace caffe
