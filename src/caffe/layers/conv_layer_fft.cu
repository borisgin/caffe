#ifdef USE_FFT

#include <algorithm>
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/fft.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

  template <typename Dtype>
  void ConvolutionLayerFFT<Dtype>::fft_gpu_setup() {
    if (fft_gpu_initialized_)
    fft_gpu_clean();
    fft_gpu_initialized_ = false;
    // Evaluate memory needed for buffers
    num_weights = num_output_ * (channels_ / group_);
    map_out_size_ = height_out_ * width_out_;
    bottom_dim_2_ = num_ * channels_;
    top_dim_2_ = num_ * num_output_;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&fft_gpu_weights_complex_),
            num_weights * fft_map_complex_size_ * sizeof(std::complex<Dtype>)));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&fft_gpu_weights_complex_t),
            num_weights * fft_map_complex_size_ * sizeof(std::complex<Dtype>)));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&fft_gpu_map_in_real_),
            fft_map_real_size_ * sizeof(Dtype)));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(
            &fft_gpu_multi_map_bottom_complex_), bottom_dim_2_ *
            fft_map_complex_size_ * sizeof(std::complex<Dtype>)));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(
            &fft_gpu_multi_map_bottom_complex_t), bottom_dim_2_ *
            fft_map_complex_size_ * sizeof(std::complex<Dtype>)));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&fft_gpu_map_in_complex_),
            fft_map_complex_size_ * sizeof(std::complex<Dtype>)));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&fft_gpu_map_out_complex_),
            std::max(num_output_, channels_) * fft_map_complex_size_ *
            sizeof(std::complex<Dtype>)));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(
            &fft_gpu_multi_map_top_complex_t), top_dim_2_ *
            fft_map_complex_size_ * sizeof(std::complex<Dtype>)));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(
            &fft_gpu_multi_map_top_complex_), top_dim_2_ *
            fft_map_complex_size_ * sizeof(std::complex<Dtype>)));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&fft_gpu_map_out_real_),
            std::max(num_output_, channels_) * fft_map_real_size_ *
            sizeof(Dtype)));

    int n[2] = {fft_height_, fft_width_};
    int inembed[] = {fft_height_, 2 * (fft_width_/2 + 1)};
    int in_size = fft_height_ * 2 * (fft_width_/2 + 1);
    int onembed[] = {fft_height_, (fft_width_/2 + 1)};

    // cufft plans
    if (sizeof(Dtype) == sizeof(float)) {
      CUFFT_CHECK(cufftPlan2d(&fft_gpu_handle_, fft_height_, fft_width_,
              CUFFT_R2C));
      CUFFT_CHECK(cufftPlan2d(&ifft_gpu_handle_, fft_height_, fft_width_,
              CUFFT_C2R));
      CUFFT_CHECK(cufftCreate(&fft_gpu_many_weights_handle_));
      CUFFT_CHECK(cufftPlanMany(&fft_gpu_many_weights_handle_, 2, n, inembed,
              1, in_size, onembed, 1, fft_map_complex_size_, CUFFT_R2C,
              num_weights));
    } else if (sizeof(Dtype) == sizeof(double)) {
      CUFFT_CHECK(cufftPlan2d(&fft_gpu_handle_, fft_height_, fft_width_,
              CUFFT_D2Z));
      CUFFT_CHECK(cufftPlan2d(&ifft_gpu_handle_, fft_height_, fft_width_,
              CUFFT_Z2D));
      CUFFT_CHECK(cufftCreate(&fft_gpu_many_weights_handle_));
      CUFFT_CHECK(cufftPlanMany(&fft_gpu_many_weights_handle_, 2, n, inembed,
              1, in_size, onembed, 1, fft_map_complex_size_, CUFFT_D2Z,
              num_weights));
    }
    fft_gpu_initialized_ = true;
  }

  template <typename Dtype>
  void ConvolutionLayerFFT<Dtype>::fft_gpu_clean() {
    if (fft_gpu_initialized_) {
      cudaFree(fft_gpu_weights_complex_);
      cudaFree(fft_gpu_weights_complex_t);
      cudaFree(fft_gpu_map_in_real_);
      cudaFree(fft_gpu_map_in_complex_);
      cudaFree(fft_gpu_multi_map_bottom_complex_);
      cudaFree(fft_gpu_multi_map_bottom_complex_t);
      cudaFree(fft_gpu_map_out_complex_);
      cudaFree(fft_gpu_multi_map_top_complex_);
      cudaFree(fft_gpu_multi_map_top_complex_t);
      cudaFree(fft_gpu_map_out_real_);
      cufftDestroy(fft_gpu_handle_);
      cufftDestroy(ifft_gpu_handle_);
      cufftDestroy(fft_gpu_many_weights_handle_);
    }
    fft_gpu_initialized_ = false;
  }

//=====================Forward GPU========================

  template <typename Dtype>
  void ConvolutionLayerFFT<Dtype>::fft_gpu_compute_weights() {
    const Dtype *weight = this->blobs_[0]->gpu_data();
    // 0-padding of weights before FFT ----------------------
    caffe_gpu_memset((num_weights * fft_map_complex_size_
            * sizeof(std::complex<Dtype>)), 0., fft_gpu_weights_complex_);

    // Copy weights 2 buffer----------------------------------
    fft_gpu_copy2buffer(reinterpret_cast<Dtype*>(fft_gpu_weights_complex_),
        weight, num_output_, group_, channels_, kernel_h_, kernel_w_,
        fft_height_, fft_width_);

    // FFT of weights in place ------------------------------
    caffe_gpu_fft_execute_dft_r2c_inplace(fft_gpu_many_weights_handle_,
        fft_gpu_weights_complex_);
    std::complex<Dtype> alpha(1.0, 0);
    std::complex<Dtype> beta(0.0, 0);
    // transpose the weights
    caffe_gpu_c_geam(CblasTrans, CblasNoTrans, num_weights,
        fft_map_complex_size_, &alpha, fft_gpu_weights_complex_, &beta,
        (std::complex<Dtype> *)NULL, fft_gpu_weights_complex_t);
  }

  template <typename Dtype>
  void ConvolutionLayerFFT<Dtype>::Forward_gpu_fft_task(
      const Dtype* bottom_data, Dtype* top_data) {
    // loop over all channels ---------------------
    int map_in_size = height_* width_;
    std::complex<Dtype>* fft_gpu_in_complex_;
    std::complex<Dtype>* weights_complex;
    std::complex<Dtype>* map_out_complex;
    Dtype* map_out_real;
    Dtype* map_out;
    int gc_ = channels_ / group_;  // channel_index inside group
    int go_ = num_output_ / group_;
    std::complex<Dtype> alpha(1.0, 0);
    std::complex<Dtype> beta(0.0, 0);
    // GPU FFT
    for (int n = 0; n < num_; ++n) {
      for (int c = 0; c < channels_; c++) {
        const Dtype* map_in = bottom_data + (n * channels_ + c) * map_in_size;
        fft_gpu_in_complex_ = fft_gpu_multi_map_bottom_complex_ +
        (n * channels_ + c) * fft_map_complex_size_;
        caffe_gpu_memset((fft_map_complex_size_ * sizeof(std::complex<Dtype>)),
            0., fft_gpu_in_complex_);
        //  0-padding: map_in --> fft_map_in_real -------------
        caffe_gpu_memset((fft_map_real_size_ * sizeof(Dtype)), 0.,
            fft_gpu_map_in_real_);
        fft_gpu_copy2buffer2D_in(reinterpret_cast<Dtype*>(fft_gpu_map_in_real_),
            map_in, fft_width_, height_, width_, 1, 1, pad_h_, pad_w_ ,
            (Dtype)1.0);
        //  FFT: map_in_real --> map_in_complex
        caffe_gpu_fft_execute_dft_r2c(fft_gpu_handle_, fft_gpu_map_in_real_,
            fft_gpu_in_complex_);
      }
    }

    caffe_gpu_c_geam(CblasTrans, CblasNoTrans, bottom_dim_2_,
        fft_map_complex_size_, &alpha, fft_gpu_multi_map_bottom_complex_, &beta,
        (std::complex<Dtype> *)NULL, fft_gpu_multi_map_bottom_complex_t);
    for (int i = 0; i < fft_map_complex_size_; i++) {
      for (int g = 0; g < group_; g++) {
        std::complex<Dtype>* in_t_row = fft_gpu_multi_map_bottom_complex_t +
        i * bottom_dim_2_ + g * gc_;
        std::complex<Dtype>* w_t_row = fft_gpu_weights_complex_t +
        i * num_weights + g * gc_;
        std::complex<Dtype>* out_t_row = fft_gpu_multi_map_top_complex_t
        + i * top_dim_2_ + g * go_;

        caffe_gpu_c_gemm(CblasConjTrans, CblasNoTrans, go_, num_, gc_,
            alpha, w_t_row, channels_, in_t_row, channels_, beta,
            out_t_row, num_output_);
      }
    }
    caffe_gpu_c_geam(CblasTrans, CblasNoTrans, fft_map_complex_size_,
        top_dim_2_, &alpha, fft_gpu_multi_map_top_complex_t, &beta,
        (std::complex<Dtype> *)NULL, fft_gpu_multi_map_top_complex_);

    //  IFFT: map_out_complex --> map_out_real
    Dtype ifft_scale = 1./((Dtype)fft_map_real_size_);
    for (int n = 0; n < num_; ++n) {
      for (int out = 0; out < num_output_; out++) {
        map_out_complex = fft_gpu_multi_map_top_complex_ +
        (n * num_output_ + out) * fft_map_complex_size_;
        caffe_gpu_fft_execute_dft_c2r(ifft_gpu_handle_,
            map_out_complex, fft_gpu_map_out_real_);
        //  post-process: map_out_real --> map_out
        map_out = top_data + (n * num_output_ + out) * map_out_size_;

        fft_gpu_copy2buffer2D_out(map_out , fft_gpu_map_out_real_, height_out_,
            width_out_, fft_height_, fft_width_, stride_h_, stride_w_ , 0, 0,
            ifft_scale);
      }
    }
    //  bias
    if (bias_term_) {
      for (int n = 0; n < num_; ++n) {
        int top_offset_ = n * (num_output_ * height_out_* width_out_);
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
            N_, 1, (Dtype)1., this->blobs_[1]->gpu_data(),
            bias_multiplier_.gpu_data(), (Dtype)1., top_data + top_offset_);
      }
    }
  }

  template <typename Dtype>
  Dtype ConvolutionLayerFFT<Dtype>::Forward_gpu_fft(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    if (!fft_gpu_initialized_)
    fft_setup();
    fft_gpu_compute_weights();

    for (int i = 0; i < bottom.size(); ++i) {
      const Dtype* bottom_data = bottom[i]->gpu_data();
      Dtype* top_data = top[i]->mutable_gpu_data();
      const Dtype* weight = this->blobs_[0]->gpu_data();
      Forward_gpu_fft_task(bottom_data, top_data);
    }
    return Dtype(0.);
  }

  template <typename Dtype>
  void ConvolutionLayerFFT<Dtype>::Forward_gpu_task(
      const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top, int i, int n) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* top_data = top[i]->mutable_gpu_data();
    const Dtype* weight = this->blobs_[0]->gpu_data();
    int weight_offset = M_ * K_;
    int col_offset = K_ * N_;
    int top_offset = M_ * N_;
    Dtype* col_buff = NULL;
    if (!is_1x1_) {
      col_buff = col_buffer_.mutable_gpu_data();
    }
    // im2col transformation: unroll input regions for filtering
    // into column matrix for multiplication.
    if (!is_1x1_) {
      im2col_gpu(bottom_data + bottom[i]->offset(n), channels_, height_,
          width_, kernel_h_, kernel_w_, pad_h_, pad_w_, stride_h_, stride_w_,
          col_buff);
    } else {
      col_buff = bottom[i]->mutable_gpu_data() + bottom[i]->offset(n);
    }
    // Take inner products for groups.
    for (int g = 0; g < group_; ++g) {
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, K_,
          (Dtype)1., weight + weight_offset * g, col_buff + col_offset * g,
          (Dtype)0., top_data + top[i]->offset(n) + top_offset * g);
    }
    // Add bias
    if (bias_term_) {
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
          N_, 1, (Dtype)1., this->blobs_[1]->gpu_data(),
          bias_multiplier_.gpu_data(),
          (Dtype)1., top_data + top[i]->offset(n));
    }
  }

  template <typename Dtype>
  void ConvolutionLayerFFT<Dtype>::Forward_gpu(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    if (fft_on_) {
      Forward_gpu_fft(bottom, top);
    } else {
      for (int i = 0; i < bottom.size(); ++i) {
        for (int n = 0; n < num_; ++n) {
          Forward_gpu_task(bottom, top, i, n);
        }
      }
    }
  }

//===========================BACKWARD GPU============================

  template <typename Dtype>
  void ConvolutionLayerFFT<Dtype>::Backward_gpu_fft_task(
      Dtype* bottom_diff, const Dtype* top_diff) {
    int map_in_size = height_out_* width_out_;
    int gc_ = channels_ / group_;
    int go_ = num_output_ / group_;
    std::complex<Dtype> alpha(1.0, 0);
    std::complex<Dtype> beta(0.0, 0);
    const Dtype* map_in;

    for (int n = 0; n< num_; ++n) {
      for (int out = 0; out < num_output_; out++) {
        map_in = top_diff + (n * num_output_ + out) * map_in_size;
        caffe_gpu_memset((fft_map_real_size_ * sizeof(Dtype)), 0.,
            fft_gpu_map_in_real_);
        //  0-padding: map_out --> fft_map_in_real -------------
        fft_gpu_copy2buffer2D_in(reinterpret_cast<Dtype*>(
                fft_gpu_map_in_real_),
            map_in, fft_width_, height_out_, width_out_,
            stride_h_, stride_w_, 0, 0, (Dtype)1.0);
        std::complex<Dtype>* fft_gpu_map_in_complex_ =
        fft_gpu_multi_map_top_complex_ +
        (n * num_output_ + out) * fft_map_complex_size_;
        //  FFT: map_in_real --> map_in_complex
        caffe_gpu_fft_execute_dft_r2c(fft_gpu_handle_, fft_gpu_map_in_real_,
            fft_gpu_map_in_complex_);
      }
    }

    complex<Dtype>* null_ptr = NULL;
    caffe_gpu_c_geam(CblasTrans, CblasNoTrans, top_dim_2_,
        fft_map_complex_size_, &alpha, fft_gpu_multi_map_top_complex_,
        &beta, null_ptr, fft_gpu_multi_map_top_complex_t);
    for (int i = 0; i < fft_map_complex_size_; i++) {
      for (int g = 0; g < group_; g++) {
        std::complex<Dtype>* in_t_row = fft_gpu_multi_map_top_complex_t +
        i * top_dim_2_ + g * go_;
        std::complex<Dtype>* w_t_row = fft_gpu_weights_complex_t +
        i * num_weights + g * gc_;
        std::complex<Dtype>* out_t_row = fft_gpu_multi_map_bottom_complex_t +
        i * bottom_dim_2_ + g * gc_;
        caffe_gpu_c_gemm(CblasNoTrans, CblasNoTrans, gc_, num_, go_,
            alpha, w_t_row, channels_, in_t_row , num_output_, beta,
            out_t_row, channels_);
      }
    }
    // back transpose map_out_complex
    caffe_gpu_c_geam(CblasTrans, CblasNoTrans, fft_map_complex_size_,
        bottom_dim_2_ , &alpha, fft_gpu_multi_map_bottom_complex_t, &beta,
        null_ptr, fft_gpu_multi_map_bottom_complex_);

    //  IFFT: map_out_complex --> map_out_real
    //  Option: fft_execute (many) ?
    Dtype ifft_scale = 1./((Dtype) fft_map_real_size_);
    Dtype* map_out_real;
    Dtype* map_out;
    for (int n = 0; n < num_; ++n) {
      for (int c = 0; c < channels_; c++) {
        map_out = bottom_diff + (n * channels_ + c) * map_size_;
        std::complex<Dtype>* map_out_complex =
        fft_gpu_multi_map_bottom_complex_ +
        (n * channels_ + c) * fft_map_complex_size_;
        map_out_real = fft_gpu_map_out_real_;
        caffe_gpu_fft_execute_dft_c2r(ifft_gpu_handle_, map_out_complex,
            map_out_real);
        fft_gpu_copy2buffer2D_out(map_out, map_out_real, height_, width_,
            fft_height_, fft_width_, 1, 1, pad_h_, pad_w_, ifft_scale);
      }
    }
  }

  template <typename Dtype>
  void ConvolutionLayerFFT<Dtype>::Backward_gpu_bottom_diff_task(
      const Dtype* top_diff, Dtype* bottom_diff,
      const Dtype* weight, int i, int n) {
    int weight_offset = M_ * K_;
    int col_offset = K_ * N_;
    int top_offset = M_ * N_;
    Dtype* col_buff = NULL;
    int bottom_offset = channels_ * height_ * width_;
    int top_offset_n = n * (num_output_ * height_out_ * width_out_);

    // gradient w.r.t. bottom data, if necessary
    if (!is_1x1_) {
      col_buff = col_buffer_.mutable_gpu_data();
    } else {
      col_buff = bottom_diff + n*bottom_offset;
    }
    for (int g = 0; g < group_; ++g) {
      caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, K_, N_, M_,
          (Dtype)1., weight + weight_offset * g,
          top_diff + top_offset_n + top_offset * g,
          (Dtype)0., col_buff + col_offset * g);
    }
    // col2im back to the data
    if (!is_1x1_) {
      col2im_gpu(col_buff, channels_, height_, width_,
          kernel_h_, kernel_w_, pad_h_, pad_w_, stride_h_, stride_w_,
          bottom_diff + bottom_offset*n);
    }
  }

  template <typename Dtype>
  void ConvolutionLayerFFT<Dtype>::Backward_gpu_weight_diff_task(
      const Dtype* top_diff, const vector<Blob<Dtype>*>& bottom,
      int i, int n) {
    int weight_offset = M_ * K_;
    int col_offset = K_ * N_;
    int top_offset = M_ * N_;
    int top_offset_n = n* (num_output_ * height_out_* width_out_);
    Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
    Dtype* col_buff = NULL;
    if (!is_1x1_) {
      col_buff = col_buffer_.mutable_gpu_data();
    }
    const Dtype* bottom_data = bottom[i]->gpu_data();
    // recompute im2col
    if (!is_1x1_) {
      im2col_gpu(bottom_data + bottom[i]->offset(n), channels_, height_,
          width_, kernel_h_, kernel_w_, pad_h_, pad_w_,
          stride_h_, stride_w_, col_buff);
    } else {
      col_buff = bottom[i]->mutable_gpu_data() + bottom[i]->offset(n);
    }
    // gradient w.r.t. weight. Note that we will accumulate diffs.
    if (this->param_propagate_down_[0]) {
      for (int g = 0; g < group_; ++g) {
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, K_, N_,
            (Dtype)1., top_diff + top_offset_n + top_offset * g,
            col_buff + col_offset * g, (Dtype)1.,
            weight_diff+ weight_offset * g);
      }
    }
  }

  template <typename Dtype>
  void ConvolutionLayerFFT<Dtype>::Backward_gpu(
      const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
      const vector<Blob<Dtype>*>& bottom) {
    const Dtype* weight = NULL;
    Dtype* weight_diff = NULL;

    if (this->param_propagate_down_[0]) {
      weight = this->blobs_[0]->gpu_data();
      weight_diff = this->blobs_[0]->mutable_gpu_diff();
      caffe_gpu_set(this->blobs_[0]->count(), Dtype(0), weight_diff);
    }
    Dtype* bias_diff = NULL;
    if (bias_term_ && this->param_propagate_down_[1]) {
      bias_diff = this->blobs_[1]->mutable_gpu_diff();
      caffe_gpu_set(this->blobs_[1]->count(), Dtype(0), bias_diff);
    }

    for (int i = 0; i < top.size(); ++i) {
      const Dtype* top_diff = NULL;
      // Bias gradient, if necessary.
      if (bias_term_ && this->param_propagate_down_[1]) {
        top_diff = top[i]->gpu_diff();
        for (int n = 0; n < num_; ++n) {
          caffe_gpu_gemv<Dtype>(CblasNoTrans, num_output_, N_,
              1., top_diff + top[0]->offset(n),
              bias_multiplier_.gpu_data(), 1.,
              bias_diff);
        }
      }
      if (this->param_propagate_down_[0] || propagate_down[i]) {
        if (!top_diff) {
          top_diff = top[i]->gpu_diff();
        }
        const Dtype* bottom_data = bottom[i]->gpu_data();
        Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
        for (int n = 0; n < num_; ++n) {
          // Since we saved memory in the forward pass by not storing all col
          // data, we will need to recompute them.
          Backward_gpu_weight_diff_task(top_diff, bottom, i, n);
          // gradient w.r.t. bottom data, if necessary
        }
        if (propagate_down[i]) {
          if (weight == NULL) {
            weight = this->blobs_[0]->gpu_data();
          }

          if (fft_on_ && (!fft_gpu_initialized_))
          fft_setup();
          // WARNING: Assume fft for weights was computed in Forward
          // This assumption can fail in runtest, if only Backward() is
          // called!
          if ((fft_on_) && (!is_1x1_)) {
            Backward_gpu_fft_task(bottom_diff, top_diff);
          } else {
            for (int n = 0; n < num_; ++n) {
              Backward_gpu_bottom_diff_task(
                  top_diff, bottom_diff, weight, i, n);
            }
          }
        }
      }
    }
  }

// float instantiation
  template
  void ConvolutionLayerFFT<float>::fft_gpu_setup();
  template
  void ConvolutionLayerFFT<float>::fft_gpu_clean();
  template
  float ConvolutionLayerFFT<float>::Forward_gpu_fft(
      const vector<Blob<float>*>& bottom, const vector<Blob<float>*>& top);
  template
  void ConvolutionLayerFFT<float>::Forward_gpu_fft_task(
      const float *bottom_data, float* top_data);
  template
  void ConvolutionLayerFFT<float>::Forward_gpu_task(
      const vector<Blob<float>*>& bottom, const vector<Blob<float>*>& top,
      int i, int n);
  template
  void ConvolutionLayerFFT<float>::fft_gpu_compute_weights();
  template
  void ConvolutionLayerFFT<float>::Forward_gpu(
      const vector<Blob<float>*>& bottom, const vector<Blob<float>*>& top);
  template
  void ConvolutionLayerFFT<float>::Backward_gpu_fft_task(
      float* bottom_diff, const float* top_diff);

  template
  void ConvolutionLayerFFT<float>::Backward_gpu_weight_diff_task(
      const float* top_diff, const vector<Blob<float>*>& bottom,
      int i, int n);
  template
  void ConvolutionLayerFFT<float>::Backward_gpu_bottom_diff_task(
      const float* top_diff, float* bottom_diff,
      const float* weight, int i, int n);
  template
  void ConvolutionLayerFFT<float>::Backward_gpu(
      const vector<Blob<float>*>& top, const vector<bool>& propagate_down,
      const vector<Blob<float>*>& bottom);

// double instantiation
  template
  void ConvolutionLayerFFT<double>::fft_gpu_setup();
  template
  void ConvolutionLayerFFT<double>::fft_gpu_clean();
  template
  double ConvolutionLayerFFT<double>::Forward_gpu_fft(
      const vector<Blob<double>*>& bottom,
      const vector<Blob<double>*>& top);
  template
  void ConvolutionLayerFFT<double>::Forward_gpu_fft_task(
      const double *bottom_data, double* top_data);
  template
  void ConvolutionLayerFFT<double>::Forward_gpu_task(
      const vector<Blob<double>*>& bottom,
      const vector<Blob<double>*>& top, int i, int n);
  template
  void ConvolutionLayerFFT<double>::fft_gpu_compute_weights();
  template
  void ConvolutionLayerFFT<double>::Forward_gpu(
      const vector<Blob<double>*>& bottom,
      const vector<Blob<double>*>& top);
  template
  void ConvolutionLayerFFT<double>::Backward_gpu_fft_task(
      double* bottom_diff, const double* top_diff);
  template
  void ConvolutionLayerFFT<double>::Backward_gpu_weight_diff_task(
      const double* top_diff, const vector<Blob<double>*>& bottom,
      int i, int n);
  template
  void ConvolutionLayerFFT<double>::Backward_gpu_bottom_diff_task(
      const double* top_diff, double* bottom_diff,
      const double* weight, int i, int n);
  template
  void ConvolutionLayerFFT<double>::Backward_gpu(
      const vector<Blob<double>*>& top, const vector<bool>& propagate_down,
      const vector<Blob<double>*>& bottom);

}  // namespace caffe
#endif  // USE_FFT
