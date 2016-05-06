#ifndef CAFFE_BATCHNORM_LAYER_HPP_
#define CAFFE_BATCHNORM_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Normalizes the input to have 0-mean and/or unit (1) variance across
 *        the batch.
 *
 * This layer computes Batch Normalization described in [1].  For
 * each channel in the data (i.e. axis 1), it subtracts the mean and divides
 * by the variance, where both statistics are computed across both spatial
 * dimensions and across the different examples in the batch.
 *
 * By default, during training time, the network is computing global mean/
 * variance statistics via a running average, which is then used at test
 * time to allow deterministic outputs for each input.  You can manually
 * toggle whether the network is accumulating or using the statistics via the
 * use_global_stats option.  IMPORTANT: for this feature to work, you MUST
 * set the learning rate to zero for all three parameter blobs, i.e.,
 * param {lr_mult: 0} three times in the layer definition.
 *
 * Note that the original paper also included a per-channel learned bias and
 * scaling factor.  It is possible (though a bit cumbersome) to implement
 * this in caffe using a single-channel DummyDataLayer filled with zeros,
 * followed by a Convolution layer with output the same size as the current.
 * This produces a channel-specific value that can be added or multiplied by
 * the BatchNorm layer's output.
 *
 * [1] S. Ioffe and C. Szegedy, "Batch Normalization: Accelerating Deep Network
 *     Training by Reducing Internal Covariate Shift." arXiv preprint
 *     arXiv:1502.03167 (2015).
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
class BatchNormLayer : public Layer<Dtype> {
 public:
  explicit BatchNormLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "BatchNorm"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

  static const int   BN_WARMUP_PERIOD = 200;
  static const Dtype BN_VARIANCE_CLIP_CONST = 4.0;

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
     const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  virtual void multicast_cpu(int N, int C, int S, const Dtype *x, Dtype *y);
  virtual void compute_sum_per_channel_cpu(int N, int C, int S,
      const Dtype *x, Dtype *y);
  virtual void compute_mean_per_channel_cpu(int N, int C, int S,
      const Dtype *x, Dtype *y);
#ifndef CPU_ONLY
  virtual void compute_sum_per_channel_gpu(int N, int C, int S,
      const Dtype *x, Dtype *y);
  virtual void multicast_gpu(int N, int C, int S, const Dtype *x, Dtype *y);
  virtual void compute_mean_per_channel_gpu(int N, int C, int S,
      const Dtype *x, Dtype *y);
#endif

  Blob<Dtype> batch_mean_, batch_variance_, mean_, variance_,
              inv_variance_, x_norm_;
  bool use_global_stats_, clip_variance_;
  Dtype moving_average_fraction_, eps_;
  int channels_, iter_;
  // arrays of ones
  Blob<Dtype> ones_N_, ones_HW_, ones_C_;  // ones[N], ones[H*W], ones[c]
  // auxiliary arrays
  Blob<Dtype> temp_;      // temp_[N,C,H,W]
  Blob<Dtype> temp_C_;    // temp_C_[C]
  Blob<Dtype> temp_NC_;   // temp_NC_[N,C]
};

}  // namespace caffe

#endif  // CAFFE_BATCHNORM_LAYER_HPP_
