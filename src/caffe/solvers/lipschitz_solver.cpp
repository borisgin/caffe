#include <algorithm>
#include <vector>

#include "caffe/sgd_solvers.hpp"

namespace caffe {

template<typename Dtype>
void LipschitzSolver<Dtype>::LipschitzPreSolve() {
  // Add the extra history entries for Lipschitz after those from
  // SGDSolver<Dtype>::PreSolve()

  LOG(INFO) << " Lipschitz presolve" ;
  const vector<shared_ptr<Blob>>& net_params = this->net_->learnable_params();

  for (int i = 0; i < net_params.size(); ++i) {
	  TBlob<Dtype>* h = this->history_[i].get();
	  h->set_data(0.);
	  h->set_diff(0.);
  }
  int N=net_params.size();
  local_rates_.resize(N);
}

template<typename Dtype>
void Lipschitz_update_and_clear_gpu(int N,
    Dtype* g, Dtype* w, Dtype* h_g, Dtype* h_w,
    float rate, void* handle, bool clear_grads);

template <typename Dtype>
float LipschitzSolver<Dtype>::ComputeUpdateValue(int param_id, void* handle, float rate,
    bool clear_grads) {

  const vector<shared_ptr<Blob>>& net_params = this->net_->learnable_params();
  shared_ptr<Blob> param = net_params[param_id];
  const int type_id = this->net_->learnable_types()[0] == param->diff_type() ? 0 : 1;
  const vector<float>& net_params_lr = this->net_->params_lr();
  float local_rate = rate * net_params_lr[param_id];
  const float decay = this->local_decay(param_id);
  const	float beta =  this->param_.momentum();

// const float momentum = this->param_.momentum();

  // alias for convenience
  TBlob<Dtype>* val_m = this->history_[param_id].get();
  const int N = param->count();

  if (Caffe::mode() == Caffe::CPU) {
    if (decay > 0.) {
    	caffe_cpu_axpby<Dtype>(N, (Dtype)decay, param->cpu_data<Dtype>(),
    			                  (Dtype)1.F,   param->mutable_cpu_diff<Dtype>());
    }

	caffe_sub<Dtype>(N, param->cpu_data<Dtype>(),  val_m->cpu_data(),
			            val_m->mutable_cpu_data());
	caffe_sub<Dtype>(N, param->cpu_diff<Dtype>(),  val_m->cpu_diff(),
			            val_m->mutable_cpu_diff());
    float wdiff_norm = std::sqrt(caffe_cpu_sumsq(N, val_m->mutable_cpu_data()));
    float gdiff_norm= std::sqrt(caffe_cpu_sumsq(N, val_m->mutable_cpu_diff()));
    if (std::isnan(gdiff_norm)) {
       caffe_set(N, (Dtype)0.F, param->mutable_cpu_diff<Dtype>());
    } else {
    	if ((gdiff_norm > 0) && (wdiff_norm >0)) {
    		local_rate = local_rate *(wdiff_norm / gdiff_norm);
    	}
    }

	caffe_copy<Dtype>(N, param->cpu_data<Dtype>(), val_m->mutable_cpu_data());
	caffe_copy<Dtype>(N, param->cpu_diff<Dtype>(), val_m->mutable_cpu_diff());

    caffe_scal<Dtype>(N, (Dtype)local_rate, param->mutable_cpu_diff<Dtype>());
    param->Update();
    if (clear_grads) {
      param->set_diff(0.F);
    }
  } else if (Caffe::mode() == Caffe::GPU) {
    if (decay > 0.) {
      caffe_gpu_axpby<Dtype>(N, (Dtype)decay, param->gpu_data<Dtype>(),
				                (Dtype)1.F,  param->mutable_gpu_diff<Dtype>());
	}

    caffe_gpu_sub<Dtype>(N, param->gpu_data<Dtype>(), val_m->gpu_data(),
    		val_m->mutable_gpu_data());
    caffe_gpu_sub<Dtype>(N, param->gpu_diff<Dtype>(),  val_m->gpu_diff(),
	        val_m->mutable_gpu_diff());
    float wdiff_sumsq = param->sumsq_data(type_id);
	float gdiff_sumsq = param->sumsq_diff(type_id);
    float wdiff_norm= std::sqrt(wdiff_sumsq);
	float gdiff_norm= std::sqrt(gdiff_sumsq);

	if ((gdiff_norm > 0) && (wdiff_norm >0)) {
	  float lr = wdiff_norm / gdiff_norm;
	  if (local_rates_[param_id]==0) {
	  	local_rates_[param_id]= lr;
	  } else {
		 local_rates_[param_id]= beta * local_rates_[param_id] + (1.-beta) * lr;
	  }
	  lr = std::min(lr, local_rates_[param_id]);
	  local_rate =local_rate*lr;
	}


    Lipschitz_update_and_clear_gpu<Dtype>(N,
		  param->mutable_gpu_diff<Dtype>(), param->mutable_gpu_data<Dtype>(),
		  val_m->mutable_gpu_data(),  val_m->mutable_gpu_diff(),
		  local_rate,  handle, clear_grads);
    /*
		  caffe_copy<Dtype>(N, param->gpu_data<Dtype>(), val_m->mutable_gpu_data());
		  caffe_copy<Dtype>(N, param->gpu_diff<Dtype>(), val_m->mutable_gpu_diff());
		  caffe_gpu_scal<Dtype>(N, (Dtype)local_rate, param->mutable_gpu_diff<Dtype>());
		  param->Update();
		  if (clear_grads) {
			param->set_diff(0.F);
		  }
	 */
  } else {
    LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
  }
  return local_rate;
}

INSTANTIATE_CLASS(LipschitzSolver);

REGISTER_SOLVER_CLASS(Lipschitz);

}  // namespace caffe
