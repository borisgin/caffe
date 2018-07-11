#include <algorithm>
#include <vector>

#include "caffe/sgd_solvers.hpp"

namespace caffe {



template<typename Dtype>
void SAG_reg_update_and_clear_gpu(int N,
    Dtype* g, Dtype* w,  Dtype* h,
    float momentum, float rate, const std::string& reg_type, float decay,
    void *handle, bool clear_grads);

template<typename Dtype>
float SAGSolver<Dtype>::ComputeUpdateValue(int param_id, void* handle, float rate,
    bool clear_grads) {
  if (this->param_.debug_info()) {
    SGDSolver<Dtype>::PrintParams(param_id);
  }
  Blob* param = this->net_->learnable_params()[param_id].get();
  TBlob<Dtype>* history = this->history_[param_id].get();

  float wgrad_sq = 0.F;
  float local_rate = SAGSolver<Dtype>::GetLocalRate(param_id, wgrad_sq);
  const bool larc = this->param_.larc();
  if (larc) {
    const string& larc_policy = this->param_.larc_policy();
    if (larc_policy == "scale") {
      local_rate = rate * local_rate;
    } else if (larc_policy == "clip") {
      local_rate = std::min(rate, local_rate);
    } else {
      LOG(FATAL) << "Unknown larc policy: " << larc_policy;
    }
  } else {
    local_rate = rate * local_rate;
  }

  float momentum = this->GetMomentum();
  float N=param->count();
  // Compute the update to history, then copy it to the parameter diff.

  if (Caffe::mode() == Caffe::CPU) {
    caffe_cpu_axpby<Dtype>(param->count(), local_rate, param->cpu_diff<Dtype>(), momentum,
        history->mutable_cpu_data());
    caffe_copy<Dtype>(param->count(), history->cpu_data(), param->mutable_cpu_diff<Dtype>());
    param->Update();
    if (clear_grads) {
      param->set_diff(0.F);
    }
  } else if (Caffe::mode() == Caffe::GPU) {
    const std::string& reg_type = this->param_.regularization_type();
    float decay = SGDSolver<Dtype>::local_decay(param_id);
    if (this->iter_ <= 1) {
      caffe_copy<Dtype>(N, param->gpu_diff<Dtype>(), history->mutable_gpu_data());
      if (reg_type == "L2"){
        caffe_gpu_axpy<Dtype>(N, Dtype(decay), param->gpu_data<Dtype>(),
            history->mutable_gpu_data());
      }
    }
    SAG_reg_update_and_clear_gpu<Dtype>(param->count(),
            param->mutable_gpu_diff<Dtype>(), param->mutable_gpu_data<Dtype>(),
            history->mutable_gpu_data(),
            momentum, local_rate, reg_type, decay, handle, clear_grads);
  } else {
    LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
  }
  return wgrad_sq;
}

template<typename Dtype>
float SAGSolver<Dtype>::GetLocalRate(int param_id, float& wgrad_sq)  {
  const vector<float>& net_params_lr = this->net_->params_lr();
  float local_lr = net_params_lr[param_id];

  if (this->net_->global_grad_scale_enabled() || this->param_.larc()) {
    shared_ptr<Blob> param = this->net_->learnable_params()[param_id];
    const int type_id = this->net_->learnable_types()[0] == param->diff_type() ? 0 : 1;
    wgrad_sq = param->sumsq_diff(type_id);
    if (std::isnan(wgrad_sq)) {
      wgrad_sq = 0.F;  // skip this

    }
    if (this->param_.larc()) {
      const float wgrad_norm = std::sqrt(wgrad_sq);
      const float w_norm = std::sqrt(param->sumsq_data(type_id));
      if (std::isnan(wgrad_sq)) {
        LOG(INFO) << "LARC: nan";
      }
      const float larc_eta = this->param_.larc_eta();
      float rate = 1.F;
      if (w_norm > 0.F && wgrad_norm > 0.F) {
        rate =  larc_eta * w_norm /wgrad_norm;
        // float momentum=this->GetMomentum();
        // rate = (1.0 - momentum)* larc_eta * w_norm /wgrad_norm ;
      }
      if (local_lr > 0.) {
        local_lr = rate;
      }
      if (this->param_.larc_turbo())  {
        TBlob<Dtype>* hist = this->history_[param_id].get();
        const int N = param->count();
        if (this->iter_ > 1) {
          float g_m_dot;
          caffe_gpu_dot<Dtype>(N,param->gpu_diff<Dtype>(),  hist->gpu_data(), &g_m_dot);
          float m_norm=std::sqrt(hist->sumsq_data(type_id));
          float g1_go_corr = 0;
          if ((wgrad_norm > 0.) && (m_norm > 0.)) {
             g1_go_corr = g_m_dot / (wgrad_norm *m_norm);
          }
          const float beta =  0.95;
          this->g_corr_[param_id]= beta * this->g_corr_[param_id] + (1.-beta) * g1_go_corr;
          float boost = 1.0F - 0.9F* this->g_corr_[param_id];
          local_lr = local_lr * boost;
        }
      }// end of turbo
      this->local_rates_[param_id] = local_lr;

//#ifdef DEBUG
          if (Caffe::root_solver() && this->param_.display()
               && (this->iter_ % this->param_.display() == 0)
               && (local_lr>0)) {
             //using namespace std;
             LOG(INFO) << std::setw(2) << param_id
                  << " lr="       << std::fixed << std::setprecision(6) << local_lr
                  << "  g_corr="  << std::fixed << std::setprecision(6) << this->g_corr_[param_id]
                  << "  w="       << w_norm
                  << "  g="       << wgrad_norm
                 ;
          }
//#endif

#ifdef DEBUG
        if (Caffe::root_solver()
            && this->param_.display()
            && (this->iter_ % this->param_.display() == 0)) {
          const int layer_id = this->net_->param_layer_indices(param_id).first;
          const string &layer_name = this->net_->layer_names()[layer_id];
          const int blob_id = this->net_->param_layer_indices(param_id).second;
          LOG(INFO) << layer_name << "." << blob_id << " lr=" << local_lr
                    << " " << "\t  w=" << w_norm << "\t g=" << wgrad_norm;
        }
#endif

    } // end of larc
  }
  return local_lr;
}

/*
template <typename Dtype>
float SAGSolver<Dtype>::ComputeUpdateValue(int param_id, void* handle, float rate,
    bool clear_grads) {

  const vector<shared_ptr<Blob>>& net_params = this->net_->learnable_params();
  shared_ptr<Blob> param = net_params[param_id];
  const int type_id = this->net_->learnable_types()[0] == param->diff_type() ? 0 : 1;
  const vector<float>& net_params_lr = this->net_->params_lr();
  rate = rate * net_params_lr[param_id];

  //const float decay = this->local_decay(param_id);
  const float decay = SGDSolver<Dtype>::GetWeightDecay();
  const std::string& reg_type = this->param_.regularization_type();

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
        rate = rate *(wdiff_norm / gdiff_norm);
      }
    }
    caffe_copy<Dtype>(N, param->cpu_data<Dtype>(), val_m->mutable_cpu_data());
    caffe_copy<Dtype>(N, param->cpu_diff<Dtype>(), val_m->mutable_cpu_diff());
    caffe_scal<Dtype>(N, (Dtype)rate, param->mutable_cpu_diff<Dtype>());
    param->Update();
    if (clear_grads) {
      param->set_diff(0.F);
    }
    // end of CPU part
  } else if (Caffe::mode() == Caffe::GPU) {

//    if (decay > 0.) {
//      caffe_gpu_axpby<Dtype>(N, (Dtype)decay, param->gpu_data<Dtype>(),
//                                (Dtype)1.F,  param->mutable_gpu_diff<Dtype>());
//    }
    if (this ->iter_ == 0) {
      local_rates_[param_id]= -1.;
//      caffe_copy<Dtype>(N, param->gpu_data<Dtype>(), val_m->mutable_gpu_data());
//      caffe_copy<Dtype>(N, param->gpu_diff<Dtype>(), val_m->mutable_gpu_diff());

    } else {
      float w1_norm = std::sqrt(param->sumsq_data(type_id));
      float g1_norm = std::sqrt(param->sumsq_diff(type_id));
      float w0_norm = std::sqrt(val_m->sumsq_data(type_id));
      float g0_norm = std::sqrt(val_m->sumsq_diff(type_id));

      float g1_go_dot;
      caffe_gpu_dot<Dtype>(N,param->gpu_diff<Dtype>(),  val_m->gpu_diff(), &g1_go_dot );
      float g1_go_corr= g1_go_dot / (g0_norm* g1_norm);

      caffe_gpu_sub<Dtype>(N, param->gpu_data<Dtype>(), val_m->gpu_data(),
          val_m->mutable_gpu_data());
      caffe_gpu_sub<Dtype>(N, param->gpu_diff<Dtype>(),  val_m->gpu_diff(),
          val_m->mutable_gpu_diff());

      float dw_norm = std::sqrt(val_m->sumsq_data(type_id));
      float dg_norm = std::sqrt(val_m->sumsq_diff(type_id));
      float lr;
      if ((dg_norm > 0) && (dw_norm >0)) {
        const float gw_ratio = this->param_.larc_eta();
        //lr = gw_ratio* dw_norm/ dg_norm;
        lr = 1. + gw_ratio* g1_go_corr;
//        if (Caffe::root_solver()
//            && this->param_.display() && (this->iter_ % this->param_.display() == 0)) {
//              //const int layer_id = this->net_->param_layer_indices(param_id).first;
//              //const string &layer_name = this->net_->layer_names()[layer_id];
//
//          LOG(INFO) << param_id << " lr=" << lr
//                     << " " << local_rates_[param_id];
////                     << " wdiff=" << dw_norm << " gdiff=" << dg_norm
////                     << " w1_norm=" << w1_norm   << " w0_norm=" << w0_norm
////                     << " g1_norm=" << g1_norm   << " g0_norm=" << g0_norm
////                     << " g1_go_corr=" << g1_go_corr;
//        }
        if (local_rates_[param_id] <= 0.) {
          local_rates_[param_id]= lr;
        } else {
          if (lr > 0.) {
            //local_rates_[param_id]= std::min( local_rates_[param_id], lr);
            const float beta =  this->param_.momentum();
            local_rates_[param_id]= beta * local_rates_[param_id] + (1.-beta) * lr;

            //local_rates_[param_id]=  local_rates_[param_id]  * lr;
            //local_rates_[param_id]=  std::min(local_rates_[param_id],10.F);
            //local_rates_[param_id]=  std::max(local_rates_[param_id],0.1F);
          }
        }
      }
      lr = local_rates_[param_id];
      if (Caffe::root_solver()
          && this->param_.display() && (this->iter_ % this->param_.display() == 0)) {
        LOG(INFO) << param_id << " lr=" << lr  ;
      }

//      if (lr > 0.) {
//        const string& larc_policy = this->param_.larc_policy();
//        if (larc_policy == "scale") {
//          rate =  rate * lr;
//        } else if (larc_policy == "clip") {
//          rate = std::min(rate, lr);
//        } else if (larc_policy == "auto") {
//          rate = lr ;
//        }
//      }

    }

//    caffe_copy<Dtype>(N, param->gpu_data<Dtype>(), val_m->mutable_gpu_data());
//    caffe_copy<Dtype>(N, param->gpu_diff<Dtype>(), val_m->mutable_gpu_diff());
//    if (decay > 0.) {
//      caffe_gpu_axpby<Dtype>(N, (Dtype)decay, param->gpu_data<Dtype>(),
//                                      (Dtype)1.F,  param->mutable_gpu_diff<Dtype>());
//    }
//    caffe_gpu_scal<Dtype>(N, (Dtype)rate, param->mutable_gpu_diff<Dtype>());
//    param->Update();
//    if (clear_grads) {
//      param->set_diff(0.F);
//    }
    SAG_reg_update_and_clear_gpu<Dtype>(N,
        param->mutable_gpu_diff<Dtype>(), param->mutable_gpu_data<Dtype>(),
        val_m->mutable_gpu_diff(), val_m->mutable_gpu_data(),
        rate,  reg_type , decay, handle, clear_grads);

  } else {
    LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
  }
  return rate;
}

*/



INSTANTIATE_CLASS(SAGSolver);

REGISTER_SOLVER_CLASS(SAG);

}  // namespace caffe
