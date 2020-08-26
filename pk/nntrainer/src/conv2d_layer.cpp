// SPDX-License-Identifier: Apache-2.0-only
/**
 * Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file	conv2d_layer.h
 * @date	02 June 2020
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 * @brief	This is Convolution Layer Class for Neural Network
 *
 */

#include <conv2d_layer.h>
#include <layer.h>
#include <lazy_tensor.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <parse_util.h>
#include <string>
#include <util_func.h>

namespace nntrainer {

int Conv2DLayer::initialize() {
  int status = ML_ERROR_NONE;

  if (input_dim.getDataLen() == 1) {
    ml_logw("Warning: the length of previous layer dimension is one");
  }

  TensorDim dim =
    TensorDim(1, input_dim.channel(), kernel_size[0], kernel_size[1]);
  TensorDim bias_dim = TensorDim(1, 1, 1, 1);

  std::string kernelPrefix = "Conv2d:filter";
  std::string biasPrefix = "Conv2d:bias";
  setParamSize(filter_size * 2);

  for (unsigned int i = 0; i < filter_size; ++i) {
    Tensor Knl = initializeWeight(dim, weight_initializer, status);
    NN_RETURN_STATUS();

    Tensor bias = initializeWeight(bias_dim, bias_initializer, status);
    NN_RETURN_STATUS();

    Tensor delK(dim);
    delK.setZero();

    Tensor delBias(bias_dim);
    delBias.setZero();

    /*< @note: order of weight and bias are:
               w0 w1 w2 ... w3
    */
    paramsAt(i) = {std::move(Knl), std::move(delK),
                   kernelPrefix + std::to_string(i)};
    paramsAt(i + filter_size) = {std::move(bias), std::move(delBias),
                                 biasPrefix + std::to_string(i)};
  }

  // this output_dim should be the same with dimension of hidden
  output_dim.batch(input_dim.batch());
  output_dim.channel(filter_size);
  output_dim.height(
    (input_dim.height() - kernel_size[0] + 2 * padding[0]) / stride[0] + 1);
  output_dim.width(
    (input_dim.width() - kernel_size[1] + 2 * padding[1]) / stride[1] + 1);

  return status;
}

void Conv2DLayer::read(std::ifstream &file) { Layer::read(file); }

void Conv2DLayer::save(std::ofstream &file) { Layer::save(file); }

sharedConstTensor Conv2DLayer::forwarding(sharedConstTensor in) {

  int status = ML_ERROR_NONE;
  input = *in;

  if (normalization) {
    input = input.normalization();
  }

  if (standardization) {
    input = input.standardization();
  }

  TensorDim hidden_dim = output_dim;
  hidden_dim.batch(in->batch());
  hidden = Tensor(hidden_dim);
  hidden.setZero();

  TensorDim kdim (filter_size, input_dim.channel(), kernel_size[0], kernel_size[1]);  

  std::vector<float> imKernel(kdim.getFeatureLen() * filter_size);
  
  for (unsigned int i = 0; i < filter_size; ++i) {
    Tensor &filters = paramsAt(i).weight;
    float *d = imKernel.data();
    memcpy(&d[i * kdim.getFeatureLen()], filters.getData(),
           kdim.getFeatureLen() * sizeof(float));
  }

  status = conv2d_gemm(imKernel.data(), kdim, input, output_dim);
  if (status != ML_ERROR_NONE)
    throw std::runtime_error("Forwarding Convolution failed.");

  for (unsigned int i = 0; i < filter_size; i++) {
    Tensor &bias = paramsAt(i + filter_size).weight;
    Tensor tmp(1, 1, hidden.height(), hidden.width());
    tmp.setValue(bias.getValue(0, 0, 0, 0));
    cblas_saxpy(hidden.height() * hidden.width(), 1, tmp.getData(), 1,
                hidden.getAddress(i * hidden.height() * hidden.width()), 1);
  }

  // TensorDim kdim(filter_size, input_dim.channel(), kernel_size[0],
  //                kernel_size[1]);

  // // std::vector<float> output(output_dim.width() * output_dim.height());
  // std::vector<float> inCol(kdim.getFeatureLen() * output_dim.width() *
  //                          output_dim.height());

  // std::vector<float> imKernel(kdim.getFeatureLen()*filter_size);

  // std::vector<float> out(filter_size * output_dim.width() *
  //                           output_dim.height());

  // for (unsigned int i = 0; i < filter_size; ++i) {
  //   Tensor &filters = paramsAt(i).weight;
  //   float * d = imKernel.data();
  //   memcpy(&d[i * kdim.getFeatureLen()], filters.getData(),
  // 	   kdim.getFeatureLen() * sizeof(float));
  // }

  // for (unsigned int b = 0; b < input.batch(); ++b) {
  //   Tensor in_padded = zero_pad(b, input, padding);
  //   status = im2col(in_padded, kdim, inCol.data());
  //   if (status != ML_ERROR_NONE)
  //     throw std::runtime_error("Forwarding Convolution failed.");

  //   float alpha_dgemm = 1.0f;
  //   float beta_dgemm = 0.0f;
  //   const float *data = imKernel.data();
  //   const float *mdata = inCol.data();
  //   float *rdata = out.data();
  //   memset(rdata, 0, out.size() * sizeof(out[0]));

  //   cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, filter_size,
  //               output_dim.width() * output_dim.height(), kdim.getFeatureLen(),
  //               alpha_dgemm, data, kdim.getFeatureLen(), mdata,
  //               output_dim.width() * output_dim.height(), beta_dgemm, rdata,
  //               output_dim.width() * output_dim.height());

  //   memcpy(hidden.getAddress(b * hidden.getDim().getFeatureLen()), rdata,
  //          out.size() * sizeof(float));

  //   for (unsigned int i = 0; i < filter_size; i++) {
  //     Tensor &bias = paramsAt(i + filter_size).weight;
  //     Tensor tmp(1, 1, hidden.height(), hidden.width());
  //     tmp.setValue(bias.getValue(0, 0, 0, 0));
  //     cblas_saxpy(hidden.height() * hidden.width(), 1, tmp.getData(), 1,
  //                 hidden.getAddress(i * hidden.height() * hidden.width()), 1);
  //   }

  // }
  

  loss = 0.0f;
  if (weight_decay.type == WeightDecayType::l2norm) {
    for (unsigned int i = 0; i < filter_size; ++i) {
      Tensor &weight = paramsAt(i).weight;
      loss += weight_decay.lambda * 0.5f * (weight.l2norm());
    }
    loss /= filter_size;
  }

  return MAKE_SHARED_TENSOR(hidden);  
};

sharedConstTensor Conv2DLayer::backwarding(sharedConstTensor derivative,
                                           int iteration) {

  // Calculate delK : [batch, channel, height, width ] * filter_size
  unsigned int same_pad[CONV2D_DIM];
  unsigned int o_size = kernel_size[0] * kernel_size[1];
  std::vector<float> output(o_size);

  for (unsigned int i = 0; i < filter_size; ++i) {
    Tensor &delK = paramsAt(i).grad;
    Tensor &delBias = paramsAt(i + filter_size).grad;
    delK.setZero();
    delBias.setZero();
  }

  TensorDim in_dim(1, 1, derivative->height(), derivative->width());

  for (unsigned int b = 0; b < input_dim.batch(); ++b) {
    Tensor in_padded = zero_pad(b, input, padding);
    TensorDim p_dim(1, 1, in_padded.height(), in_padded.width());

    for (unsigned int i = 0; i < filter_size; i++) {
      float sum = 0.0f;
      Tensor &delK = paramsAt(i).grad;
      Tensor &delBias = paramsAt(i + filter_size).grad;
      for (unsigned int j = 0; j < in_padded.channel(); ++j) {
        conv2d(in_padded.getAddress(j * in_padded.height() * in_padded.width()),
               p_dim,
               derivative->getAddress(b * derivative->getDim().getFeatureLen() +
                                      i * derivative->height() *
                                        derivative->width()),
               in_dim, output.data(), stride, 0.0f);
        float *del = delK.getAddress(j * o_size);
        for (unsigned k = 0; k < o_size; ++k) {
          del[k] += output[k];
        }
      }

      // Calculate delBias [ 1, 1, 1, filter_size]
      for (unsigned int j = 0; j < derivative->height(); ++j) {
        for (unsigned int k = 0; k < derivative->width(); ++k) {
          sum += derivative->getValue(b, i, j, k);
        }
      }
      delBias.setValue(0, 0, 0, 0, sum + delBias.getValue(0, 0, 0, 0));
    }
  }

  // Calculate delS : returns ( Full pad )

  Tensor ret(input_dim.batch(), input_dim.channel(),
             input_dim.height() + padding[0] * 2,
             input_dim.width() + padding[1] * 2);
  ret.setZero();

  same_pad[0] = kernel_size[0] - 1;
  same_pad[1] = kernel_size[1] - 1;

  TensorDim kdim(1, 1, kernel_size[0], kernel_size[1]);

  output.clear();
  output.resize(ret.height() * ret.width());

  for (unsigned int b = 0; b < derivative->batch(); ++b) {
    Tensor in_padded = zero_pad(b, *derivative, same_pad);
    TensorDim p_dim(1, 1, in_padded.height(), in_padded.width());

    for (unsigned int in_c = 0; in_c < input_dim.channel(); ++in_c) {
      for (unsigned int i = 0; i < derivative->channel(); ++i) {
        Tensor &filters = paramsAt(i).weight;

        conv2d(in_padded.getAddress(i * in_padded.height() * in_padded.width()),
               p_dim,
               filters.getAddress(in_c * kernel_size[0] * kernel_size[1]), kdim,
               output.data(), stride, 0.0f);
        float *ret_vec = ret.getAddress(b * ret.getDim().getFeatureLen() +
                                        in_c * ret.height() * ret.width());
        for (unsigned int j = 0; j < ret.height() * ret.width(); ++j) {
          ret_vec[j] += output[j];
        }
      }
    }
  }

  if (trainable) {
    //  Update K / bias
    for (unsigned int i = 0; i < filter_size; ++i) {
      Tensor &delK = paramsAt(i).grad;
      Tensor &filters = paramsAt(i).weight;

      delK = delK.chain()
               .applyIf(this->isWeightDecayL2Norm(), _LIFT(add_i), filters,
                        weight_decay.lambda)
               .run();
    }

    opt.apply_gradients(params, param_size, iteration);
  }

  return MAKE_SHARED_TENSOR(std::move(strip_pad(ret, padding)));
}

void Conv2DLayer::copy(std::shared_ptr<Layer> l) {
  Layer::copy(l);
  std::shared_ptr<Conv2DLayer> from = std::static_pointer_cast<Conv2DLayer>(l);
  this->filter_size = from->filter_size;
  for (unsigned int i = 0; i < CONV2D_DIM; ++i) {
    this->kernel_size[i] = from->kernel_size[i];
    this->stride[i] = from->stride[i];
    this->padding[i] = from->padding[i];
  }

  this->input.copy(from->input);
  this->hidden.copy(from->hidden);
  this->input_dim = from->input_dim;
  this->output_dim = from->output_dim;
}

int Conv2DLayer::setSize(int *size, PropertyType type) {
  int status = ML_ERROR_NONE;
  switch (type) {
  case PropertyType::kernel_size:
    for (int i = 0; i < CONV2D_DIM; ++i) {
      kernel_size[i] = size[i];
    }
    break;
  case PropertyType::stride:
    for (int i = 0; i < CONV2D_DIM; ++i) {
      stride[i] = size[i];
    }
    break;
  case PropertyType::padding:
    for (int i = 0; i < CONV2D_DIM; ++i) {
      padding[i] = size[i];
    }
    break;
  default:
    ml_loge("Error: Unknown Layer Property type");
    status = ML_ERROR_INVALID_PARAMETER;
    break;
  }
  return status;
}

int Conv2DLayer::setFilter(int f) {
  int status = ML_ERROR_NONE;
  if (f <= 0) {
    ml_loge("Error: number of filters must be greater than 0");
    status = ML_ERROR_INVALID_PARAMETER;
  }
  filter_size = f;
  return status;
}

void Conv2DLayer::setProperty(const PropertyType type,
                              const std::string &value) {
  int status = ML_ERROR_NONE;

  switch (type) {
  case PropertyType::filters: {
    if (!value.empty()) {
      status = setUint(filter_size, value);
      throw_status(status);
    }
  } break;
  case PropertyType::kernel_size:
    if (!value.empty()) {
      status = getValues(CONV2D_DIM, value, (int *)(kernel_size));
      throw_status(status);
      if (kernel_size[0] == 0 || kernel_size[1] == 0) {
        throw std::invalid_argument(
          "[Conv2DLayer] kernel_size must be greater than 0");
      }
    }
    break;
  case PropertyType::stride:
    if (!value.empty()) {
      status = getValues(CONV2D_DIM, value, (int *)(stride));
      throw_status(status);
      if (stride[0] == 0 || stride[1] == 0) {
        throw std::invalid_argument(
          "[Conv2DLayer] stride must be greater than 0");
      }
    }
    break;
  case PropertyType::padding:
    if (!value.empty()) {
      status = getValues(CONV2D_DIM, value, (int *)(padding));
      throw_status(status);
    }
    break;
  case PropertyType::normalization:
    if (!value.empty()) {
      status = setBoolean(normalization, value);
      throw_status(status);
    }
    break;
  case PropertyType::standardization:
    if (!value.empty()) {
      status = setBoolean(standardization, value);
      throw_status(status);
    }
    break;
  default:
    Layer::setProperty(type, value);
    break;
  }
}

int Conv2DLayer::conv2d(float *in, TensorDim indim, const float *kernel,
                        TensorDim kdim, float *out, unsigned int const *stride,
                        float bias) {

  int status = ML_ERROR_NONE;
  unsigned int channel = indim.channel();
  unsigned int height = indim.height();
  unsigned int width = indim.width();
  unsigned int k_width = kdim.width();
  unsigned int k_height = kdim.height();
  unsigned int o_width = ((indim.width() - kdim.width()) / stride[0] + 1);

  if (indim.channel() != kdim.channel()) {
    ml_loge("Error: Input and Kenel Dimension is not match!");
    return ML_ERROR_INVALID_PARAMETER;
  }

  // Optimizer This routine : There are lots of dulicated calculations
  unsigned int I = 0;
  unsigned int J = 0;
  for (unsigned int j = 0; j < height - k_height + 1; j += stride[0]) {
    J = 0;
    for (unsigned int k = 0; k < width - k_width + 1; k += stride[1]) {
      float sum = 0.0f;
      for (unsigned int i = 0; i < channel; ++i) {
        for (unsigned int ki = 0; ki < k_height; ++ki) {
          for (unsigned int kj = 0; kj < k_width; ++kj) {
            float kern_value = kernel[i * k_height * k_width + ki * k_width + kj];
            float in_value = in[i * height * width + (j + ki) * width + (k + kj)];
            float add_val = kern_value * in_value;
            sum += add_val;
          }
        }
      }
      sum += bias;
      out[I * o_width + J] = sum;
      J++;
    }
    I++;
  }
  return status;
}

  int Conv2DLayer::conv2d_gemm(const float *mKernel, TensorDim kdim, Tensor const &in, TensorDim outdim) {

  int status = ML_ERROR_NONE;

  std::vector<float> inCol(kdim.getFeatureLen() * outdim.width() *
                           outdim.height());

  std::vector<float> out(outdim.getFeatureLen());

  for (unsigned int b = 0; b < in.batch(); ++b) {
    Tensor in_padded = zero_pad(b, in, padding);
    status = im2col(in_padded, kdim, inCol.data(), outdim);
    if (status != ML_ERROR_NONE)
      throw std::runtime_error("Forwarding Convolution failed.");

    float alpha_dgemm = 1.0f;
    float beta_dgemm = 0.0f;
    const float *data = mKernel;
    const float *mdata = inCol.data();
    float *rdata = out.data();
    memset(rdata, 0, out.size() * sizeof(out[0]));

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, filter_size,
                outdim.width() * outdim.height(), kdim.getFeatureLen(),
                alpha_dgemm, data, kdim.getFeatureLen(), mdata,
                outdim.width() * outdim.height(), beta_dgemm, rdata,
                outdim.width() * outdim.height());

    memcpy(hidden.getAddress(b * hidden.getDim().getFeatureLen()), rdata,
           out.size() * sizeof(float));

  }

  return status;
}

int Conv2DLayer::im2col(Tensor in_padded, TensorDim kdim, float *inCol,
                        TensorDim outdim) {

  int status = ML_ERROR_NONE;
  unsigned int count;
  unsigned int channel = in_padded.channel();
  unsigned int height = in_padded.height();
  unsigned int width = in_padded.width();
  unsigned int k_width = kdim.width();
  unsigned int k_height = kdim.height();

  if (channel != kdim.channel()) {
    ml_loge("Error: Input and Kenel Dimension is not match!");
    return ML_ERROR_INVALID_PARAMETER;
  }

  // Optimizer This routine : There are lots of dulicated calculations
  unsigned int J = 0;
  for (unsigned int j = 0; j <= height - k_height; j += stride[0]) {
    for (unsigned int k = 0; k <= width - k_width; k += stride[1]) {
      count = 0;
      for (unsigned int i = 0; i < channel; ++i) {
        for (unsigned int ki = 0; ki < k_height; ++ki) {
          for (unsigned int kj = 0; kj < k_width; ++kj) {
            inCol[count * (outdim.width()*outdim.height())+J] =
              in_padded.getData()[i * height * width + (j + ki) * width + (k + kj)];
	    count++;
          }
        }
      }
      J++;
    }
  }

  if (J != outdim.width()*outdim.height())
    status = ML_ERROR_INVALID_PARAMETER;

  return status;
}
  

} /* namespace nntrainer */
