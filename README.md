# NNtrainer

[![Code Coverage](https://img.shields.io/endpoint?url=https://nntrainer.github.io/coverage_result/coverage.json)](https://nntrainer.github.io/coverage_result/)
![GitHub repo size](https://img.shields.io/github/repo-size/nnstreamer/nntrainer)
![GitHub issues](https://img.shields.io/github/issues/nnstreamer/nntrainer)
![GitHub pull requests](https://img.shields.io/github/issues-pr/nnstreamer/nntrainer)
<a href="https://scan.coverity.com/projects/nnstreamer-nntrainer">
  <img alt="Coverity Scan Build Status"
       src="https://scan.coverity.com/projects/22512/badge.svg"/>
</a>
[![DailyBuild](https://github.com/nnstreamer/nntrainer/actions/workflows/daily_build.yml/badge.svg)](https://github.com/nnstreamer/nntrainer/actions/workflows/daily_build.yml)
[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/9179/badge)](https://www.bestpractices.dev/projects/9179)

NNtrainer is a Software Framework for training Neural Network models on devices.

## Overview

NNtrainer is an Open Source Project. The aim of the NNtrainer is to develop a Software Framework to train neural network models on embedded devices which have relatively limited resources. Rather than training whole layers of a network from the scratch, NNtrainer finetunes the neural network model on device with user data for the personalization.

Even if NNtrainer runs on device, it provides full functionalities to train models and also utilizes limited device resources efficiently. NNTrainer is able to train various machine learning algorithms such as k-Nearest Neighbor (k-NN), Neural Networks, Logistic Regression, Reinforcement Learning algorithms, Recurrent network and more. We also provide examples for various tasks such as Few-shot learning, ResNet, VGG, Product Rating and more will be added. All of these were tested on Samsung Galaxy smart phone with Android and PC (Ubuntu).

[ A New Frontier of AI: On-Device AI Training and Personalization ](https://dl.acm.org/doi/abs/10.1145/3639477.3639716), ICSE-SEIP, 2024 <br />
[ NNTrainer: Light-Weight On-Device Training Framework ](https://arxiv.org/pdf/2206.04688.pdf), arXiv, 2022 <br />
[  Open Source On-Device AI SW Platform ](https://youtu.be/im3uNrPLYx4?si=gMbw7LKKSnpXi59U), Samsung Developer Conference 2023 (Korean) <br />
[ NNTrainer: Personalize neural networks on devices! ](https://www.youtube.com/watch?v=HKKowY78P1A), Samsung Developer Conference 2021 <br />
[ NNTrainer: "On-device learning" ](https://www.youtube.com/embed/Jy_auavraKg?start=4035&end=4080), Samsung AI Forum 2021

## Running MoE On Device

NNTrainer enables running large-scale Mixture of Experts (MoE) models directly on device. Below are demonstrations of running MoE LLM on mobile devices:

| GPT-OSS 20B | Qwen3 MoE 30B-A3B |
|:-----------:|:-----------------:|
| <video src="./docs/videos/GPT_OSS_20B_Demo.mp4" autoplay loop muted playsinline width="320"></video> | <video src="./docs/videos/Qwen_30B_Demo.mp4" autoplay loop muted playsinline width="320"></video> |
| Running GPT-OSS 20B model on device | Running Qwen3 MoE 30B-A3B model on device |

## Official Releases

|     | [Tizen](http://download.tizen.org/snapshots/tizen/unified/latest/repos/standard/packages/) | [Ubuntu](https://launchpad.net/~nnstreamer/+archive/ubuntu/ppa) | Android/NDK Build |
| :-- | :--: | :--: | :--: |
|     | 6.0M2 and later | 22.04/24.04 | 9/P |
| arm | [![armv7l badge](http://ci.nnstreamer.ai/nntrainer/ci/daily-build/badge/tizen.armv7l_result_badge.svg)](http://ci.nnstreamer.ai/nntrainer/ci/daily-build/build_result/) | Available  | Ready |
| arm64 |  [![aarch64 badge](http://ci.nnstreamer.ai/nntrainer/ci/daily-build/badge/tizen.aarch64_result_badge.svg)](http://ci.nnstreamer.ai/nntrainer/ci/daily-build/build_result/) | Available  | [![android badge](http://ci.nnstreamer.ai/nntrainer/ci/daily-build/badge/arm64_v8a_android_result_badge.svg)](http://ci.nnstreamer.ai/nntrainer/ci/daily-build/build_result/) |
| x64 | [![x64 badge](http://ci.nnstreamer.ai/nntrainer/ci/daily-build/badge/tizen.x86_64_result_badge.svg)](http://ci.nnstreamer.ai/nntrainer/ci/daily-build/build_result/)  | [![ubuntu badge](http://ci.nnstreamer.ai/nntrainer/ci/daily-build/badge/ubuntu_result_badge.svg)](http://ci.nnstreamer.ai/nntrainer/ci/daily-build/build_result/)  | Ready  |
| x86 | [![x86 badge](http://ci.nnstreamer.ai/nntrainer/ci/daily-build/badge/tizen.i586_result_badge.svg)](http://ci.nnstreamer.ai/nntrainer/ci/daily-build/build_result/)  | N/A  | N/A  |
| Publish | [Tizen Repo](http://download.tizen.org/snapshots/tizen/unified/latest/repos/standard/packages/) | [PPA](https://launchpad.net/~nnstreamer/+archive/ubuntu/ppa) |   |
| API | C (Official) | C/C++ | C/C++  |

- Ready: CI system ensures build-ability and unit-testing. Users may easily build and execute. However, we do not have automated release & deployment system for this instance.
- Available: binary packages are released and deployed automatically and periodically along with CI tests.
- [Daily Release](http://ci.nnstreamer.ai/nntrainer/ci/daily-build/build_result/)
- SDK Support: Tizen Studio (6.0 and later)

## Getting Started

### [Installation](https://github.com/nnstreamer/nntrainer/blob/main/docs/getting-started.md)

Instructions for installing NNTrainer.

### [Tutorial](https://github.com/nnstreamer/nntrainer/blob/main/docs/how-to-create-model.md)

Introductions for creating your own model.

### [Running Examples](https://github.com/nnstreamer/nntrainer/blob/main/docs/how-to-run-examples.md)

Instructions for preparing NNTrainer for execution

### [Examples for NNTrainer](https://github.com/nnstreamer/nntrainer/tree/main/Applications)

NNTrainer examples for a variety of networks

## Components

### Supported Layers

This component defines layers which consist of a neural network model. Layers have their own properties to be set.

#### Neural Network Layers

 | Keyword | Layer Class Name | Description |
 |:-------:|:---:|:---|
 | conv1d | Conv1DLayer | Convolution 1-Dimensional Layer |
 | conv2d | Conv2DLayer | Convolution 2-Dimensional Layer |
 | conv2dtranspose | Conv2DTransposeLayer | Transposed Convolution 2-Dimensional Layer |
 | depthwiseconv2d | DepthwiseConv2DLayer | Depthwise Convolution 2-Dimensional Layer |
 | pooling2d | Pooling2DLayer | Pooling 2-Dimensional Layer. Support average / max / global average / global max pooling |
 | flatten | FlattenLayer | Flatten layer |
 | fully_connected | FullyConnectedLayer | Fully connected layer |
 | input | InputLayer | Input Layer. This is not always required |
 | batch_normalization | BatchNormalizationLayer | Batch normalization layer |
 | layer_normalization | LayerNormalizationLayer | Layer normalization layer |
 | activation | ActivationLayer | Set by layer property |
 | dropout | DropOutLayer | Dropout Layer |
 | embedding | EmbeddingLayer | Embedding Layer |
 | positional_encoding | PositionalEncodingLayer | Positional Encoding Layer |

#### Recurrent Layers

 | Keyword | Layer Class Name | Description |
 |:-------:|:---:|:---|
 | rnn | RNNLayer | Recurrent Layer |
 | rnncell | RNNCellLayer | Recurrent Cell Layer |
 | gru | GRULayer | Gated Recurrent Unit Layer |
 | grucell | GRUCellLayer | Gated Recurrent Unit Cell Layer |
 | lstm | LSTMLayer | Long Short-Term Memory Layer |
 | lstmcell | LSTMCellLayer | Long Short-Term Memory Cell Layer |
 | zoneout_lstmcell | ZoneoutLSTMCellLayer | Zoneout Long Short-Term Memory Cell Layer |
 | time_dist | TimeDistLayer | Time distributed Layer |

#### Attention Layers

 | Keyword | Layer Class Name | Description |
 |:-------:|:---:|:---|
 | attention | AttentionLayer | Attention layer |
 | multi_head_attention | MultiHeadAttentionLayer | Multi Head Attention Layer |
 | mol_attention | MolAttentionLayer | Mixture of Logistic Attention Layer |

#### Tensor Operation Layers

 | Keyword | Layer Class Name | Description |
 |:-------:|:---:|:---|
 | add | AddLayer | Element-wise addition layer |
 | addition | AdditionLayer | Add input layers |
 | subtract | SubtractLayer | Element-wise subtraction layer |
 | multiply | MultiplyLayer | Element-wise multiplication layer |
 | divide | DivideLayer | Element-wise division layer |
 | negative | NegativeLayer | Unary negation layer |
 | matmul | MatMulLayer | Matrix multiplication layer |
 | gather | GatherLayer | Gather (indexing) layer |
 | cast | CastLayer | Type casting layer |

#### Mathematical Function Layers

 | Keyword | Layer Class Name | Description |
 |:-------:|:---:|:---|
 | sin | SineLayer | Sine function layer |
 | cos | CosineLayer | Cosine function layer |
 | tan | TangentLayer | Tangent function layer |
 | sqrt | SQRTLayer | Square root function layer |
 | pow | PowLayer | Power/exponentiation function layer |

#### Reduction Layers

 | Keyword | Layer Class Name | Description |
 |:-------:|:---:|:---|
 | reduce_mean | ReduceMeanLayer | Mean reduction across dimensions |
 | reduce_sum | ReduceSumLayer | Sum reduction across dimensions |

#### Shape Manipulation Layers

 | Keyword | Layer Class Name | Description |
 |:-------:|:---:|:---|
 | reshape | ReshapeLayer | Reshape tensor dimension layer |
 | permute | PermuteLayer | Permute layer for transpose |
 | flatten | FlattenLayer | Flatten layer |
 | slice | SliceLayer | Slice layer to extract tensor portions |
 | concat | ConcatLayer | Concatenate input layers |
 | split | SplitLayer | Split layer |
 | upsample2d | Upsample2dLayer | 2D upsampling layer |

#### Utility Layers

 | Keyword | Layer Class Name | Description |
 |:-------:|:---:|:---|
 | identity | IdentityLayer | Identity layer (pass-through) |
 | weight | WeightLayer | Trainable weight tensor layer |
 | tensor | TensorLayer | QNN Tensor layer |
 | multiout | MultiOutLayer | Multi-Output Layer |
 | centroid_knn | CentroidKNN | Centroid K-nearest neighbor layer |
 | channel_shuffle | ChannelShuffle | Channel shuffle layer |

#### Preprocessing Layers

 | Keyword | Layer Class Name | Description |
 |:-------:|:---:|:---|
 | preprocess_flip | PreprocessFlipLayer | Preprocess random flip layer |
 | preprocess_l2norm | PreprocessL2NormLayer | Preprocess simple l2norm layer to normalize |
 | preprocess_translate | PreprocessTranslateLayer | Preprocess translate layer |

#### Backbone Layers

 | Keyword | Layer Class Name | Description |
 |:-------:|:---:|:---|
 | backbone_nnstreamer | NNStreamerLayer | Encapsulate NNStreamer layer |
 | backbone_tflite | TfLiteLayer | Encapsulate tflite as a layer |

#### OpenCL Accelerated Layers

 | Keyword | Layer Class Name | Description |
 |:-------:|:---:|:---|
 | fully_connected | FullyConnectedLayerCl | Fully connected layer (OpenCL) |
 | reshape | ReshapeLayerCl | Reshape layer (OpenCL) |
 | concat | ConcatLayerCl | Concatenation layer (OpenCL) |
 | addition | AdditionLayerCl | Addition layer (OpenCL) |
 | transpose | TransposeLayerCl | Transpose layer (OpenCL) |
 | swiglu | SwiGLULayerCl | SwiGLU activation layer (OpenCL) |
 | rmsnorm | RMSNormLayerCl | RMS Normalization layer (OpenCL) |


### Supported Optimizers

NNTrainer Provides

 | Keyword | Optimizer Name | Description |
 |:-------:|:---:|:---:|
 | sgd | Stochastic Gradient Decent | - |
 | adam | Adaptive Moment Estimation | - |
 | adamw | Adam with decoupled weight decay regularization | - |

 | Keyword | Learning Rate | Description |
 |:-------:|:---:|:---:|
 | exponential | exponential learning rate decay | - |
 | constant | constant learning rate | - |
 | step | step learning rate | - |

### Supported Loss Functions

NNTrainer provides

 | Keyword | Class Name | Description |
 |:-------:|:---:|:---:|
 | cross | CrossEntropyLossLayer | Cross entropy loss layer |
 | cross_sigmoid | CrossEntropySigmoidLossLayer | Cross entropy sigmoid loss layer |
 | cross_softmax | CrossEntropySoftmaxLossLayer | Cross entropy softmax loss layer |
 | constant_derivative | ConstantDerivativeLossLayer | Constant derivative loss layer |
 | mse | MSELossLayer | Mean square error loss layer |
 | kld | KLDLossLayer | Kullback-Leibler Divergence loss layer |

### Supported Activation Functions

NNTrainer provides

 | Keyword | Loss Name | Description |
 |:-------:|:---:|:---|
 | tanh | tanh function | set as layer property |
 | sigmoid | sigmoid function | set as layer property |
 | softmax | softmax function | set as layer property |
 | relu | relu function | set as layer property |
 | leaky_relu | leaky_relu function | set as layer property |
 | swish | swish function | set as layer property |
 | gelu | gelu function | set as layer property |
 | quick_gelu | quick gelu function | set as layer property |
 | elu | elu function | set as layer property |
 | selu | selu function | set as layer property |
 | softplus | softplus function | set as layer property |
 | mish | mish function | set as layer property |

### Tensor

Tensor is responsible for calculation of a layer. It executes several operations such as addition, division, multiplication, dot production, data averaging and so on. In order to accelerate calculation speed, CBLAS (C-Basic Linear Algebra: CPU) and CUBLAS (CUDA: Basic Linear Algebra) for PC (Especially NVIDIA GPU) are implemented for some of the operations. Later, these calculations will be optimized.
Currently, we support lazy calculation mode to reduce complexity for copying tensors during calculations.

#### Supported Data Types

 | Type | Description |
 |:----:|:---|
 | FP32 | 32-bit floating point (FloatTensor) |
 | FP16 | 16-bit floating point (HalfTensor) |
 | Quantized | QInt4, QInt8, QInt16, Q4K, Q6K, UINT4, UINT8 |

#### Supported Formats

 | Format | Description |
 |:------:|:---|
 | NCHW | Batch, Channel, Height, Width (default) |
 | NHWC | Batch, Height, Width, Channel |

#### Tensor Operations

 | Category | Operations |
 |:--------:|:---|
 | Shape | 4D Tensor (B, C, H, W), reshape, transpose, split, concat/cat |
 | Arithmetic (element-wise) | add, subtract, multiply, divide (with in-place and strided variants) |
 | Linear Algebra | dot, dotBatched (with transpose flags and beta scaling), derivative support |
 | Reduction | sum (single/multi axis, by batch), average, max/min, argmax, argmin, topK |
 | Mathematical Functions | pow, sqrt, inv_sqrt, abs, erf, sin, cos, tan |
 | Normalization | normalization, standardization, l2norm |
 | Masking | dropout_mask, zoneout_mask, filter_mask |
 | Initialization | setZero, setValue, setRandNormal, setRandUniform, setRandBernoulli, Xavier/LeCun/HE initializers |
 | Data Management | save, read, copy, clone, getBatchSlice, getSharedDataTensor |
 | Function Application | apply (element-wise function), chain (lazy evaluation) |

### Others

NNTrainer provides

 | Keyword | Loss Name | Description |
 |:-------:|:---:|:---|
 | weight_initializer | Weight Initialization | Xavier(Normal/Uniform), LeCun(Normal/Uniform),  HE(Normal/Uniform) |
 | weight_regularizer | weight decay ( L2Norm only ) | needs set weight_regularizer_param & type |

### APIs
Currently, we provide [C APIs](https://github.com/nnstreamer/nntrainer/blob/master/api/capi/include/nntrainer.h) for Tizen. [C++ APIs](https://github.com/nnstreamer/nntrainer/blob/master/api/ccapi/include) are also provided for other platform. Java & C# APIs will be provided soon.

## Maintainer
* [Jijoong Moon](https://github.com/jijoongmoon)
* [MyungJoo Ham](https://github.com/myungjoo)
* [Geunsik Lim](https://github.com/leemgs)

## Reviewers
* [Sangjung Woo](https://github.com/again4you)
* [Wook Song](https://github.com/wooksong)
* [Jaeyun Jung](https://github.com/jaeyun-jung)
* [Hyoungjoo Ahn](https://github.com/helloahn)
* [Parichay Kapoor](https://github.com/kparichay)
* [Dongju Chae](https://github.com/dongju-chae)
* [Gichan Jang](https://github.com/gichan-jang)
* [Yongjoo Ahn](https://github.com/anyj0527)
* [Jihoon Lee](https://github.com/zhoonit)
* [Hyeonseok Lee](https://github.com/lhs8928)
* [Mete Ozay](https://github.com/meteozay)
* [Hyunil Park](https://github.com/songgot)
* [Jiho Chu](https://github.com/jihochu)
* [Yelin Jeong](https://github.com/niley7464)
* [Donghak Park](https://github.com/DonghakPark)
* [Hyungjun Seo](https://github.com/SeoHyungjun)
* [Seungbaek Hong](https://github.com/baek2sm)
* [Sungsik Kong](https://github.com/skykongkong8)
* [Donghyeon Jeong](https://github.com/djeong20)
* [Eunju Yang](https://github.com/EunjuYang)
* [Daekyoung Jung](https://github.com/dkjung)
* [Haehun Yang](https://github.com/haehun)

## Open Source License

The NNtrainer is an open source project released under the terms of the Apache License version 2.0.

## Contributing

Contributions are welcome! Please see our [Contributing](https://github.com/nnstreamer/nntrainer/blob/main/docs/contributing.md) Guide for more details.

[![](https://sourcerer.io/fame/dongju-chae/nnstreamer/nntrainer/images/0)](https://sourcerer.io/fame/dongju-chae/nnstreamer/nntrainer/links/0)[![](https://sourcerer.io/fame/dongju-chae/nnstreamer/nntrainer/images/1)](https://sourcerer.io/fame/dongju-chae/nnstreamer/nntrainer/links/1)[![](https://sourcerer.io/fame/dongju-chae/nnstreamer/nntrainer/images/2)](https://sourcerer.io/fame/dongju-chae/nnstreamer/nntrainer/links/2)[![](https://sourcerer.io/fame/dongju-chae/nnstreamer/nntrainer/images/3)](https://sourcerer.io/fame/dongju-chae/nnstreamer/nntrainer/links/3)[![](https://sourcerer.io/fame/dongju-chae/nnstreamer/nntrainer/images/4)](https://sourcerer.io/fame/dongju-chae/nnstreamer/nntrainer/links/4)[![](https://sourcerer.io/fame/dongju-chae/nnstreamer/nntrainer/images/5)](https://sourcerer.io/fame/dongju-chae/nnstreamer/nntrainer/links/5)[![](https://sourcerer.io/fame/dongju-chae/nnstreamer/nntrainer/images/6)](https://sourcerer.io/fame/dongju-chae/nnstreamer/nntrainer/links/6)[![](https://sourcerer.io/fame/dongju-chae/nnstreamer/nntrainer/images/7)](https://sourcerer.io/fame/dongju-chae/nnstreamer/nntrainer/links/7)

## Citation

If you find this NNTrainer project useful or relevant to your research, please consider citing our paper:


```
@inproceedings{10.1145/3639477.3639716,
author = {Moon, Jijoong and Lee, Hyeonseok and Chu, Jiho and Park, Donghak and Hong, Seungbaek and Seo, Hyungjun and Jeong, Donghyeon and Kong, Sungsik and Ham, Myungjoo},
title = {A New Frontier of AI: On-Device AI Training and Personalization},
year = {2024},
isbn = {9798400705014},
publisher = {Association for Computing Machinery},
url = {https://doi.org/10.1145/3639477.3639716},
doi = {10.1145/3639477.3639716},
booktitle = {Proceedings of the 46th International Conference on Software Engineering: Software Engineering in Practice},
pages = {323–333},
numpages = {11},
keywords = {on-device AI, neural network, personalization, training, software framework},
series = {ICSE-SEIP '24}
}
```
