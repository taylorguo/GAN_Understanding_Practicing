# GAN Applications


********

## Video Synthesis

********

:tomato: [**A Survey on Visual Transformer**](https://arxiv.org/pdf/2012.12556v1.pdf)   :date:   2020.12.23v1    :blush:  Huawei/

A Survey on Visual Transformer


#### Implementation 

- <img src="../../README/images/pytorch.png" height="13">  [flownet2-pytorch](https://github.com/NVIDIA/flownet2-pytorch)


********

:tomato: [**RAFT**](https://arxiv.org/pdf/2003.12039.pdf)   :date:   2020.03.26v1    :blush:  Princeton University

RAFT: Recurrent All-Pairs Field Transforms for Optical Flow

#### Implementation 

- <img src="../../README/images/pytorch.png" height="13">  [RAFT-pytorch](https://github.com/princeton-vl/RAFT)


********

:tomato: [**Video-to-Video Synthesis**](https://arxiv.org/pdf/1808.06601.pdf)   :date:   2018.08.20v1    :blush:  NVidia

Video-to-Video Synthesis



#### Implementation 

- <img src="../../README/images/pytorch.png" height="13">  [Vid2Vid Pytorch](https://github.com/NVIDIA/vid2vid)



********


:tomato: [**Few-shot Video-to-Video Synthesis**](https://arxiv.org/pdf/1910.12713.pdf)   :date:   2019.10.28v1    :blush:  NVidia

Few-shot Video-to-Video Synthesis: compose a video based on a small number of reference images and a semantic images based on vid2vid

Video-to-video synthesis (vid2vid): converting an input semantic video to an output photorealistic video.

Conditional GAN framework, user input data not sampling from noise distribution

Vid2vid is based on Image-to-image synthesis, and keeps frames temporally consistent as a whole

Adaptive Network: part of weights are dynamically computed based on input data 

#### Network

flow prediction network W : reuse vid2vid
soft occlusion map prediction network M : reuse vid2vid
intermediate image synthesis network H : conditional image generator, adopt SPADE generator for semantic image synthesis



#### Implementation 

- <img src="../../README/images/pytorch.png" height="13">  [Few-Shot-Vid2Vid Pytorch](https://github.com/NVlabs/few-shot-vid2vid)



********
:tomato: [**NaviGAN**](https://arxiv.org/pdf/2011.13786.pdf)   :date:   2020.11.27v1    :blush:  Yandex

Navigating the GAN Parameter Space for Semantic Image Editing

#### Network

- <img src="../../README/images/first-order-model_net.png">

#### Implementation

- <img src="../../README/images/pytorch.png" height="13"> [NaviGAN](https://github.com/yandex-research/navigan)
