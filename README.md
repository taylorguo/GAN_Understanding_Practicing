# GAN_Understanding_Practicing

## [GAN Understand](./GAN_Understanding/README.md)

********

## GAN Practice

:shipit: :sparkles: :+1: :clap:

********

:lemon:  [**GAN**](https://arxiv.org/pdf/1406.2661.pdf)   :date:   2015

#### Network

Discriminator
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d_1 (Conv2D)            (None, 14, 14, 128)       1280
_________________________________________________________________
leaky_re_lu_1 (LeakyReLU)    (None, 14, 14, 128)       0
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 7, 7, 128)         147584
_________________________________________________________________
leaky_re_lu_2 (LeakyReLU)    (None, 7, 7, 128)         0
_________________________________________________________________
flatten_1 (Flatten)          (None, 6272)              0
_________________________________________________________________
dropout_1 (Dropout)          (None, 6272)              0
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 6273
=================================================================
Total params: 155,137
Trainable params: 155,137
Non-trainable params: 0
_________________________________________________________________
```

Generator
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense_2 (Dense)              (None, 6272)              633472
_________________________________________________________________
leaky_re_lu_3 (LeakyReLU)    (None, 6272)              0
_________________________________________________________________
reshape_1 (Reshape)          (None, 7, 7, 128)         0
_________________________________________________________________
conv2d_transpose_1 (Conv2DTr (None, 14, 14, 128)       262272
_________________________________________________________________
leaky_re_lu_4 (LeakyReLU)    (None, 14, 14, 128)       0
_________________________________________________________________
conv2d_transpose_2 (Conv2DTr (None, 28, 28, 128)       262272
_________________________________________________________________
leaky_re_lu_5 (LeakyReLU)    (None, 28, 28, 128)       0
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 28, 28, 1)         6273
=================================================================
Total params: 1,164,289
Trainable params: 1,164,289
Non-trainable params: 0
_________________________________________________________________
```

GAN
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
sequential_2 (Sequential)    (None, 28, 28, 1)         1164289
_________________________________________________________________
sequential_1 (Sequential)    (None, 1)                 155137
=================================================================
Total params: 1,319,426
Trainable params: 1,164,289
Non-trainable params: 155,137
_________________________________________________________________
```

********

:lemon:  [**DCGAN**](https://arxiv.org/pdf/1511.06434.pdf)   :date:   2015

#### Loss Function 

- 【Binary_Cross_Entropy Loss】    
  
   <img src="./README/images/binary_crossentropy.png" height="50">
    
- 【Loss Function】    
  <!-- > ![CGAN loss](README/images/dcgan.png)  -->

   <img src="./README/images/dcgan.png" height="25">


#### Network 

   <img src="./README/images/dcganfull.png">

#### Implementation 
<!-- - ![tf1](README/images/tf1.png)  [DCGAN tensorflow Official](https://github.com/tensorflow/docs-l10n/blob/master/site/zh-cn/tutorials/generative/dcgan.ipynb) -->

- <img src="./README/images/pytorch.png" height="13">

- <img src="./README/images/keras.png" height="13">

- <img src="./README/images/tf1.png" height="13">

- <img src="./README/images/tf2.png" height="13">   


#### Reference 

- [DCGAN TensorFlow2.x Official](https://github.com/tensorflow/docs-l10n/blob/master/site/zh-cn/tutorials/generative/dcgan.ipynb)
- [DCGAN PyTorch](https://github.com/znxlwm/pytorch-MNIST-CelebA-GAN-DCGAN)

********

:lemon:  [**LAPGAN**](https://arxiv.org/pdf/1506.05751.pdf)   :date:   2015

#### Loss Function 

- 【Binary_Cross_Entropy Loss】    
  
   <img src="./README/images/binary_crossentropy.png" height="50">
    
- 【Loss Function】    
  <!-- > ![CGAN loss](README/images/dcgan.png)  -->

   <img src="./README/images/dcgan.png" height="25">


#### Network 

   <img src="./README/images/dcganfull.png">

#### Implementation 
<!-- - ![tf1](README/images/tf1.png)  [DCGAN tensorflow Official](https://github.com/tensorflow/docs-l10n/blob/master/site/zh-cn/tutorials/generative/dcgan.ipynb) -->

- <img src="./README/images/pytorch.png" height="13">

- <img src="./README/images/keras.png" height="13">

- <img src="./README/images/tf1.png" height="13">

- <img src="./README/images/tf2.png" height="13">   


#### Reference 

- [LAPGAN TensorFlow](https://github.com/jimfleming/LAPGAN)
- [LAPGAN PyTorch](https://github.com/AaronYALai/Generative_Adversarial_Networks_PyTorch/tree/master/LAPGAN)

********

:lemon:  [**ConditionalGAN**](https://arxiv.org/pdf/1411.1784.pdf)   :date:    2014

- Log-Likelihood Estimates 对数最大似然估计
- Conditioning the model to direct the data generation process possibly 模型增加条件控制控制数据生成过程; conditioning based on class labels 条件控制是数据类别标签
- Probabilistic one-to-many mapping is instantiated as a conditional predictive distribution, the input is taken to be conditioning variable 输入条件变量可以实现一对多的条件生成分布,从而生成样本

#### Loss Function
- 【Loss Function】    
  
   <img src="./README/images/cganloss.png" height="16">

#### Network 

#### Implementation 

- <img src="./README/images/pytorch.png" height="13">

- <img src="./README/images/keras.png" height="13">

- <img src="./README/images/tf1.png" height="13">

- <img src="./README/images/tf2.png" height="13">   

#### Reference 

- [cGAN Keras: How to Develop a Conditional GAN (cGAN) From Scratch](https://machinelearningmastery.com/how-to-develop-a-conditional-generative-adversarial-network-from-scratch/)
- [cDCGAN Keras](https://github.com/gaborvecsei/CDCGAN-Keras)

********

:lemon:  [**WassersteinGAN**](https://arxiv.org/pdf/1701.07875.pdf)   :date:    2017

#### Implementation 

- <img src="./README/images/pytorch.png" height="15">

- <img src="./README/images/keras.png" height="15">

- <img src="./README/images/tf1.png" height="15">

- <img src="./README/images/tf2.png" height="15">   

********

:lemon:  [**InfoGAN**](https://arxiv.org/pdf/1606.03657.pdf)   :date:    2016

#### Implementation 

- <img src="./README/images/pytorch.png" height="15">

- <img src="./README/images/keras.png" height="15">

- <img src="./README/images/tf1.png" height="15">

- <img src="./README/images/tf2.png" height="15">   


********
:lemon:  [**SN-GAN**](https://arxiv.org/pdf/1802.05751v3.pdf)   :date:    2018


   _________________________________________________________________
   Algorithm: SGD with Spectral Normalization
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   - Initialize  <img src="https://latex.codecogs.com/gif.latex?\tilde{u}_{l}\in&space;R^{d_{l}}" title="\tilde{u}_{l}\in R^{d_{l}}">,   for l = 1, ..., L with a random vector (sampled from isotropic distribution).
   - For each update & each layer l: 
      
      ❶ Apply power iteration method to a unnormalized weight <img src="https://latex.codecogs.com/gif.latex?W^{l}">:

      <img src="https://latex.codecogs.com/gif.latex?\tilde{\upsilon}&space;_{l}&space;\leftarrow&space;(W^{l})^{T}&space;\tilde{u}_{l}&space;/&space;\parallel&space;(W^{l})^{T}&space;\tilde{u}_{l}&space;\parallel&space;_{2}">

      <img src="https://latex.codecogs.com/gif.latex?\tilde{u}&space;_{l}&space;\leftarrow&space;W^{l}&space;\tilde{\upsilon}_{l}&space;/&space;\parallel&space;W^{l}&space;\tilde{\upsilon}_{l}&space;\parallel&space;_{2}">

      ❷ Calculate <img src="https://latex.codecogs.com/gif.latex?\bar{W}_{SN}(W^{l})" title="\bar{W}_{SN}(W^{l})"> with the spectral normalization:

      <img src="https://latex.codecogs.com/gif.latex?\bar{W}_{SN}(W^{l})&space;=&space;W^{l}&space;/&space;\sigma&space;(W^{l});&space;where,&space;\sigma&space;(W^{l})&space;=&space;{\tilde{u}_{l}}^{T}W^{l}\tilde{\upsilon&space;}_{l}" title="\bar{W}_{SN}(W^{l})"> 

      ❸ Update <img src="https://latex.codecogs.com/gif.latex?W^{l}"> with SGD on mini-batch dataset <img src="https://latex.codecogs.com/gif.latex?D_{M}"> with a learning rate α:

      <img src="https://latex.codecogs.com/gif.latex?W^{l}&space;\leftarrow&space;W^{l}&space;-&space;\alpha&space;{\bigtriangledown}_{W^{l}}&space;l&space;(\bar{W}_{SN}^{l}(W^{l}),D_{M})" title="W^{l} \leftarrow W^{l} - \alpha {\bigtriangledown}_{W^{l}} l (\bar{W}_{SN}^{l}(W^{l}),D_{M})">
      
         


   


#### Reference 

- [2017 - Spectral Norm Regularization for Improving the Generalizability of Deep Learning](https://arxiv.org/pdf/1705.10941.pdf)

- [2018 - 【Blog】 - Spectral Normalization Explained](https://christiancosgrove.com/blog/2018/01/04/spectral-normalization-explained.html)

********


:lemon:  [**Image Transformer**](https://arxiv.org/pdf/1802.05751v3.pdf)   :date:    2018



********

:lemon:  [**Self-Attention GAN**](https://arxiv.org/pdf/1805.08318.pdf)   :date:    2018



#### Network 

- Self-Attention Module: modeling long-range dependencies
- Spectral Normalization in Generator: stabilize GAN training
- TTUR: Speed-up training of regularized discriminator


#### Implementation 
<!-- - ![tf1](README/images/tf1.png)  [DCGAN tensorflow Official](https://github.com/tensorflow/docs-l10n/blob/master/site/zh-cn/tutorials/generative/dcgan.ipynb) -->

- <img src="./README/images/pytorch.png" height="15">

- <img src="./README/images/keras.png" height="15">

- <img src="./README/images/tf1.png" height="15">

- <img src="./README/images/tf2.png" height="15">   


#### Reference 

- [DCGAN TensorFlow2.x Official](https://github.com/tensorflow/docs-l10n/blob/master/site/zh-cn/tutorials/generative/dcgan.ipynb)
- [DCGAN PyTorch](https://github.com/znxlwm/pytorch-MNIST-CelebA-GAN-DCGAN)


********

:lemon:  [**Progressive Growing GAN**](https://arxiv.org/pdf/1710.10196.pdf)   :date:    2017



#### Network 



#### Implementation 
<!-- - ![tf1](README/images/tf1.png)  [DCGAN tensorflow Official](https://github.com/tensorflow/docs-l10n/blob/master/site/zh-cn/tutorials/generative/dcgan.ipynb) -->

- <img src="./README/images/pytorch.png" height="15">

- <img src="./README/images/keras.png" height="15">

- <img src="./README/images/tf1.png" height="15">

- <img src="./README/images/tf2.png" height="15">   


#### Reference 

- [DCGAN TensorFlow2.x Official](https://github.com/tensorflow/docs-l10n/blob/master/site/zh-cn/tutorials/generative/dcgan.ipynb)
- [DCGAN PyTorch](https://github.com/znxlwm/pytorch-MNIST-CelebA-GAN-DCGAN)


********

:lemon:  [**BigGAN**](https://arxiv.org/pdf/1809.11096.pdf)   :date:    2018



#### Network 



#### Implementation 
<!-- - ![tf1](README/images/tf1.png)  [DCGAN tensorflow Official](https://github.com/tensorflow/docs-l10n/blob/master/site/zh-cn/tutorials/generative/dcgan.ipynb) -->

- <img src="./README/images/pytorch.png" height="15">

- <img src="./README/images/keras.png" height="15">

- <img src="./README/images/tf1.png" height="15">

- <img src="./README/images/tf2.png" height="15">   


#### Reference 

- [DCGAN TensorFlow2.x Official](https://github.com/tensorflow/docs-l10n/blob/master/site/zh-cn/tutorials/generative/dcgan.ipynb)
- [DCGAN PyTorch](https://github.com/znxlwm/pytorch-MNIST-CelebA-GAN-DCGAN)

********

## Dataset

- [Kaggle - CelebA: 200k images with 40 binary attribute](https://www.kaggle.com/jessicali9530/celeba-dataset/data#)
- [CelebA in Google Drive](https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8)

********

## Reference

- [Faster Guaranteed GAN-based Recovery in Linear Inverse Problems](http://www.ima.umn.edu/materials/2019-2020/SW10.14-18.19/28282/IMA2019_Computation_Imaging_Talk_Bresler_Slides.pdf)

- [Generative model](https://en.wikipedia.org/wiki/Generative_model)

- [Lecture 13: Generative Models](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture13.pdf)

- [Ian Goodfellow GANs PPT @ NIPS 2016](http://www.iangoodfellow.com/slides/2016-12-04-NIPS.pdf)
