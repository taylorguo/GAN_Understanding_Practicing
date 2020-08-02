# GAN Applications

********

## Image-to-Image Translation

********
:strawberry:  [**Pix2Pix**](https://arxiv.org/pdf/1611.07004.pdf)   :date:   2016.11v1    :blush:  UC Berkeley

Image-to-Image Translation with Conditional Adversarial Networks

改进：图像到图像的风格转移： 无需手工处理映射函数，损失函数

思路：cGAN + L1 Loss, U-Net;  PatchGAN Discriminator on NxN image patch

   ------
      Conditional GANs suitable for image-to-image translation tasks, 

      condition on an input image and generate a corresponding output image

      cGAN比较适合图像到图像的风格转移: 给输入图像施加条件, 生成对应的输出图像

   ------
      Generator: use a “U-Net”-based architecture; add skip connections

      For image translation, there is a great deal of low-level information shared between the input and output

      and it would be desirable to shuttle this information directly across the net

      Discriminator: use a convolutional “PatchGAN” classifier, which only penalizes structure at the scale of image patches

      Both generator and discriminator use modules of the form convolution-BatchNorm-ReLu

      生成器使用 类U-Net 架构; 添加跳层连接, 而不仅仅是使用自动编解码器

      图像转移, 希望输入和输出共享低阶信息, 直接把这些信息透传到网络中, 例如边缘信息

      判别器使用 Markovian discriminator 马尔科夫判别器/PatchGAN分类器, 只约束图像 NxN区块结构

      N远小于图像尺寸时, 生成的图像质量较高

      生成器和判别器都是用convolution-BatchNorm-ReLu

   ------
   [It's beneficial to mix the GAN objective with a more traditional loss, such as L2 distance](https://arxiv.org/pdf/1604.07379.pdf)

      discriminator’s job remains unchanged, 

      generator is tasked to not only fool the discriminator but also to be near the ground truth output in an L2 sense

      Using L1 distance rather than L2 as L1 encourages less blurring

      生成器不仅仅是要与判别器博弈, 还需要更可能地接近 训练样本;

      相比L2, 使用L1距离可以使图片更清晰


#### Loss Function

   <img src="../../README/images/pix2pix_loss.png" height=200>

   λ = 0  , only cGAN gives much sharper results but introduces visual artifacts

   λ = 100, both terms together reduces these artifacts


#### Implementation 

- <img src="../../README/images/pytorch.png" height="13">  [Pix2Pix + BEGAN PyTorch](https://github.com/taey16/pix2pixBEGAN.pytorch)

- <img src="../../README/images/keras.png" height="13"> [Pix2Pix Keras](https://github.com/tdeboissiere/DeepLearningImplementations/tree/master/pix2pix)


#### Reference

- [How to Develop a Pix2Pix GAN for Image-to-Image Translation](https://machinelearningmastery.com/how-to-develop-a-pix2pix-gan-for-image-to-image-translation/)

- [New York Google Map dataset for Experiment](http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/maps.tar.gz)


********

:strawberry:  [**Pix2PixHD**](https://arxiv.org/pdf/1711.11585.pdf)   :date:   2017.11v1    :blush:  NVidia / UC Berkeley 

High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs


#### Reference 

-  <img src="../../README/images/pytorch.png" height="13">  [Pix2PixHD - NVidia Official PyTorch](https://github.com/NVIDIA/pix2pixHD)



********
:strawberry:  [**CycleGAN**](https://arxiv.org/pdf/1703.10593.pdf)   :date:   2017.03v1    :blush:  UC Berkeley 

Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks



#### Reference 

- [CycleGAN Offical](https://github.com/junyanz/CycleGAN)

- [How to Develop a CycleGAN for Image-to-Image Translation with Keras](https://machinelearningmastery.com/cyclegan-tutorial-with-keras/)

********

:strawberry:  [**UNIT**](https://arxiv.org/pdf/1703.00848.pdf)   :date:   2017.03v1    :blush:  Cornell University / NVidia

Unsupervised Image-to-Image Translation Networks




#### Network 

   <img src="../../README/images/unit_net.png"> 

#### Implementation 

- <img src="../../README/images/pytorch.png" height="13"> 

- <img src="../../README/images/keras.png" height="13">

- <img src="../../README/images/tf1.png" height="13">

- <img src="../../README/images/tf2.png" height="13">   


#### Reference 

********

:strawberry:  [**BicycleGAN**](https://arxiv.org/pdf/1711.11586.pdf)   :date:   2017.11v1    :blush:  UC Berkeley / Adobe Research

Toward Multimodal Image-to-Image Translation


#### Network 

   <img src="../../README/images/bicyclegan_net.png"> 

#### Implementation 

- <img src="../../README/images/pytorch.png" height="13">  [BicycleGAN Official PyTorch](https://github.com/junyanz/BicycleGAN)

- <img src="../../README/images/keras.png" height="13">

- <img src="../../README/images/tf1.png" height="13">

- <img src="../../README/images/tf2.png" height="13">   


#### Reference 




********
:strawberry:  [**MUNIT**](https://arxiv.org/pdf/1804.04732.pdf)   :date:   2018.04v1    :blush:  Cornell University / NVidia

MUNIT: Multimodal UNsupervised Image-to-image Translation


#### Network 

   <img src="../../README/images/munit_net.png"> 

#### Reference 

-  <img src="../../README/images/pytorch.png" height="13">  [MUNIT - NVidia Official PyTorch](https://github.com/NVlabs/MUNIT)

- <img src="../../README/images/keras.png" height="13"> [MUNIT - Keras](https://github.com/shaoanlu/MUNIT-keras)

- <img src="../../README/images/tf1.png" height="13"> [MUNIT - tensorflow1.4](https://github.com/taki0112/MUNIT-Tensorflow)

- <img src="../../README/images/tf2.png" height="13"> 

********

:strawberry:  [**U-GAN-IT**](https://arxiv.org/pdf/1907.10830.pdf)   :date:   2019.07v1

#### Loss Function 

   - Adaptive Layer-Instance Normalization (AdaLIN)

      Combine the advantages of AdaIN and LN by selectively keeping or changing the content information
      选择或改变特定内容信息,融合了AdaIN和LN的优势

#### Network 

   <img src="../../README/images/u-gan-it_net.png"> 

#### Implementation 

- <img src="../../README/images/pytorch.png" height="13">  [FID PyTorch](https://github.com/mseitzer/pytorch-fid)

- <img src="../../README/images/keras.png" height="13">

- <img src="../../README/images/tf1.png" height="13">

- <img src="../../README/images/tf2.png" height="13">   


********

:strawberry:  [**FUNIT**](https://arxiv.org/pdf/1905.01723.pdf)   :date:   2019.05v1

Few-Shot Unsupervised Image-to-Image Translation


#### Network 

   <img src="../../README/images/funit_block.jpg" height=450> 

********



## Face Synthesis





## Video Synthesis

:tomato: [**Few-shot Video-to-Video Synthesis**](https://arxiv.org/pdf/1910.12713.pdf)   :date:   2016.11v1    :blush:  NVidia




#### Network




#### Implementation 

- <img src="../../README/images/pytorch.png" height="13">  [Few-Shot-Vid2Vid Pytorch](https://github.com/NVlabs/few-shot-vid2vid)


********

:tomato: [**Face2Face**](https://zollhoefer.com/papers/CACM19_F2F/paper.pdf)   :date:   2016.03.23v1    :blush: Max-Planck-Institute for Informatics / TUM

Face2Face: Real-time Face Capture and Reenactment of RGB Videos

#### Implementation 

- <img src="../../README/images/pytorch.png" height="13">  

- <img src="../../README/images/keras.png" height="13">

- <img src="../../README/images/tf1.png" height="13"> [face2face-demo](https://github.com/datitran/face2face-demo)

- <img src="../../README/images/tf2.png" height="13">   

********

:tomato: [**Neural Talking Head**](https://arxiv.org/pdf/1905.08233.pdf)   :date:   2019.05.20v1    :blush:  Samsung AI Center, Moscow / Skolkovo Institute of Science and Technology

Few-Shot Adversarial Learning of Realistic Neural Talking Head Models

#### Network

   <img src="../../README/images/neural-head_net.png" height=260>

********

:tomato: [**MarioNETte**](https://arxiv.org/pdf/1911.08139.pdf)   :date:   2019.11.19v1    :blush:  Hyperconnect Korea

MarioNETte: Few-shot Face Reenactment Preserving Identity of Unseen Targets

#### Network

   <img src="../../README/images/marionette_net.png" height=230>


********
:tomato: [**First Order Motion Model**](https://arxiv.org/pdf/2003.00196.pdf)   :date:   2020.02.29v1    :blush:  University of Trento

First Order Motion Model for Image Animation

#### Network

- <img src="../../README/images/first-order-model_net.png">





