# Generative Model

**目录**

- [Generative Model](#generative-model)
  - [Generative Model based on Likelihood-based method](#generative-model-based-on-likelihood-based-method)
  - [Glow: Generative Flow with Invertible 1x1 Convolutions](#glow-generative-flow-with-invertible-1x1-convolutions)
    - [Flow-based Generative Model](#flow-based-generative-model)
    - [**Glow Architecture**](#glow-architecture)
    - [**Glow Three Components**](#glow-three-components)
    - [**Actnorm**](#actnorm)
    - [**可逆 1x1 卷积**](#可逆-1x1-卷积)
    - [**仿射耦合层**](#仿射耦合层)
  - [参考文献](#参考文献)

## 生成模型

生成模型的本质，就是希望用一个已知的概率模型来拟合所给的数据样本，也就是说，得写出一个带参数θ的分布qθ(x)。然而，神经网络只是“万能函数拟合器”，却不是“万能分布拟合器”，也就是它原则上能拟合任意函数，但不能随意拟合一个概率分布，因为概率分布有“非负”和“归一化”的要求。

这样一来，能直接写出来的只有离散型的分布，或者是连续型的高斯分布。






## Generative Model based on Likelihood-based method

- Autoregressive models:  simplicity, but limited parallelizability
- Variational auto-encoders (VAE) :  optimize a lower bound on the log-likelihood of the data
- Flow-based generative model :  NICE 中首先介绍,  RealNVP 中进行扩展。

## [Glow: Generative Flow with Invertible 1x1 Convolutions](https://d4mucfpksywv.cloudfront.net/research-covers/glow/paper/glow.pdf)

### Flow-based Generative Model

- 基于log-likelihood , 提取隐变量进行推理。VAE可以推理隐变量近似值的数据点；GAN根本没有encoder来推理隐变量。可逆生成模型中, 这可以精确推理无需估计。
- 流式生成模型，数据通过一系列可逆的函数变换得到与其对应的隐变量（一一对应关系），而且通过选择合适的可逆函数变换，可以准确计算出训练数据的对数似然。

### **Glow Architecture**

<img src="README/images/flow-architecture.png" height=400>

### **Glow Three Components**

x 表示输入； y 表示输出；NN() 表示非线性映射

<img src="README/images/flow-3-components.png" height=400>

### **Actnorm**

RealNVP中用到了BN层，而Glow中提出了名为**Actnorm层来取代BN**。

所谓Actnorm层事实上只不过是NICE中的尺度变换层的一般化，也就是 缩放平移变换。

$\vec{z}= \frac{z-μ}{σ}$   ,    其中μ, σ都是训练参数。

Glow在论文中提出的创新点是用初始的batch的均值和方差去初始化μ, σ这两个参数，但事实上所提供的源码并没有做到这一点，纯粹是零初始化。

缩放平移的加入，确实有助于更好地训练模型。

由于Actnorm的存在，仿射耦合层的尺度变换已经显得不那么重要了。

相比于加性耦合层，仿射耦合层多了一个尺度变换层，从而计算量翻了一倍。

但事实上相比加性耦合，仿射耦合效果的提升并不高（尤其是加入了Actnorm后），所以要训练大型的模型，为了节省资源，一般都只用加性耦合，比如Glow训练256x256的高清人脸生成模型，就只用到了加性耦合。

### **可逆 1x1 卷积**

Glow模型在RealNVP的基础上引入 1x1可逆卷积 代替 打乱通道轴的操作，对RealNVP的原始模型做了简化和规范，使得它更易于理解和使用。

不是任意情况下套用卷积都是合理的，用卷积的前提是输入（在空间维度）具有局部相关性。图像本身是具有局部相关性的，相邻像素有一定关联，一般图像模型都可以使用卷积。

但flow中的两个操作：

1. 将输入分割为两部分x1,x2，然后输入到耦合层中，而模型s,t只对x1进行处理；
2. 特征输入耦合层之前，要随机打乱原来特征的各个维度（相当于乱序的特征）。

这两个操作都会破坏局部相关性，比如分割操作有可能割裂原来相邻的像素，随机打乱也可能将原来相邻的两个像素分割得很远。

如果还要使用卷积，就要保留空间局部相关性。

图像有三个轴：高度（height）、宽度（width）、通道（channel），前两个属于空间轴，具有局部相关性，因此只处理“通道”轴。

RealNVP约定分割和打乱操作，都只对“通道”轴执行。
沿着通道将输入分割为x1,x2后，x1还是具有局部相关性的，还有沿着通道按着同一方式打乱整体后，空间部分的相关性依然得到保留，因此在模型s,t中就可以使用卷积了。

可逆1x1卷积源于 对置换操作的一般化。

在flow模型中，一步很重要的操作就是将各个维度重新排列，NICE 是简单反转，而 RealNVP 则是随机打乱。

不管是哪一种，都对应着向量的置换操作。

事实上，对向量的置换操作，可以用矩阵乘法来描述，比如原来向量是[1,2,3,4]，分别交换第一、二和第三、四两个数，得到[2,1,4,3]，

这个操作可以用矩阵乘法来描述：

$$
\begin{pmatrix} 2 \\1\\4\\3\\  \end{pmatrix} =   \begin{pmatrix} 0&&1&&0&&0 \\1&&0&&0&&0\\0&&0&&0&&1\\0&&0&&1&&0\\  \end{pmatrix} \begin{pmatrix} 1 \\2\\3\\4\\  \end{pmatrix}

$$

右端第一项是“由单位矩阵不断交换两行或两列最终得到的矩阵”，称为置换矩阵。

将置换矩阵换成一般的可训练的参数矩阵呢？ 1x1可逆卷积就是这么来的。

flow模型提出时就已经明确指出，flow模型中的变换要满足两个条件：一是可逆，二是雅可比行列式容易计算。

如果直接写出变换:  h = xW

那么它就只是一个普通的没有bias的全连接层，并不能满足这两个条件。

为此，要做一些准备工作。

首先，我们让h和x的维度一样，也就是说W是一个方阵，这是最基本的设置；

其次，由于这只是一个线性变换，因此它的雅可比矩阵就是[∂h/∂x]=W，所以它的行列式就是det W，因此我们需要把−log|detW|这一项加入到loss中；

最后，初始化时为了保证W的可逆性，一般使用“随机正交矩阵”初始化。

Glow的论文做了对比实验，表明相比于直接反转，shuffle能达到更低的loss，

而相比shuffle，可逆1x1卷积能达到更低的loss。

可逆1x1卷积虽然能降低loss，但是有一些要注意的问题。

第一，loss的降低不代表生成质量的提高，

比如A模型用了shuffle，训练200个epoch训练到loss=-50000，

B模型用了可逆卷积，训练150个epoch就训练到loss=-55000，

那么通常来说在当前情况下B模型的效果还不如A（假设两者都还没有达到最优）。

事实上可逆1x1卷积只能保证大家都训练到最优的情况下，B模型会更优。

第二，简单实验中发现，用可逆1x1卷积达到饱和所需要的epoch数，要远多于简单用shuffle的epoch数。

### **仿射耦合层**

[**NICE**](https://arxiv.org/pdf/1410.8516.pdf) 中，提出了加性耦合层，也提到了乘性耦合层，不过没有用上；

<img src="README/images/glow-nice-coupling.png" height=150>

NICE通过交叉耦合，充分混合信息

h1 = x1
h2 = s(x1) ⊗ x2 + t(x1)

在 **[RealNVP](https://arxiv.org/pdf/1605.08803.pdf)**，加性和乘性耦合层结合在一起，成为一个“仿射耦合层”。

s, t 都是x1的向量函数，形式上第二个式子对应于x2的一个仿射变换，因此称为“仿射耦合层”。
仿射耦合的雅可比矩阵依然是一个三角阵，但对角线不全为1，用分块矩阵表示为:
<img src="README/images/glow-matrix-segment.png" height=100>

<img src="README/images/glow-realnvp-coupling.png" height=150>

RealNVP通过随机打乱每一步的输出向量，使信息混合得更充分均匀

这里随机打乱，是将每一步flow输出的两个向量h1,h2拼接成一个向量h，然后将这个向量重新随机排序。

## 参考文献

- [RealNVP与Glow：流模型的传承与升华](https://kexue.fm/archives/5807)
