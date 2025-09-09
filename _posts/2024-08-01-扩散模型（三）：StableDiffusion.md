---
layout: post
comments: True
title: "扩散模型（三）：StableDiffusion"
date: 2024-08-01 01:09:00

---

<!--more-->

{: class="table-of-content"}
* TOC
{:toc}

---

StableDiffusion，出自于CVPR2022的这篇论文：[High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/pdf/2112.10752)。作为当下最出圈的扩散模型之一，StableDiffusion具有图像生成效果更好、训练速度更快、可以根据输入的文本生成图像等优点。而且，利用预训练好的StableDiffusion模型来进行各种下游任务更是如今一个热门方向，所以了解StableDiffusion的原理充分且必要。

StableDiffusion也是一种扩散模型，但其相比较于一般的扩散模型，主要有三个区别：（1）采样空间从像素空间变成了feature空间，所以网络需要增加一个从图像到feature的Encoder，以及从feature到图像的Decoder；（2）允许根据输入文本进行图像生成，且生成的图像语义信息和输入文本相匹配。其具体实现是在扩散模型原本的U-Net结构基础上，增加了multi-head attention mechanism。下面就根据这两个改动来介绍StableDiffusion的原理。

如下，是StableDiffusion采样（即生成新数据）的过程：

![11]({{ '/assets/images/diffusion_11.png' | relative_url }})
{: style="width: 1200px; max-width: 100%;"}

由流程图可以看到，输入的文本（例子里是An astronout riding a horse）经过一个freezed的text feature extractor获取embedding（是利用CLIP模型预训练好的，这样可以保证和图像特征的语义一致性）。而初始的特征是从标准高斯分布里采样的一个$$64 \times 64$$的feature，其和text embedding一起喂给一个text conditioned latent UNet，得到更新的feature，再和text embedding一同喂给该UNet，如此重复$$N$$次，最终的feature，经过一个variational autoencoder Decoder，得到输出图像，即为去噪后的生成图像。

## 1. 改进一：在feature层面上做diffusion，而不在像素层面上

StableDiffusion在原论文里的名字叫做latent diffusion model（LDM），而latent就表明这个扩散过程是在latent的feature上进行的，而并非在原图像空间上。这样做，可以大大加快StableDiffusion的速度，毕竟feature space的维度要比图像空间维度小很多，且一定程度上可以缓解低维流形假设带来的影响。

StableDiffusion的框架里，有一个encoder，将原图片压缩到低维的latent feature上，还有一个decoder，对于latent code的输入，reconstruct到图片空间上。而在encoder将图片映射到latent feature上之后，便在latent feature上做扩散过程：

![12]({{ '/assets/images/diffusion_12.png' | relative_url }})
{: style="width: 1200px; max-width: 100%;"}



## 2. 改进二：通过在U-Net结构里引入multi-heads attention mechanism来允许文本指导图片生成

为了允许扩散模型能根据输入文本生成语义匹配的图片，需要在反向扩散过程，即从噪声生成图片的过程中，将原先的只接受加噪图片（训练过程）或者噪声（采样过程）以及时间$$t$$的UNet，改为还能够再接受一个text embedding作为输入，而要作此改动，并且需要让图片的features能够学会对应的文本embedding里的语义信息，则需要将UNet改造为含有attention模块的新结构，attention模块就可以用来在图片的features和文本的embedding之间学习信息。

加了文本condition的StableDiffusion的反向扩散过程如下：

![13]({{ '/assets/images/diffusion_13.png' | relative_url }})
{: style="width: 1200px; max-width: 100%;"}

而具体来看改进后的UNet结构，则是如下图所示：

![14]({{ '/assets/images/diffusion_14.png' | relative_url }})
{: style="width: 1200px; max-width: 100%;"}

UNet新增加的多头注意力机制$$\textbf{Attention}(Q,K,V)$$的原理如下（以最右边的第一个模块为例）：

$$\textbf{Attention}(Q,K,V) = \text{softmax}(\frac{QK^T}{\sqrt{d}})V$$

其中$$Q = W_Q z_T$$，$$K=W_K \tau_{\theta}(y)$$，$$V = W_V \tau_{\theta}(y)$$，其中$$W_Q, W_K, W_V$$是三个矩阵，也就是该注意力模块里需要被学习的参数。

图里的switch的作用是：

* 如果输入的是文本，那么$$\tau_{\theta}$$就是某种text embedding extractor，比如预训练的CLIP或者BERT，获取了text embedding之后，和feature $$z_t$$计算cross-attention。
* 如果输入的是其它的可以和图片spatially aligned的输入，比如说semantic maps，images，inpaintings等， 那么$$\tau_{\theta}$$就变成了其它对应的feature extractor，而得到的feature也不再与图片feature进行cross attention计算了，而是直接concatenate起来输入给UNet来获取图片feature $$z_t$$的更新输出$$z_{t-1}$$。

## 3. StableDiffusion的训练和采样

### (1). 训练

StableDiffusion的训练数据是图片文本对，且每一对数据语义信息相同。

类似于DDPM的推导，我们可以直接写出LDM（也就是StableDiffusion）的训练loss如下：

$$
\begin{align}
z_0 &= \textbf{Encoder}(x_0) \\
z_t &= \sqrt{\bar{\alpha}_t} z_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, \  \text{where} \  \epsilon \sim \mathcal{N}(\textbf{0}, \textbf{I}) \\
\mathcal{L}_{LDM} = \mathop{\mathbb{E}}\limits_{t \sim \left[2, T \right], (x_0,y) \sim q(x_0,y), \epsilon \sim \mathcal{N}(\textbf{0}, \textbf{I})} \left[ \lVert \epsilon - f_{\phi}(z_t, t, \tau_{\theta}(y)) \rVert_2^2 \right]
\end{align}
$$

其中$$(x_0, y)$$是输入图片文本对，$$\tau_{\theta}$$是freezed的text embedding extractor，$$f_{\phi}$$是我们的UNet。

和DDPM的训练目标函数相比，只有两点区别：

* 引入了encoder来将输入图片映射到feature空间上
* UNet的输入增加了text embedding


### (2). 采样

StableDiffusion的采样过程如下：

![15]({{ '/assets/images/diffusion_15.png' | relative_url }})
{: style="width: 1200px; max-width: 100%;"}


## 4. 一些补充说明

### (1). DDPM和LDM的对比

普通的DDPM流程图：

![16]({{ '/assets/images/diffusion_16.png' | relative_url }})
{: style="width: 1200px; max-width: 100%;"}


LDM流程图：

![17]({{ '/assets/images/diffusion_17.png' | relative_url }})
{: style="width: 1200px; max-width: 100%;"}


### (2). 带有多头注意力机制的UNet的具体架构设计

参考下面的参考资料里4，5，6三个博客内容。


**参考资料**
* https://medium.com/@steinsfu/stable-diffusion-clearly-explained-ed008044e07e
* https://andrewkchan.dev/posts/diffusion.html
* https://zhuanlan.zhihu.com/p/582266032
* http://blog.cnbang.net/tech/3823/
* https://blog.csdn.net/xd_wjc/article/details/134441396
* https://zhuanlan.zhihu.com/p/582266032
