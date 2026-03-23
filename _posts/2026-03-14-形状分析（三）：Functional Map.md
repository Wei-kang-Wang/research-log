---
layout: post
comments: True
title: "形状分析（三）：Functional Map"
date: 2026-03-14 03:09:00
tags: shape_analysis
---

<!--more-->

{: class="table-of-content"}
* TOC
{:toc}

---

### 1. Functional map的提出与定义

Functional map在2012年由Ovsjanikov Maks等人于$$\left[1 \right]$$中提出，用来描述两个shapes之间的映射关系。其最大的创新点在于，不同于之前的工作将shapes之间的映射关系表示为点与点之间的对应关系（pointwise map），而是用两个shapes上定义的real-valued functions之间的对应关系（functional map）来表示两个shapes之间的映射关系。

$$\left[1 \right]$$中总结了其主要的贡献：
* 使用Laplace-Beltrami算子的eigenfunctions作为shape上定义的函数空间的基，利用其global geometry-aware的特性（小的eigenvalue对应的eigenfunctions一般表示更加global的特征，大的eigenvalues对应的eigenfunctions一般表示更细节的特征），可以让shapes之间的functional map的大小一般远小于pointwise map（取一定数量比如说100个eigenfunctions就可以很好的表示shapes之间的对应关系了）
* functional maps除了可以用于shape matching的任务，还可以用来做其他shape analysis，比如symmetry detection等。

下面我们就详细介绍一下functional map。

对于两个流形，$$\mathcal{M}, \mathcal{N}$$，用$$T: \mathcal{M} \rightarrow \mathcal{N}$$来表示它们之间的bijective pointwise map（可以是离散的，即$$\mathcal{M}, \mathcal{N}$$是shapes，也可以是连续的，即一般意义上的流形）。那么对于定义在$$\mathcal{M}$$上的scalar函数$$f: \mathcal{M} \rightarrow \mathbb{R}$$，$$T$$自然的引入了一个定义在$$\mathcal{N}$$上的函数$$g = f \circ T^{-1}: \mathcal{N} \rightarrow \mathbb{R}$$。对于任意的定义在$$\mathcal{M}$$上的函数$$f$$，都可以按照上面的方式得到一个定义在$$\mathcal{N}$$上的函数$$g$$，那么就建立了一个依赖于$$T$$的定义在$$\mathcal{M}$$上的函数构成的函数空间$$\mathcal{F}(\mathcal{M}, \mathbb{R})$$到定义在$$\mathcal{N}$$上的函数构成的函数空间$$\mathcal{F}(\mathcal{N}, \mathbb{R})$$的映射，记为$$T_{\mathcal{F}}: \mathcal{F}(\mathcal{M}, \mathbb{R}) \rightarrow \mathcal{F}(\mathcal{N}, \mathbb{R})$$。$$T_{\mathcal{F}}$$就被称作映射$$T$$的functional representation。

$$T_{\mathcal{F}}$$和$$T$$之间有如下关系

**性质一**：给定$$T_{\mathcal{F}}$$，我们可以还原原始的mapping $$T$$

证明：对于$$\mathcal{M}$$上任意一点$$a$$，构建一个定义在$$\mathcal{M}$$上的indicator function $$f$$，满足$$f(a)=1$$，$$f(v) = 0, \forall v \neq a$$。而$$g(y) = T_{\mathcal{F}}(f)(y) = f \circ T^{-1}(y)$$，对于$$\mathcal{N}$$上的任一点$$b$$，一方面$$g(b)$$由$$T_{\mathcal{F}}(b)$$给定，另一方面根据构造只有在$$T^{-1}(b)=a$$的情况下$$g(y)$$才等于$$1$$，别的情况下都是$$0$$，从而就找到了满足$$T^{-1}(b)=a$$的$$b$$。而且因为$$T$$是bijective的，从而上述点$$b$$是唯一的，即$$T(a) = b$$。

**性质二**：对于任意的bijective pointwise map $$T: \mathcal{M} \rightarrow \mathcal{N}$$，由它得到的functional representation $$T_{\mathcal{F}}$$是定义在$$\mathcal{M}$$上的函数构成的函数空间的线性算子。

证明：对于任意两个定义在$$\mathcal{M}$$上的函数$$f_1, f_2$$以及两个scalar $$\alpha_1, \alpha_2$$，$$T_{\mathcal{F}}(\alpha_1 f_1 + \alpha_2 f_2) = (\alpha_1 f_1 + \alpha_2 f_2) \circ T^{-1} = \alpha f_1 \circ T^{-1} + \alpha_2 f_2 \circ T^{-1} = \alpha T_{\mathcal{F}}(f_1) + \alpha_2 T_{\mathcal{F}}(f_2)$$。

> 由上面两个结论我们可以说：$$T_{\mathcal{F}}$$和$$T$$是等价的，而且$$T_{\mathcal{F}}$$在函数空间上是线性的，但$$T$$是定义在流形上的任意bijective mapping（非线性）。

如果$$\mathcal{M}$$上的函数空间有一组基$$\lbrace \phi_i^{\mathcal{M}} \rbrace_{i=1}^{\infty}$$，那么定义在$$\mathcal{M}$$上的任意函数$$f$$是这组基的一个线性组合$$f = \sum_{i}^{\infty} a_i \phi_i^{\mathcal{M}}$$，从而：

$$T_{\mathcal{F}}(f) = T_{\mathcal{F}}(\sum_{i}^{\infty} a_i \phi_i^{\mathcal{M}}) = \sum_{i=1}^{\infty} a_i T_{\mathcal{F}}(\phi_i^{\mathcal{M}})$$

如果$$\mathcal{N}$$上的函数空间也有一组基$$\lbrace \phi_i^{\mathcal{N}} \rbrace_{i=1}^{\infty}$$，且$$T_{\mathcal{F}}(\phi_i^{\mathcal{M}}) = \sum_{j=1}^{\infty} c_{ij} \phi_j^{\mathcal{N}}$$，那么：

$$T_{\mathcal{F}}(f) = \sum_{i=1}^{\infty} a_i \sum_{j=1}^{\infty} c_{ij} \phi_j^{\mathcal{N}} = \sum_{j=1}^{\infty} \sum_{i=1}^{\infty} a_i c_{ij} \phi_j^{\mathcal{N}}$$

也就是说，如果我们将$$f \in \mathcal{F}(\mathcal{M}, \mathbb{R})$$表示为系数向量$$\boldsymbol{a} = (a_1, a_2, \cdots)$$，将$$g = T_{\mathcal{F}}(f) \in \mathcal{F}(\mathcal{N}, \mathbb{R})$$表示为稀疏向量$$\boldsymbol{b} = (b_1, b_2, \cdots)$$，那么上述结论表明：$$b_j = \sum_{i=1}^{\infty} a_i c_{ij}$$，其中$$a_i, i=1,2,\cdots$$由$$f$$决定，$$c_{ij}$$由$$\phi_i^{\mathcal{M}}$$和$$\phi_j^{\mathcal{N}}$$决定，即由$$\mathcal{M}$$和$$\mathcal{N}$$决定，和$$f$$无关。将所有的$$c_{ij}$$表示为一个矩阵$$C$$（可能是无穷矩阵），即有如下结论：

**性质三**：由$$\mathcal{M}$$和$$\mathcal{N}$$之间的bijective pointwise map $$T$$得到的functional representation $$T_{\mathcal{F}}$$可以由矩阵$$C$$（可能是无穷大小的）完全表示，即对于以系数向量$$\boldsymbol{a}$$表示的定义在$$\mathcal{M}$$上的函数$$f$$，$$T_{\mathcal{F}}(f) = C\boldsymbol{a}$$。

> 性质三和性质一表明$$C$$和$$T$$完全等价

有了上述的结论，我们就可以来正式的定义两个流形之间的functional mapping了，其比上面定义的两个流形之间的bijective pointwise mappings的functional representation要更加general。严格来说：

**定义一**：对于两个流形$$\mathcal{M}, \mathcal{N}$$，它们上面定义的scalar functions构成的函数空间$$\mathcal{F}(\mathcal{M}, \mathbb{R}), \mathcal{F}(\mathcal{N}, \mathbb{R})$$的一组基分别为$$\lbrace \phi_i^{\mathcal{M}} \rbrace_{i=1}^{\infty}, \lbrace \phi_i^{\mathcal{N}} \rbrace_{i=1}^{\infty}$$。那么依赖这两组基的functional mapping $$T_{\mathcal{F}}: \mathcal{F}(\mathcal{M}, \mathbb{R}) \rightarrow \mathcal{F}(\mathcal{N}, \mathbb{R})$$是由下述结果定义的$$\mathcal{F}(\mathcal{M}, \mathbb{R})$$上的算子：

$$T_{\mathcal{F}}(\sum_{i=1}^{\infty} a_i \phi_i^{\mathcal{M}}) = \sum_{j=1}\sum_{i=1} a_i c_{ij} \phi_i^{\mathcal{N}}$$

其中$$\lbrace c_{ij} \rbrace _{i,j=1}^{\infty}$$是满足上述结果的矩阵（可能是无限的）。

下面是shape matching的一个例子，用来说明pointwise map和functional map，其中颜色转移用来表示pointwise maps，而矩阵用来表示functional maps（使用Laplace-Beltrami算子的eigenfunctions作为basis，且只使用了前20个eigenfunctions）。左一是source shape，左二是ground truth的target shape，左三是left-right翻转的target shape，最右是将tail-head翻转的target shape。可以看出来，对于isometry transformations，即左二左三，functional maps是稀疏的，且大约是对角的，对于最右的这个non-isometric transformation，functional map是稠密的。

![1]({{ '/assets/images/functional_map_1.png' | relative_url }}){: width=100px style="float:center"} 



> 参考文献：
> * $$\left[1 \right]$$ Ovsjanikov, Maks, et al. "Functional maps: a flexible representation of maps between shapes." ACM Transactions on Graphics (ToG) 31.4 (2012): 1-11.
> * Ovsjanikov, Maks, et al. "Computing and processing correspondences with functional maps." SIGGRAPH ASIA 2016 Courses. 2016. 1-60.
> * Nogneng, Dorian, et al. "Improved functional mappings via product preservation." Computer Graphics Forum. Vol. 37. No. 2. 2018.
> * Melzi, S., et al. "ZoomOut: spectral upsampling for efficient shape correspondence." ACM TRANSACTIONS ON GRAPHICS 38.6 (2019).
> * Ren, Jing, et al. "Structured regularization of functional map computations." Computer Graphics Forum. Vol. 38. No. 5. 2019.
> * Sharma, Abhishek, and Maks Ovsjanikov. "Weakly supervised deep functional maps for shape matching." Advances in Neural Information Processing Systems 33 (2020): 19264-19275.
> * Donati, Nicolas, Abhishek Sharma, and Maks Ovsjanikov. "Deep geometric functional maps: Robust feature learning for shape correspondence." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2020.
> * Attaiki, Souhaib, Gautam Pai, and Maks Ovsjanikov. "Dpfm: Deep partial functional maps." 2021 International Conference on 3D Vision (3DV). IEEE, 2021.
> * Donati, Nicolas, et al. "Complex functional maps: A conformal link between tangent bundles." Computer Graphics Forum. Vol. 41. No. 1. 2022.
> * Li, Lei, Nicolas Donati, and Maks Ovsjanikov. "Learning multi-resolution functional maps with spectral attention for robust shape matching." Advances in Neural Information Processing Systems 35 (2022): 29336-29349.
> * Attaiki, Souhaib, and Maks Ovsjanikov. "Understanding and improving features learned in deep functional maps." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023.
> * Sun, Mingze, et al. "Spatially and spectrally consistent deep functional maps." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2023.
> * Magnet, Robin, and Maks Ovsjanikov. "Scalable and efficient functional map computations on dense meshes." Computer Graphics Forum. Vol. 42. No. 2. 2023.
> * Viganò, Giulio, Maks Ovsjanikov, and Simone Melzi. "NAM: Neural Adjoint Maps for refining shape correspondences." ACM Transactions on Graphics (TOG) 44.4 (2025): 1-15.
