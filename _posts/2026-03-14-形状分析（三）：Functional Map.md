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


### 1. Functional map的提出

#### (1). Functional map的定义

Functional map在2012年由Ovsjanikov Maks等人于$$\left[1 \right]$$中提出，用来描述两个shapes之间的映射关系。其最大的创新点在于，不同于之前的工作将shapes之间的映射关系表示为点与点之间的对应关系（pointwise map），而是用两个shapes上定义的real-valued functions之间的对应关系（functional map）来表示两个shapes之间的映射关系。

$$\left[1 \right]$$中总结了其主要的贡献：
* 使用Laplace-Beltrami算子的eigenfunctions作为shape上定义的函数空间的基，利用其global geometry-aware的特性（小的eigenvalue对应的eigenfunctions一般表示更加global的特征，大的eigenvalues对应的eigenfunctions一般表示更细节的特征），可以让shapes之间的functional map的大小一般远小于pointwise map（取一定数量比如说100个eigenfunctions就可以很好的表示shapes之间的对应关系了）
* functional maps除了可以用于shape matching的任务，还可以用来做其他shape analysis，比如symmetry detection等。

下面我们就详细介绍一下functional map。

对于两个流形，$$\mathcal{M}, \mathcal{N}$$，用$$T: \mathcal{M} \rightarrow \mathcal{N}$$来表示它们之间的bijective pointwise map（可以是离散的，即$$\mathcal{M}, \mathcal{N}$$是shapes，也可以是连续的，即一般意义上的流形）。那么对于定义在$$\mathcal{M}$$上的scalar函数$$f: \mathcal{M} \rightarrow \mathbb{R}$$，$$T$$自然的引入了一个定义在$$\mathcal{N}$$上的函数$$g = f \circ T^{-1}: \mathcal{N} \rightarrow \mathbb{R}$$。对于任意的定义在$$\mathcal{M}$$上的函数$$f$$，都可以按照上面的方式得到一个定义在$$\mathcal{N}$$上的函数$$g$$，那么就建立了一个依赖于$$T$$的定义在$$\mathcal{M}$$上的函数构成的函数空间$$\mathcal{F}(\mathcal{M}, \mathbb{R})$$到定义在$$\mathcal{N}$$上的函数构成的函数空间$$\mathcal{F}(\mathcal{N}, \mathbb{R})$$的映射，记为$$T_{\mathcal{F}}: \mathcal{F}(\mathcal{M}, \mathbb{R}) \rightarrow \mathcal{F}(\mathcal{N}, \mathbb{R})$$。$$T_{\mathcal{F}}$$就被称作映射$$T$$的functional representation。

$$T_{\mathcal{F}}$$和$$T$$之间有如下关系

**性质一**：给定$$T_{\mathcal{F}}$$，我们可以还原原始的mapping $$T$$

证明：对于$$\mathcal{M}$$上任意一点$$a$$，构建一个定义在$$\mathcal{M}$$上的indicator function $$f$$，满足$$f(a)=1$$，$$f(v) = 0, \forall v \neq a$$。而$$g(y) = T_{\mathcal{F}}(f)(y) = f \circ T^{-1}(y)$$，对于$$\mathcal{N}$$上的任一点$$b$$，一方面$$g(b)$$由$$T_{\mathcal{F}}(f)(b)$$给定，另一方面根据构造只有在$$T^{-1}(b)=a$$的情况下$$g(y)$$才等于$$1$$，别的情况下都是$$0$$，从而就找到了满足$$T^{-1}(b)=a$$的$$b$$。而且因为$$T$$是bijective的，从而上述点$$b$$是唯一的，即$$T(a) = b$$。

> 注意，这个结论并不是说$$T_{\mathcal{F}}$$空间等价于$$T$$，其说的是一个由某个bijective pointwise mapping $$T$$推导出的functional representation $$T_{\mathcal{F}}$$可以用来恢复其对应的$$T$$，从而它们是等价的。但对于$$T_{\mathcal{F}}$$空间里任意一个元素，其不一定有对应的$$T$$，比如说将任意定义在$$\mathcal{M}$$上的函数$$f$$都映射为定义在$$\mathcal{N}$$上的全零函数，其就不存在对应的$$T$$。

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

> 注意，作者提到所有的这些map都不是diagonal的，并给出了$$\left[2,3,4 \right]$$三个参考文献，这是因为即使是isometric的shapes，它们的eigenvalues以及eigenfunctions也不会完全相同，而且即使是同一个shape（做self matching），shape可能会有重复的eigenvalues，其会产生一些块状正交矩阵（块的大小是该eigenvalue对应的eigenvectors张成的空间维度）。

![1]({{ '/assets/images/functional_map_1.png' | relative_url }}){: width=100px style="float:center"} 


#### (2). Functional maps的基的选择

Functional maps的提出并不依赖于基的选择，理论上可以是任意定义在流形上的函数空间的基。但实际操作过程中，这些基要满足：（1）compactness，即定义在流形上的绝大部分函数，可以用较少数量的基的线性组合就能够较高精度的表示；（2）stability，即即使shape有小的deformations，其基的线性组合组成的空间也不会有太大的变化。这两条性质可以让functional maps $$T_{\mathcal{F}}$$能够鲁棒的用较少的数量的基来表示，即

$$\sum_{j=1}^{\infty} \sum_{i=1}^{\infty} a_i c_{ij} \phi_j^{\mathcal{N}} \approx \sum_{j=1}^{n} \sum_{i=1}^{m} a_i c_{ij} \phi_j^{\mathcal{N}}$$

其中$$n,m$$是某个设定好的正整数（比如100）。

在实际操作中，Laplace-Beltrami算子的eigenfunctions是最常见的基。


#### (3). Functional maps的一些性质

**Functional map的稀疏性**

理论上，如果流形$$\mathcal{M}, \mathcal{N}$$是isometric的，且$$T$$作为bijective pointwise mapping也是一个isometry，那么正交基对应的functional mapping $$C_{ij}$$仅仅当$$\phi_j^{\mathcal{M}}$$和$$\phi_i^{\mathcal{N}}$$对应的eigenvalue值相等的时候，才不为零。也就是说，如果所有的eigenvalues都区分度比较大，那么$$C$$就是个对角矩阵。但实际计算中，$$T$$是near isometry，且$$C$$也是near对角矩阵，其更像是funnel-shaped，即左上角对应较小eigenvalues的区域基本上对角的，越往右下角越发散（对应几何细节）。


**关于Functional maps的线性约束**

对于一对给定的函数$$f: \mathcal{M} \rightarrow \mathbb{R}, g: \mathcal{N} \rightarrow \mathbb{R}$$，函数$$f$$和$$g$$之间的对应关系可以由$$C\boldsymbol{a} = \boldsymbol{b}$$来表示，其中$$\boldsymbol{a}, \boldsymbol{b}$$分别是函数$$f,g$$在$$\mathcal{M}, \mathcal{N}$$下某组基的系数向量，$$C$$是该映射的functional representation。在shape matching任务里，$$C$$以及pointwise mapping $$T$$都是未知的，而可能有多组$$(f_i, g_i)$$，即对应的$$(\boldsymbol{a_i}, \boldsymbol{b_i})$$是已知的，而它们的关系$$C\boldsymbol{a_i} = \boldsymbol{b_i}$$都是线性的，便于我们设计优化算法以及计算。

上述约束的常见几种是：

* Descriptor preservation，即$$f,g$$是per-vertex的features，如果$$f,g$$是scalar functions，比如每个点的高斯曲率，那么就有一个上述的约束。如果$$f,g$$是向量函数，比如每个点的高维特征，那么每个特征维度都有一个上述的约束（将每个维度都视作一个独立的函数）。
* Landmark point correspondences，如果我们给定两个shapes上对应的关键点的标注，即$$x \in \mathcal{M}, y \in \mathcal{N}$$，满足$$T(x) = y$$，其中$$T$$是未知的pointwise mapping。那么我们可以设计$$f,g$$是分别以$$x,y$$为中心的距离函数，或者分布函数。
* Segment correspondences，类似于landmark point correspondence，如果我们给定两个shapes上对应区域，也可以设计类似的函数$$f,g$$。


**Functional maps与线性算子的交换性**

如果我们在shapes $$\mathcal{M}, \mathcal{N}$$上还定义了别的线性算子（注意是算子，是函数与函数的映射，而不是函数），那么functional maps还可以与这些线性算子结合提供约束。比如说，对于对称的shape $$\mathcal{M}$$，如果我们有一个pointwise mapping$$S: \mathcal{M} \rightarrow \mathcal{M}$$来表示对称性，那么我们就可以定义一个symmetry operator $$S_{\mathcal{F}}: \mathcal{F}(\mathcal{M}, \mathbb{R}) \rightarrow \mathcal{F}(\mathcal{M}, \mathbb{R})$$，将定义在$$\mathcal{M}$$上的任意函数$$f: \mathcal{M} \rightarrow \mathbb{R}$$映射到另一个定义在$$\mathcal{M}$$的函数$$S_{\mathcal{F}}(f)$$上：$$S_{\mathcal{F}}(f) = f(S^{-1}(x)), \forall x \in \mathcal{M}$$。另一个例子是Laplace-Beltrami算子，即热传导算子。

实际上，对于任意定义在$$\mathcal{M}$$上的线性算子$$S_{\mathcal{F}}$$，和定义在$$\mathcal{N}$$上的线性算子$$R_{\mathcal{F}}$$，下述的约束都需要成立：

$$\lVert R_{\mathcal{F}}C - CS_{\mathcal{F}} \rVert = 0$$

> 注意，$$S_{\mathcal{F}}$$和$$R_{\mathcal{F}}$$是同样的算子，区别仅仅在于一个定义在$$\mathcal{M}$$上，一个定义在$$\mathcal{N}$$上。因为我们这里想要强调的是functional map $$C$$和一般性线性算子的交换律，但交换了线性算子和$$C$$的位置之后，因为$$C$$是将$$\mathcal{M}$$上的函数映射到$$\mathcal{N}$$上的函数的算子，所以那个线性算子也需要改变其所定义在的shape。而且这样一个约束是用来在优化求解functional map $$C$$的时候用的，因此我们已经有个前提假设是这两个shapes $$\mathcal{M}, \mathcal{N}$$存在ground truth pointwise mapping以及ground truth functional mapping，因此其上定义的相同的线性算子就会有类似的效果，即如果该线性算子是上面对称性算子的例子，那么$$\mathcal{M}, \mathcal{N}$$就都是对称的，这个算子在两个shapes上的作用是相同的。

> 为什么上述约束$$\lVert R_{\mathcal{F}}C - CS_{\mathcal{F}} \rVert = 0$$是线性的？将$$C$$展平为向量$$\boldsymbol{c}$$，利用Kronecker积的经典恒等式：$$\text{vec}(ABC) = (C^{\top} \otimes A)\text{vec}(B)$$。$$R_{\mathcal{F}}C = R_{\mathcal{F}} \cdot C \cdot I$$，从而$$\text{vec}(R_{\mathcal{F}}C)) = (I \otimes R_{\mathcal{F}}) \boldsymbol{c}$$，$$CS_{\mathcal{F}} = I \cdot C \cdot S_{\mathcal{F}}$$，从而$$\text{vec}(CS_{\mathcal{F}}) = (S_{\mathcal{F}}^{\top} \otimes I) \boldsymbol{c}$$，从而约束即为：\text{vec}(R_{\mathcal{F}}C - CS_{\mathcal{F}} \rVert) = \left[(I \otimes R_{\mathcal{F}}) - (S_{\mathcal{F}}^{\top} \otimes I) \right] \boldsymbol{c} = 0$$，为关于$$\boldsymbol{c}$$的线性约束。


**Functional maps的正交约束**

如果shapes $$\mathcal{M}, \mathcal{N}$$上的函数空间选择的基是正交基，满足$$\phi_{i}^{\top} \phi_j = 0, i \neq j; \lVert \phi_{i} \rVert = 1 \forall i$$，那么其对应的functional maps，需要是规范正交的（orthonormal），即：

$$C^{\top} C = I$$

在设计优化算法时，上述约束均可以作为functional maps的约束加入。


#### (4). Functional maps和Pointwise maps之间的互相转换

由之前的推导可知，如果给定两个shapes $$\mathcal{M}, \mathcal{N}$$之间的bijective pointwise mapping $$T$$，两个shapes上函数空间的bases $$\lbrace \phi_i^{\mathcal{M}} \rbrace_{i=1}^{\infty}, \lbrace \phi_i^{\mathcal{N}} \rbrace_{i=1}^{\infty}$$ ，其functional representation $$C = \lbrace C_{ij} \rbrace_{i,j=1}^{\infty}$$即由$$\langle T_{\mathcal{F}}(\phi_i^{\mathcal{M}}), \phi_j^{\mathcal{N}} \rangle$$给出，而$$T_{\mathcal{F}}(\phi_i^{\mathcal{M}})(x) = \phi_i^{\mathcal{M}}(T^{-1}(x)), \forall x \in \mathcal{N}$$。

但由functional map $$C$$去计算pointwise map $$T$$就不那么容易了。按照前面的推导，一个最直接的办法就是对于$$\mathcal{M}$$上的每个点$$x$$，在$$\mathcal{M}$$上定义一个indicator function $$f$$或者一个峰值位于$$x$$的高斯分布，从而利用functional map将$$f$$的系数向量$$\boldsymbol{a}$$转换为$$\mathcal{N}$$上所对应的函数的系数向量$$C \boldsymbol{a}$$，再将其转换为函数$$g$$，然后找到其最大值所对应的点，即为$$x$$的对应点。但该算法需要的计算量为$$O(\lvert \mathcal{V}_{\mathcal{M}} \rvert \cdot \lvert \mathcal{V}_{\mathcal{N}} \rvert)$$。

如果使用规范正交基，比如Laplace-Beltrami算子的eigenfunctions构成的基，$$\lbrace \phi_i^{\mathcal{M}} \rbrace_{i=1}^{\infty}, \lbrace \phi_i^{\mathcal{N}} \rbrace_{i=1}^{\infty}$$，那么对于$$\mathcal{M}$$上的每个点$$x$$，在$$\mathcal{M}$$上定义的indicator function $$f$$的系数向量即为$$\boldsymbol{a} = (\phi_1^{\mathcal{M}}(x), \phi_2^{\mathcal{M}}(x), \cdots)$$，从而如果将Laplace-Beltrami算子的eigenfunctions按行排列成一个矩阵$$\Phi^{\mathcal{M}}$$，即每一行是一个eigenfunctions，那么每一列即对应一个点的indicator function的系数向量，从而$$C\Phi^{\mathcal{M}}$$对应每个$$\mathcal{M}$$上的点的indicator function由functional map映射之后在$$\mathcal{N}$$上的indicator function的系数向量，而类似的$$\Phi^{\mathcal{N}}$$每一列也对应着$$\mathcal{N}$$上的一个点的indicator function的系数向量，从而我们只需要对于$$C\Phi^{\mathcal{M}}$$的每一列，找到最接近的$$\Phi^{\mathcal{N}}$$的某一列即可，计算复杂度为$$O(\lvert \mathcal{V}_{\mathcal{M}} \rvert \text{log} \lvert \mathcal{V}_{\mathcal{M}} \rvert + \lvert \mathcal{V}_{\mathcal{N}} \rvert \text{log} \lvert \mathcal{V}_{\mathcal{N}} \rvert)$$。


#### (5). 利用Functional map进行shape matching的算法

在介绍matching算法之前，先介绍一个post-processing iterative refinement，其利用到第(4)节里我们关于functional map $$C$$的知识，对functional map $$C$$进行进一步的微调。假设我们已经得到了一个初始的functional map $$C_0$$，且其是由某个pointwise map得来的。由上一节我们可以知道$$C_0 \Phi^{\mathcal{M}}$$的每一列都应该分别和$$\Phi^{\mathcal{N}}$$某一列相同。如果我们将$$\Phi^{\mathcal{M}}$$和$$\Phi^{\mathcal{N}}$$每一列看作一个点，那么其表示两个维度为$$C_0$$维度大小的点云，而$$C_0$$尝试去rigidly align这两个点云，而我们又可知$$C_0$$还需要是orthonormal的，因此我们的优化目标就是寻找一个orthonormal的矩阵来对$$\Phi^{\mathcal{M}}$$和$$\Phi^{\mathcal{N}}$$进行rigid alignment：

* 对于$$C_0\Phi^{\mathcal{M}}$$的每一列$$x$$，找到其最接近的$$\Phi^{\mathcal{N}}$$的某一列，$$\tilde{x}$$
* 对于上面步骤找到的每一列和其对应的最接近的列，优化下述目标函数，其中$$C$$是orthonormal矩阵：$$\sum \lVert Cx - \tilde{x} \rVert$$
* 令$$C_0 = C$$，并重复前两步直至收敛

利用functional maps进行shape matching的算法流程如下：

* 对于shapes $$\mathcal{M}, \mathcal{N}$$，计算其per-vertex的descriptors
* 将其所有的线性约束进行整合，包括descriptor perservation约束，landmark point correspondence约束，segment correspondence约束，以及线性算子交换律约束，这些约束都是有关functional map的线性约束，将其整合为一个线性系统，利用least squared error作为objective来解出functional map $$C$$
* 利用post-processing iterative refinemenet来refine得到的$$C$$
* 利用第(4)节里的方法来计算得到pointwise map


#### (6). Additional阅读

$$\left[$$[5](https://dl.acm.org/doi/10.1145/3084873.3084877)$$\right]$$和$$\left[$$[6]([https://dl.acm.org/doi/10.1145/3084873.3084877](https://www.sciencedirect.com/science/chapter/handbook/abs/pii/S1570865918300012))$$\right]$$提供了更为完整和详尽的解释。

[An Introductory Perspective on Functional Maps](https://xieyizheng.com/media/papers/intro-functional-maps/Report_Functional_Maps.pdf)也提供了一个解释。



> 参考文献：
> * $$\left[1 \right]$$ Ovsjanikov, Maks, et al. "Functional maps: a flexible representation of maps between shapes." ACM Transactions on Graphics (ToG) 31.4 (2012): 1-11.
> * $$\left[2 \right]$$ Jain, Varun, Hao Zhang, and Oliver Van Kaick. "Non-rigid spectral correspondence of triangle meshes." International Journal of Shape Modeling 13.01 (2007): 101-124.
> * $$\left[3 \right]$$ Mateus, Diana, et al. "Articulated shape matching using laplacian eigenfunctions and unsupervised point registration." 2008 IEEE Conference on Computer Vision and Pattern Recognition. IEEE, 2008.
> * $$\left[4 \right]$$ Ovsjanikov, Maks, Jian Sun, and Leonidas Guibas. "Global intrinsic symmetries of shapes." Computer graphics forum. Vol. 27. No. 5. Oxford, UK: Blackwell Publishing Ltd, 2008.
> * $$\left[5 \right]$$ Ovsjanikov, Maks, et al. "Computing and processing correspondences with functional maps." SIGGRAPH ASIA 2016 Courses. 2016. 1-60.
> * $$\left[6 \right]$$ Ovsjanikov, Maks. "Shape correspondence and functional maps." Handbook of Numerical Analysis. Vol. 19. Elsevier, 2018. 91-118.





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
