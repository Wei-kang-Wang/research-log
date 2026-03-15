---
layout: post
comments: True
title: "形状分析（二）：Laplace-Beltrami Operator"
date: 2026-03-14 02:09:00
tags: shape_analysis
---

<!--more-->

{: class="table-of-content"}
* TOC
{:toc}

---

### 1. 泛函分析里的一些基本概念

首先，我们介绍一些基本概念。

**Banach空间**：给定一个流形 $$S$$，$$X$$表示定义在$$S$$上的实值函数空间，且定义该函数空间的范数$$\lVert \  \cdot \  \rVert$$。如果$$X$$里的任意柯西列都收敛到$$X$$里的某个函数，其中$$X$$里的柯西列表示一系列函数$$f_1, f_2, \cdots$$，且$$\lim_{n,m \rightarrow \infty} \lVert f_n - f_m \rVert = 0$$，那么该函数空间$$X$$是完备的。完备的向量空间称为Banach空间。

**Hilbert空间**：如果一个Banach空间里的范数是由内积定义的，即$$\lVert f \rVert = \sqrt{\langle f, f \rangle}$$，那这个空间就是Hilbert空间。

> 常见的内积可以定义为$$\langle f,g \rangle = \int_{S} f(x)g(x) dx$$，其定义的范数叫做$$L_2$$范数。

**算子（operator）**：在泛函分析里，算子被定义为函数的函数，比如将函数空间$$X$$映射到函数空间$$X$$，$$L: X \rightarrow X$$将$$f \in X$$映射到$$Lf \in X$$。一个算子$$L$$被称作线性，如果其对于所有的$$f \in X, \lambda \in \mathbb{R}$$，均满足$$L(\lambda f) = \lambda f$$。

**算子的特征函数（eigenfunctions）**：一个算子$$L$$的特征函数eigenfunctions，是满足$$Lf = \lambda f$$的函数$$f$$，其中和$$f$$对应的$$\lambda$$称为该特征函数$$f$$对应的特征值。也就是说，将算子$$L$$用在其特征函数上，等价于简单的缩放该函数。

**Hermitian算子**：一个算子如果满足对函数$$f,g \in X$$，均满足$$\langle Lf,g \rangle = \langle f, Lg \rangle$$，则称其为Hermitian算子，或者说该算子满足Hermitian symmetry。Hermitian算子的一个重要特性是其不同eigenvalues对应的eigenfunctions都相互垂直（$$\lambda \langle f,g \rangle = \langle \lambda f, g \rangle = \langle Lf,g \rangle = \langle f, Lg \rangle = \langle f, \mu g \rangle = \mu \langle f,g \rangle$$，且$$\lambda \neq \mu$$，说明$$\langle f,g \rangle=0$$）。


### 2. 连续流形上的Laplace-Beltrami算子

Laplace-Beltrami算子的一般性定义为：

**Laplace-Beltrami算子（一般性定义）**：Laplace-Beltrami算子是Laplace算子在欧氏空间流形上针对实值函数（real-valued function）的拓展。具体来说，Laplace-Beltrami算子是定义在欧氏空间$$\mathbb{R}^n$$里流形（manifold）上的实值函数（real-valued functions）的Laplace算子。

其另一个常见定义为：

**Laplace-Beltrami算子（形状分析里的常见定义）**：$$\mathcal{M}$$是一个紧且联通（compact以及connected）的2维流形，$$L^2(\mathcal{M}) = \lbrace f: \mathcal{M} \rightarrow \mathbb{R} \vert \langle f, f \rangle_{\mathcal{M}} = \int_{\mathcal{M}} f^2(x) dx < \infty \rbrace$$表示的是在$$\mathcal{M}$$上定义的所有平方可积函数构成的函数空间。$$\mathcal{M}$$上的Laplace-Beltrami算子$$\Delta_{\mathcal{M}}: L^2(\mathcal{M}) \rightarrow L^2(\mathcal{M})$$定义为：$$\Delta_{\mathcal{M}} f = - \text{div}_{\mathcal{M}} (\nabla_{\mathcal{M}} f)$$。

> 和一般性定义比，上述第二个定义局限了流形的定义，且局限了实值函数的定义范围，但可以看到很容易扩展到其他实值函数的情况。

> 注意，Laplace算子和Laplace-Beltrami算子都是将一个函数映射到另一个函数，其区别仅仅在于Laplace-Beltrami算子所操作的函数是定义在欧氏空间某流形上的，Laplace算子所操作的函数可以是欧氏空间内的任意函数。

Laplace-Beltrami算子有如下重要的性质：

* Laplace-Beltrami算子是Hermitian算子。
* 满足$$\Delta_{\mathcal{M}} \phi_i(x) = \lambda_i \phi_i(x), \forall x \in \mathcal{M}$$的所有eigenfunctions构成的集合$$\lbrace \phi_1, \phi_2, \cdots \rbrace$$构成函数空间$$L^2(\mathcal{M})$$的一组正交基，即任意$$f \in L^2(\mathcal{M})$$可以被表示为$$f(x) = \sum_{i=1}^{\infty} \langle f, \phi_i \rangle_{\mathcal{M}} \phi_i(x), \forall x \in \mathcal{M}$$。
* 对于三维欧氏空间内的二维流形，函数$$\vec{p}: \mathcal{M} \rightarrow \mathbb{R}^3$$是恒等嵌入，即对于任意点$$(x,y,z) \in \mathcal{M}, \vec{p}(x,y,z) = (x,y,z)$$。且$$\Delta_{\mathcal{M}}$$是该流形$$\mathcal{M}$$上的Laplace-Beltrami算子，那么$$\Delta_{\mathcal{M}} \vec{p} = -2H \vec{n}$$，其中$$\vec{n}$$是流形上每一点处的单位长度的法向量，$$H$$是流形上每一点处的mean curvature。
> 注意$$\vec{p}$$是一个向量函数，上述公式里的Laplace-Beltrami算子对$$\vec{p}$$的值的每个分量依次计算。

> 参考文献
> * $$\left[1\right]$$ Lévy, Bruno. "Laplace-beltrami eigenfunctions towards an algorithm that" understands" geometry." IEEE International Conference on Shape Modeling and Applications 2006 (SMI'06). IEEE, 2006.
> * $$\left[2\right]$$ Nagar, Rajendra, and Shanmuganathan Raman. "Fast and accurate intrinsic symmetry detection." Proceedings of the European Conference on Computer Vision (ECCV). 2018.


### 3. 离散情况下的Laplace-Beltrami算子

在实际应用中，我们所操作的都是由$$\mathcal{M}=(\mathcal{V}, \mathcal{E})$$所表示的离散的三维网格来近似2维流形，从而我们需要将上述定义在连续流形上的Laplace-Beltrami算子离散化到离散的三维网格上。但因为Laplace-Beltrami算子是将函数映射到函数的算子，所以这样的离散化过程并不trivial。

我们先给出一种最常见的离散化方法，即cotangent Laplacian（参考*Pinkall & Polthier, 1993, Meyer et al., 2003*）的结果。

**定理一**：对于三角网格$$\mathcal{M}$$，对于每个顶点$$i$$，离散Laplace-Beltrami算子作用于任意函数$$f$$的结果为：

$$(\Delta f)_i = \frac{1}{2A_i} \sum_{j \in \mathcal{N}(i)} (\text{cot} \alpha_{ij} + \text{cot} \beta_{ij}) (f_i - f_j)$$

其中$$\alpha_{ij}, \beta_{ij}$$是边$$(i,j)$$对面的两个角，$$A_i$$是顶点$$i$$关联的面积，$$N(i)$$是$$i$$的1环邻域，如下图所示：

![1]({{ '/assets/images/laplace_beltrami.png' | relative_url }}){: width=800px style="float:center"} 

上述结果还可以写成矩阵形式。首先定义两个个大小为$$\lvert V \rvert \times \lvert V \rvert$$的矩阵

刚度矩阵或cotangent矩阵$$L$$：

$$
L_{ij} = 
\begin{cases}
    -\frac{1}{2}(\text{cot} \alpha_{ij} + \text{cot} \beta_{ij}) & (i,j) \in \mathcal{E} \\
    -\sum_{k \neq i} L_{ik}  & i=j \\
    0   & \text{otherwise}
\end{cases}
$$

质量矩阵$$M$$：

$$M = \text{diag}(A_1, A_2, \cdots, A_V)$$

其中$$A_i$$是顶点$$i$$关联的面积（Voronoi面积或者barycentric面积的$$1/3$$）。

**定理二**：离散Laplace-Beltrami算子为：$$\Delta f = M^{-1}L f$$

我们成功将Laplace-Beltrami算子定义在了离散的三维网格上，且用一个矩阵$$M^{-1}L$$来表示它。从而如果我们要计算Laplace-Beltrami算子的eigenfunctions，即寻找$$\Delta \phi = \lambda \phi$$，我们只需要寻找$$M^{-1}L \phi = \lambda \phi$$，其中$$\phi$$为一个长度为$$\lvert \mathcal{V} \rVert$$的向量，用来表示在每个顶点上的值。即转化成了计算矩阵$$$$M^{-1}L$$的特征值和特征向量。

将cotangent matrix公式显式地用于曲面上的离散Laplace-Beltrami算子，是1993年Pinkall和Polthier的论文"Computing discrete minimal surfaces and their conjugates"。他们利用cotangent公式给出了离散平均曲率向量的函数表示，并用它来计算离散极小曲面。这篇论文被广泛认为是将cotangent Laplacian引入计算几何和图形学领域的开创性工作。后来Desbrun, Meyer, Schröder, Barr在1999年的 SIGGRAPH论文"Implicit fairing of irregular meshes using diffusion and curvature flow"中进一步推广了这个公式的应用，而Meyer et al. 2003年的文章"Discrete differential-geometry operators for triangulated 2-manifolds"系统整理了包括Voronoi面积归一化在内的完整形式，成为后续文献中被引用最多的版本。

> 参考文献
> * $$\left[1 \right]$$ Pinkall, Ulrich, and Konrad Polthier. "Computing discrete minimal surfaces and their conjugates." Experimental mathematics 2.1 (1993): 15-36.
> * $$\left[2 \right]$$ Desbrun, Mathieu, et al. "Implicit fairing of irregular meshes using diffusion and curvature flow." Proceedings of the 26th annual conference on Computer graphics and interactive techniques. 1999.
> * $$\left[3 \right]$$ Meyer, Mark, et al. "Discrete differential-geometry operators for triangulated 2-manifolds." Visualization and mathematics III. Berlin, Heidelberg: Springer Berlin Heidelberg, 2003. 35-57.


### 4. 一些证明

下面我们来证明定理一。cotangent Laplacian的推导方式有多种途径，最常用的是利用有限元方法，从Dirichlet能量的变分出发。

第一步：在三角网格$$\mathcal{M}$$的每个三角形内部计算梯度

定义在$$\mathcal{M}$$上的函数$$f$$在每个顶点$$i$$的取值为$$f_i$$，我们来在三角形内部做线性插值，从而给三角形内部那些没有$$f$$定义的点赋值。取一个三角形$$T$$，顶点为$$p_1, p_2, p_3$$，对应的函数值为$$f_1, f_2, f_3$$。那么对于三角形内部任何一点$$f$$，可以用重心坐标$$(\lambda_1, \lambda_2, \lambda_3)$$来表示$$f = \lambda_1 f_1 + \lambda_2 f_2 + \lambda_3 f_3$$，其中$$\lambda_1+\lambda_2 + \lambda_3 = 1$$。从而$$\nabla f = f_1 \nabla \lambda_1 + f_2 \nabla \lambda_2 + f_2 \nabla \lambda_3$$。对于$$\lambda_1$$，其在点$$p_1$$处取值为1，在对边处取值为0，


对于流形$$\mathcal{M}$$上的函数$$f$$，其Dirichlet能量定义为：$$E(f) = \frac{1}{2}\int_{\mathcal{M}} \lVert \nabla f \rVert^2 dA$$。我们首先来计算$$E$$的$$L^2$$梯度。

>什么叫$$E$$的$$L^2$$梯度？假设$$f$$是流形上的函数，考虑对$$f$$做一个微小扰动$$f \rightarrow f + \epsilon \phi$$，其中$$\phi$$是任意光滑的流形上的函数。$$E$$的一阶变分为：$$\mathop{\lim}\limits_{\epsilon \rightarrow 0} (E(f+\epsilon \phi) - E(f)) / \epsilon$$。如果存在某个函数$$g$$，使得对于任意的$$\phi$$，都有$$E$$的一阶变分等于$$\langle g, \phi \rangle_{L^2} = \int_{\mathcal{M}} g \phi dA$$，那么就称$$g$$是$$E$$在$$f$$处的$$L^2$$梯度，记作$$\text{grad}_{L^2}E(f) = g$$。

依据$$E$$的定义，我们来计算它的一阶变分：

$$E(f + \epsilon \phi) = \frac{1}{2} \int_{\mathcal{M}} \lVert \nabla (f + \epsilon \phi) \rVert^2 dA = \frac{1}{2} \int_{\mathcal{M}} \langle \nabla f + \epsilon \nabla \phi, \nabla f + \epsilon \nabla \phi \rangle dA = \frac{1}{2} \int_{\mathcal{M}} \left[ \lVert \nabla f \rVert^2 + 2\epsilon \langle \nabla f, \nabla \phi \rangle + \epsilon^2 \lVert \nabla \phi \rVert^2 \right] dA$$

根据一阶变分的定义

$$\mathop{\lim}\limits_{\epsilon \rightarrow 0} \frac{E(f+\epsilon \phi) - E(f)}{\epsilon} = \int_{\mathcal{M}} \langle \nabla f, \nabla \phi \rangle dA$$

由Green第一恒等式，可以得到：

$$\int_{\mathcal{M}} \langle \nabla f, \nabla \phi \rangle dA = - \int_{\mathcal{M}} (\text{div} (\nabla f)) \phi dA = \langle -\text{div}(\nabla f), \phi \rangle_{L^2}$$

>具体推导过程为：考虑向量场$$\phi \nabla f$$，利用Leibniz律，对其取散度，可以得到$$\text{div}(\phi \nabla f) = \langle \nabla \phi, \nabla f \rangle + \phi \text{div}(\nabla f)$$。对两边在$$\mathcal{M}$$上积分，左边由散度定理变为边界积分（因为$$\mathcal{M}$$无边界，所以为0）：$$0 = \int_{\mathcal{M}} \langle \nabla \phi, \nabla f \rangle dA + \int_{\mathcal{M}} \phi \text{div}(\nabla f) dA$$，即$$\int_{\mathcal{M}} \langle \nabla \phi, \nabla f \rangle dA = -\int_{\mathcal{M}} \phi \text{div}(\nabla f) dA = \langle -\text{div}(\nabla f), \phi \rangle_{L^2}$$。

结合上面两个结论，可以得到：$$\text{grad}_{L^2}E(f) = -\text{div}(\nabla f), \phi \rangle_{L^2} = - \Delta_{\mathcal{M}} f$$，即**$$\Delta_{\mathcal{M}}f$$是函数$$f$$在流形$$\mathcal{M}$$上的Dirichlet能量的负$$L^2$$梯度**。
























