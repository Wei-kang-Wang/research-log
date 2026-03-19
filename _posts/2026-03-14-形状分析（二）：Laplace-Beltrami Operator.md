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

Laplace算子是将函数映射到函数的算子，其是微分几何中一个常见的重要概念，而Laplace-Beltrami算子是定义在流形上的Laplace算子。在shape analysis里，Laplace-Beltrami算子也是一种常见的分析工具。下面我们先介绍general的Laplace-Beltrami算子，再介绍将其拓展到离散网格上的结果，最后探讨其一些性质以及其在shape analysis里很多任务上的使用。

### 1. 连续流形上的Laplace-Beltrami算子

首先，我们介绍泛函分析里的一些基本概念。

**Banach空间**：给定一个流形 $$S$$，$$X$$表示定义在$$S$$上的实值函数空间，且定义该函数空间的范数$$\lVert \  \cdot \  \rVert$$。如果$$X$$里的任意柯西列都收敛到$$X$$里的某个函数，其中$$X$$里的柯西列表示一系列函数$$f_1, f_2, \cdots$$，且$$\lim_{n,m \rightarrow \infty} \lVert f_n - f_m \rVert = 0$$，那么该函数空间$$X$$是完备的。完备的向量空间称为Banach空间。

**Hilbert空间**：如果一个Banach空间里的范数是由内积定义的，即$$\lVert f \rVert = \sqrt{\langle f, f \rangle}$$，那这个空间就是Hilbert空间。

> 常见的内积可以定义为$$\langle f,g \rangle = \int_{S} f(x)g(x) dx$$，其定义的范数叫做$$L_2$$范数。

**算子（operator）**：在泛函分析里，算子被定义为函数的函数，比如将函数空间$$X$$映射到函数空间$$X$$，$$L: X \rightarrow X$$将$$f \in X$$映射到$$Lf \in X$$。一个算子$$L$$被称作线性，如果其对于所有的$$f \in X, \lambda \in \mathbb{R}$$，均满足$$L(\lambda f) = \lambda f$$。

**算子的特征函数（eigenfunctions）**：一个算子$$L$$的特征函数eigenfunctions，是满足$$Lf = \lambda f$$的函数$$f$$，其中和$$f$$对应的$$\lambda$$称为该特征函数$$f$$对应的特征值。也就是说，将算子$$L$$用在其特征函数上，等价于简单的缩放该函数。

**Hermitian算子**：一个算子如果满足对函数$$f,g \in X$$，均满足$$\langle Lf,g \rangle = \langle f, Lg \rangle$$，则称其为Hermitian算子，或者说该算子满足Hermitian symmetry，或者称该算子自伴（self-adjoint）。Hermitian算子的一个重要特性是其不同eigenvalues对应的eigenfunctions都相互垂直（$$\lambda \langle f,g \rangle = \langle \lambda f, g \rangle = \langle Lf,g \rangle = \langle f, Lg \rangle = \langle f, \mu g \rangle = \mu \langle f,g \rangle$$，且$$\lambda \neq \mu$$，说明$$\langle f,g \rangle=0$$）。


Laplace-Beltrami算子的一般性定义为：

**Laplace-Beltrami算子（一般性定义）**：Laplace-Beltrami算子是Laplace算子在欧氏空间流形上针对实值函数（real-valued function）的拓展。具体来说，Laplace-Beltrami算子是定义在欧氏空间$$\mathbb{R}^n$$里流形（manifold）上的实值函数（real-valued functions）的Laplace算子。

其另一个常见定义为：

**Laplace-Beltrami算子（形状分析里的常见定义）**：$$\mathcal{M}$$是一个紧且联通（compact以及connected）的2维流形，$$L^2(\mathcal{M}) = \lbrace f: \mathcal{M} \rightarrow \mathbb{R} \vert \langle f, f \rangle_{\mathcal{M}} = \int_{\mathcal{M}} f^2(x) dx < \infty \rbrace$$表示的是在$$\mathcal{M}$$上定义的所有平方可积函数构成的函数空间。$$\mathcal{M}$$上的Laplace-Beltrami算子$$\Delta_{\mathcal{M}}: L^2(\mathcal{M}) \rightarrow L^2(\mathcal{M})$$定义为：$$\Delta_{\mathcal{M}} f = - \text{div}_{\mathcal{M}} (\nabla_{\mathcal{M}} f)$$。

> 和一般性定义比，上述第二个定义局限了流形的定义，且局限了实值函数的定义范围，但可以看到很容易扩展到其他实值函数的情况。

> 注意，Laplace算子和Laplace-Beltrami算子都是将一个函数映射到另一个函数，其区别仅仅在于Laplace-Beltrami算子所操作的函数是定义在欧氏空间某流形上的，Laplace算子所操作的函数可以是欧氏空间内的任意函数。

Laplace-Beltrami算子有如下重要的性质：

* Laplace-Beltrami算子在闭曲面上是Hermitian算子。该结果可以通过下面第3节里介绍的Green第一恒等式证明，同样也要用到在闭曲面上，曲面的边界是空集这样的结果，从而得到$$\langle \Delta f, g \rangle$$和$$\langle \Delta g, f \rangle$$之间的关系。
* 满足$$\Delta_{\mathcal{M}} \phi_i(x) = \lambda_i \phi_i(x), \forall x \in \mathcal{M}$$的所有eigenfunctions构成的集合$$\lbrace \phi_1, \phi_2, \cdots \rbrace$$构成函数空间$$L^2(\mathcal{M})$$的一组正交基，即任意$$f \in L^2(\mathcal{M})$$可以被表示为$$f(x) = \sum_{i=1}^{\infty} \langle f, \phi_i \rangle_{\mathcal{M}} \phi_i(x), \forall x \in \mathcal{M}$$。这个结果之所以成立，是因为Laplace-Beltrami算子是Hermitian算子，且是椭圆算子，谱定理保证了它的特征函数构成$$L_2$$意义下的一组正交基。
* 对于三维欧氏空间内的二维流形，函数$$\vec{p}: \mathcal{M} \rightarrow \mathbb{R}^3$$是恒等嵌入，即对于任意点$$(x,y,z) \in \mathcal{M}, \vec{p}(x,y,z) = (x,y,z)$$。且$$\Delta_{\mathcal{M}}$$是该流形$$\mathcal{M}$$上的Laplace-Beltrami算子，那么$$\Delta_{\mathcal{M}} \vec{p} = -2H \vec{n}$$，其中$$\vec{n}$$是流形上每一点处的单位长度的法向量，$$H$$是流形上每一点处的mean curvature。
> 注意$$\vec{p}$$是一个向量函数，上述公式里的Laplace-Beltrami算子对$$\vec{p}$$的值的每个分量依次计算。

> 参考文献
> * $$\left[1\right]$$ Lévy, Bruno. "Laplace-beltrami eigenfunctions towards an algorithm that" understands" geometry." IEEE International Conference on Shape Modeling and Applications 2006 (SMI'06). IEEE, 2006.
> * $$\left[2\right]$$ Nagar, Rajendra, and Shanmuganathan Raman. "Fast and accurate intrinsic symmetry detection." Proceedings of the European Conference on Computer Vision (ECCV). 2018.


### 2. 离散情况下的Laplace-Beltrami算子

在实际应用中，我们所操作的都是由$$\mathcal{M}=(\mathcal{V}, \mathcal{E})$$所表示的离散的三维网格来近似2维流形，从而我们需要将上述定义在连续流形上的Laplace-Beltrami算子离散化到离散的三维网格上。但因为Laplace-Beltrami算子是将函数映射到函数的算子，所以这样的离散化过程并不trivial。

我们先给出一种最常见的离散化方法，即cotangent Laplacian（参考*Pinkall & Polthier, 1993, Meyer et al., 2003*）的结果。

**定理一**：对于三角网格$$\mathcal{M}$$，对于每个顶点$$i$$，离散Laplace-Beltrami算子作用于任意函数$$f$$的结果为：

$$(\Delta f)_i = \frac{1}{2A_i} \sum_{j \in \mathcal{N}(i)} (\text{cot} \alpha_{ij} + \text{cot} \beta_{ij}) (f_i - f_j)$$

其中$$\alpha_{ij}, \beta_{ij}$$是边$$(i,j)$$对面的两个角，$$A_i$$是顶点$$i$$关联的面积，$$N(i)$$是$$i$$的1环邻域，如下图所示：

![1]({{ '/assets/images/laplace_beltrami.png' | relative_url }}){: width=100px style="float:center"} 

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

我们成功将Laplace-Beltrami算子定义在了离散的三维网格上，且用一个矩阵$$M^{-1}L$$来表示它。从而如果我们要计算Laplace-Beltrami算子的eigenfunctions，即寻找$$\Delta \phi = \lambda \phi$$，我们只需要寻找$$M^{-1}L \phi = \lambda \phi$$，其中$$\phi$$为一个长度为$$\lvert \mathcal{V} \rvert$$的向量，用来表示在每个顶点上的值。即转化成了计算矩阵$$M^{-1}L$$的特征值和特征向量。

将cotangent matrix公式显式地用于曲面上的离散Laplace-Beltrami算子，是1993年Pinkall和Polthier的论文"Computing discrete minimal surfaces and their conjugates"。他们利用cotangent公式给出了离散平均曲率向量的函数表示，并用它来计算离散极小曲面。这篇论文被广泛认为是将cotangent Laplacian引入计算几何和图形学领域的开创性工作。后来Desbrun, Meyer, Schröder, Barr在1999年的 SIGGRAPH论文"Implicit fairing of irregular meshes using diffusion and curvature flow"中进一步推广了这个公式的应用，而Meyer et al. 2003年的文章"Discrete differential-geometry operators for triangulated 2-manifolds"系统整理了包括Voronoi面积归一化在内的完整形式，成为后续文献中被引用最多的版本。

> 参考文献
> * $$\left[1 \right]$$ Pinkall, Ulrich, and Konrad Polthier. "Computing discrete minimal surfaces and their conjugates." Experimental mathematics 2.1 (1993): 15-36.
> * $$\left[2 \right]$$ Desbrun, Mathieu, et al. "Implicit fairing of irregular meshes using diffusion and curvature flow." Proceedings of the 26th annual conference on Computer graphics and interactive techniques. 1999.
> * $$\left[3 \right]$$ Meyer, Mark, et al. "Discrete differential-geometry operators for triangulated 2-manifolds." Visualization and mathematics III. Berlin, Heidelberg: Springer Berlin Heidelberg, 2003. 35-57.

**性质一**：由之前的结论，Laplace-Beltrami算子的eigenfunctions构成闭曲面的一个正交基，从而对于闭曲面（即mesh）上定义的任意函数$$f$$，$$f = c_0 f_0 + c_1 f_1 + \cdots$$，其中$$c_i = \int_{\mathcal{M}} \langle f, \phi_i \rangle dp$$为系数，$$0 = \lambda_0 < \lambda_1 < \cdots $$为eigenvalues，$$\phi_0, \phi_1, \cdots$$是对应的eigenfunctions。注意，因为一个mesh的离散Laplace-Beltrami算子$$\Delta = M^{-1}L$$的特征值是0，从而$$\lambda_0=0$$，其对应的$$\phi_0$$为一个常数向量，且因为$$M^{-1}L$$是一个对称实数矩阵，其特征值都是非负实数。


### 3. 上述定理的证明

上述cotangent Laplacian结果的推导方式有多种途径，最常用的是利用有限元方法，从Dirichlet能量的变分出发。对于流形$$\mathcal{M}$$上的函数$$f$$，其Dirichlet能量定义为：

$$E(f) = \frac{1}{2}\int_{\mathcal{M}} \lVert \nabla f \rVert^2 dA$$

**第一步：在三角网格$$\mathcal{M}$$的每个三角形内部计算梯度**

定义在$$\mathcal{M}$$上的函数$$f$$在每个顶点$$i$$的取值为$$f_i$$，我们来在三角形内部做线性插值，从而给三角形内部那些没有$$f$$定义的点赋值。取一个三角形$$T$$，顶点为$$p_1, p_2, p_3$$，对应的函数值为$$f_1, f_2, f_3$$。那么对于三角形内部任何一点$$f$$，可以用坐标$$(\lambda_1, \lambda_2, \lambda_3)$$来表示$$f = \lambda_1 f_1 + \lambda_2 f_2 + \lambda_3 f_3$$，其中$$\lambda_1+\lambda_2 + \lambda_3 = 1$$。

上述这种线性插值方式，可以让$$\lambda_1, \lambda_2, \lambda_3$$是关于坐标$$(x,y,z)$$的线性函数。具体证明是设$$f=(x,y,z), f_1 = (x_1, y_1, z_1), f_2 = (x_2, y_2, z_2), f_3 = (x_3, y_3, z_3)$$，那么$$x = \lambda_1 x_1 + \lambda_2 x_2 + \lambda_3 x_3, y = \lambda_1 y_1 + \lambda_2 y_2 + \lambda_3 y_3, z = \lambda_1 z_1 + \lambda_2 z_2 + \lambda_3 z_3, \lambda_1+\lambda_2 + \lambda_3 = 1$$这四个方程描述三个未知数$$\lambda_1, \lambda_2, \lambda_3$$。但因为三个点$$f_1,f_2,f_3$$共面，这四个方程中只有三个是独立的，恰好唯一确定$$\lambda_1, \lambda_2, \lambda_3$$。消去第四个方程带入前面三个，写成矩阵的形式，可以得到：

$$
\begin{pmatrix}
x - x_3 \\
y - y_3 \\
z - z_3 \\
\end{pmatrix} = \begin{pmatrix} x_1 - x_3 & x_2 - x_3 \\ y_1 - y_3 & y_2 - y_3 \\ z_1 - z_3 & z_2 - z_3 \\ \end{pmatrix} \begin{pmatrix} \lambda_1 \\ \lambda_2 \end{pmatrix}
$$

记上述式子右侧的$$3 \times 2$$矩阵为$$E = \left[ \pmb{e}_{13} \  \pmb{e}_{23} \right]$$，其中$$\pmb{e}_{13} = f_1 - f_3, \pmb{e}_{23} = f2 - f3$$。上述方程的解为：

$$
\begin{pmatrix}
\lambda_1 \\
\lambda_2 \\
\end{pmatrix} = (E^{\top} E)^{-1}E^{\top}(f-f_3)
$$

其中$$(E^{\top} E)^{-1}E^{\top}$$是一个常数$$2 \times 3$$矩阵（只依赖于$$f_1, f_2, f_3$$），因此$$\lambda_1, \lambda_2$$都是关于$$f=(x,y,z)$$的仿射函数，即存在常数$$a_1, b_1, c_1, d_1, a_2, b_2, c_2, d_2$$满足：$$\lambda_1 = a_1x + b_1 y + c_1 z + d_1, \lambda_2 = a_2 x + b_2 y + c_2 z + d_2$$，从而$$\lambda_3$$也可以表示为这种形式。从而$$\lambda_1, \lambda_2, \lambda_3$$关于$$(x,y,z)$$仿射线性。

下面我们来推导$$\nabla \lambda_i$$的具体结果。

首先，$$f = \lambda_1 f_1 + \lambda_2 f_2 + \lambda_3 f_3$$，$$\lambda_1 + \lambda_2 + \lambda_3 = 1$$，从而可得：

$$f = \lambda_1 f_1 + \lambda_2 f_2 + (1-\lambda_1 - \lambda_2)f_3$$

$$f - f_3 = \lambda_1(f_1-f_3) + \lambda_2 (f_2-f_3)$$

两侧与$$(f_2-f_3)$$做叉乘，可得：

$$(f-f_3) \times (f_2-f_3) = \lambda_1 (f_1-f_3) \times (f_2 - f_3) + \lambda_2 (f_2 - f_3) \times (f_2 - f_3) = \lambda_1 (f_1-f_3) \times (f_2 - f_3)$$

两侧取模，右侧叉乘的模是三角形$$\triangle_{f_1f_2f_3}$$面积$$A(f_1,f_2,f_3)$$的两倍，左侧叉乘的模是三角形$$\triangle_{ff_2f_3}$$面积$$A(f,f_2,f_3)$$的两倍，从而：

$$\lambda_1 = \frac{A(f,f_2, f_3)}{A(f_1, f_2, f_3)}$$

> 注意，$$\lambda_1 \geq 0$$因为点$$f$$在三角形$$\triangle_{f_1f_2f_3}$$内部。

记$$A_T = A(f_1, f_2, f_3)$$。$$\lambda_1$$是关于$$f=(x,y,z)$$的函数

$$\lambda_1(f) = \frac{A(f,f_2, f_3)}{A(f_1, f_2, f_3)} = \frac{\left[ (f - f_2) \times (f_3 - f_2) \right] \cdot \pmb{n}}{2A} = \frac{(f \times \boldsymbol{e_{23}}) \cdot \boldsymbol{n} - (f_2 \times \boldsymbol{e_{23}}) \cdot \boldsymbol{n}}{2A} = \frac{(\boldsymbol{e_{23}} \times \boldsymbol{n}) \cdot f - (f_2 \times \boldsymbol{e_{23}}) \cdot \boldsymbol{n}}{2A}$$

其中$$\boldsymbol{e_{23}}$$是由点$$f_2$$指向$$f_3$$的向量，$$\boldsymbol{n}$$是垂直于三角形$$\triangle_{f_1f_2f_3}$$平面的单位法向量（三角形$$\triangle_{f_1f_2f_3}$$和三角形$$\triangle_{ff_2f_3}$$在同一个平面内）。

从而$$\nabla \lambda_1(f)$$对$$f=(x,y,z)$$的梯度为：

$$\nabla \lambda_1 = \frac{\boldsymbol{e_{23}} \times \boldsymbol{n}}{2A_T}$$

类似的，$$\nabla \lambda_2 = \frac{\boldsymbol{e_{31}} \times \boldsymbol{n}}{2A_T}, \nabla \lambda_3 = \frac{\boldsymbol{e_{12}} \times \boldsymbol{n}}{2A_T}$$

**第二步：计算每个三角形$$T$$面上的Dirichlet能量**

有了上述结果，我们就可以计算一个三角形$$T = \triangle_{f_1f_2f_3}$$内的Dirichlet能量了：

$$
\begin{align}
E_T &= \frac{1}{2} \int_{T} \lVert \nabla f \rVert^2 dA = \frac{1}{2} \int_{T} \lVert f_1 \nabla \lambda_1 + f_2 \nabla \lambda_2 + f_3 \nabla \lambda_3 \rVert^2 dA = \frac{1}{2} A_T \lVert f_1 \nabla \lambda_1 + f_2 \nabla \lambda_2 + f_3 \nabla \lambda_3 \rVert^2 \\
&= \frac{1}{2} A_T \langle  f_1 \nabla \lambda_1 + f_2 \nabla \lambda_2 + f_3 \nabla \lambda_3, f_1 \nabla \lambda_1 + f_2 \nabla \lambda_2 + f_3 \nabla \lambda_3 \rangle = \frac{1}{2} A_T \sum_{i=1}^3 f_i^2 \langle \nabla \lambda_i, \nabla \lambda_i \rangle + A_T \sum_{1 \leq i < j \leq 3} f_i f_j \langle \nabla \lambda_i, \nabla \lambda_j \rangle
\end{align}
$$

> 前一项对应三角形$$T$$的顶点，后一项对应三角形$$T$$的边。

**第三步：推导cotangent权重**

考虑$$\langle \nabla \lambda_i, \nabla \lambda_j \rangle$$，$$i \neq j$$的计算，以$$\langle \nabla \lambda_1, \nabla \lambda_2 \rangle$$为例，假设顶点$$f_3$$处的角大小为$$\theta_3$$，$$\langle \nabla \lambda_1, \nabla \lambda_2 \rangle$$的物理意义是两个向量的内积，其模长分别为点$$f_1,f_2$$对应的高的倒数，夹角为$$\pi - \theta_3$$：

$$\langle \nabla \lambda_1, \nabla \lambda_2 \rangle = \frac{1}{h_1h_2} \text{cos}(\pi - \theta_3)= -\frac{\text{cos}\theta_3}{h_1h_2}$$

而$$A_T = \frac{1}{2}\lVert \boldsymbol{e_{31}} \rVert \lVert \boldsymbol{e_{23}} \rVert \text{sin}(\theta_3)$$，且$$h_1 = 2A_T / \lVert \boldsymbol{e_{23}} \rVert, h_2 = 2A_T / \lVert \boldsymbol{e_{31}} \rVert$$，从而最终可以得到：$$A_T \langle \nabla \lambda_1, \nabla \lambda_2 \rangle = -\frac{1}{2} \text{cot} \theta_3$$，如下图左侧所示。

考虑$$A_T f_i^2 \langle \nabla \lambda_i, \nabla \lambda_i \rangle$$，$$i \in \lbrace 1,2,3 \rbrace$$的计算，以$$A_T f_1^2 \langle \nabla \lambda_1, \nabla \lambda_1 \rangle$$为例，假设顶点$$f_2, f_3$$处角的大小分别为$$\theta_2, \theta_3$$。从而如下图右侧所示：

$$A_T f_1^2 \langle \nabla \lambda_1, \nabla \lambda_1 \rangle = \frac{1}{2}\lVert \boldsymbol{e_{23}} \rVert h_1 f_1^2 \frac{1}{h_1^2} = f_1^2 \frac{\lVert \boldsymbol{e_{23}} \rVert}{2h_1} = f_1^2 \frac{e_{23}^{'} + e_{23}^{''}}{2h_1} = \frac{f_1^2}{2} (\text{cot} \theta_2 + \text{cot} \theta_3)$$

![2]({{ '/assets/images/laplace_beltrami_1.png' | relative_url }}){: width=100px style="float:center"}

从而对于每个三角形$$\triangle_{f_1f_2f_3}$$：

$$E_T = \frac{1}{2} A_T \sum_{i=1}^3 f_i^2 \langle \nabla \lambda_i, \nabla \lambda_i \rangle + A_T \sum_{1 \leq i < j \leq 3} f_i f_j \langle \nabla \lambda_i, \nabla \lambda_j \rangle = \frac{1}{4} \sum_{i=1}^3 f_i^2 (\sum_{j=1, j \neq i}^3 \text{cot} \theta_j) - \frac{1}{2} \sum_{i=1}^3 \text{cot} \theta_i (\prod_{j=1, j \neq i} f_j) = \frac{1}{4} \sum_{i=1}^3 \text{cot} \theta_i (\sum_{j=1, j \neq i}^3 f_j^2)  - \frac{1}{2} \sum_{i=1}^3 \text{cot} \theta_i (\prod_{j=1, j \neq i} f_j)$$


**第四步：组装三角网格所有三角形的Dirichlet能量**

按照之前计算的每个三角形的Dirichlet能量$$E_T$$，假设该三角网格的所有三角面的集合为$$\mathcal{F}$$，将所有三角形的Dirichlet能量加和，即得到这个三角网格的总Dirichlet能量

$$E(f) = \sum_{T \in \mathcal{F}} E_T = \frac{1}{4} \sum_{T \in \mathcal{F}} \sum_{i=1}^3 \text{cot} \theta_i (\sum_{j=1, j \neq i}^3 f_j^2) - \frac{1}{2} \sum_{T \in \mathcal{F}} \sum_{i=1}^3 \text{cot} \theta_i (\prod_{j=1, j \neq i} f_j)$$

注意，每个三角形的顶点都和其对面边一一对应，因此上述式子每个三角形里求和的每一项也可以用边的集合来表示，每条边$$(i,j)$$都恰好被两个三角形共享，假设其对面角分别为$$\alpha_{ij}, \beta_{ij}$$，因此上述式子第一项为：

$$
E(f)_1 = \frac{1}{4} \sum_{T \in \mathcal{F}} \sum_{i=1}^3 \text{cot} \theta_i (\sum_{j=1, j \neq i}^3 f_j^2) = \sum_{(i,j) \in \mathcal{E}} \frac{1}{4} (f_i^2 + f_j^2) (\text{cot} \alpha_{ij} + \text{cot} \beta_{ij})
$$

上述式子第二项为：

$$E(f)_2 =  - \frac{1}{2} \sum_{T \in \mathcal{F}} \sum_{i=1}^3 \text{cot} \theta_i (\prod_{j=1, j \neq i} f_j) = -\frac{1}{2} \sum_{(i,j) \in \mathcal{E}} (\text{cot} \alpha_{ij} + \text{cot} \beta_{ij}) f_i f_j$$

将这两项合并，得到最终的Dirichlet能量：

$$E(f) = \frac{1}{4} \sum_{(i,j) \in \mathcal{E}} (\text{cot} \alpha_{ij} + \text{cot} \beta_{ij}) (f_i - f_j)^2$$

将其写成矩阵形式，即$$E(f) = \frac{1}{2} \boldsymbol{f}^{\top}L \boldsymbol{f}$$，其中$$\boldsymbol{f}$$是所有顶点的函数值构成的向量，$$L$$就是之前定义的cotangent矩阵。

**第五步：计算离散Laplace-Beltrami算子作用在离散三角网格上定义的函数$$f$$的结果**

首先，我们介绍一个常用的曲面积分结果。

**Green第一恒等式**：对于曲面$$\mathcal{M}$$上定义的函数$$f, g$$，有如下结果成立：

$$\int_{\mathcal{M}} (g \Delta f + \langle \nabla g, \nabla f \rangle) dA = \int_{\partial \mathcal{M}} g \nabla f ds$$

如果$$\mathcal{M}$$是闭曲面，没有边界，那么$$\partial \mathcal{M}$$是空集，自然有：

$$\int_{\mathcal{M}} (g \Delta f + \langle \nabla g, \nabla f \rangle) dA = 0$$

即：

$$\int_{\mathcal{M}} g \Delta f  dA = -\int_{\mathcal{M}} \langle \nabla g, \nabla f \rangle dA$$

上述等式对于任意定义在流形$$\mathcal{M}$$上的函数$$f,g$$都成立，我们让$$f$$即为我们所关注的函数，而$$g$$为任意函数，推导出函数$$f$$应满足的关系，从而得出$$\Delta f$$的结果。因为在离散情况下，$$f$$只有在顶点上才有定义的值，我们来对上述恒等式两边进行离散化。

我们先来计算等式右边。注意，根据之前的步骤，我们已经利用线性插值，在每个三角形内部的点也定义了$$f$$的函数值，可以对函数$$g$$也做类似线性插值，从而$$f,g$$都是分段线性函数，在每个三角形$$\triangle_{f_1f_2f_3}$$内：

$$\nabla f = f_1 \nabla \lambda_1 + f_2 \nabla \lambda_2 + f_3 \nabla \lambda_3$$

$$\nabla g = g_1 \nabla \lambda_1 + g_2 \nabla \lambda_2 + g_3 \nabla \lambda_3$$

按照之前的结果，$$\nabla f$$和$$\nabla g$$三角形内部是常数值，因此$$\int_{T} \langle \nabla f, \nabla g \rangle = A_T \langle \nabla f, \nabla g \rangle$$，其中$$A_T$$是三角形$$T = \triangle_{f_1f_2f_3}$$的面积。

展开内积$$A_T \langle \nabla f, \nabla g \rangle = A_T \sum_{i=1}^3 \sum_{j=1}^3 f_i g_j \langle \nabla \lambda_i, \nabla \lambda_j \rangle$$。类似于之前的计算过程：

$$A_T \sum_{i=1}^3 \sum_{j=1}^3 f_i g_j \langle \nabla \lambda_i, \nabla \lambda_j \rangle = \frac{1}{2} \left[ \text{cot} \theta_1 (f_2 g_2 + f_3 g_3 - f_2 g_3 - f_3 g_2) + \text{cot} \theta_2 (f_1 g_1 + f_3 g_3 - f_1 g_3 - f_3 g_1) + \text{cot} \theta_3 (f_1 g_1 + f_2 g_2 - f_1 g_2 - f_2 g_1) \right]$$

对所有的三角面片求和，既可得：

$$\int_{\mathcal{M}} \langle \nabla f, \nabla g \rangle dA = \sum_{T \in \mathcal{F}} A_T \sum_{i=1}^3 \sum_{j=1}^3 f_i g_j \langle \nabla \lambda_i, \nabla \lambda_j \rangle = \sum_{(i,j) \in \mathcal{E}} (\text{cot} \theta_{\alpha_{ij}} + \text{cot} \theta_{\beta_{ij}}(f_ig_i + f_j g_j - f_ig_j - f_j g_i) = \sum_{(i,j) \in \mathcal{E}} (\text{cot} \theta_{\alpha_{ij}} + \text{cot} \theta_{\beta_{ij}})(f_i - f_j)(g_i - g_j)$$

写成矩阵的形式，$$\boldsymbol{f}, \boldsymbol{g}$$分别是函数$$f,g$$在所有顶点上的值构成的向量，从而：

$$\int_{\mathcal{M}} \langle \nabla f, \nabla g \rangle dA = \boldsymbol{f}^{\top}L \boldsymbol{g}$$

下面再来计算上述Green第一恒等式推出的结果的左侧$$\int_{\mathcal{M}} g \Delta f dA$$。

对于每个三角形面片，我们同样使用线性插值，利用每个顶点处的函数值，计算三角形内部点的函数值。即对于三角形$$T = \triangle_{f_1f_2f_3}$$以及其内部一点$$v$$：

$$\Delta f (v) = \lambda_1 \Delta f_1 + \lambda_2 \Delta f_2 + \lambda_3 \Delta f_3$$

$$g (v) = \lambda_1 g_1 + \lambda_2 g_2 + \lambda_3 g_3$$

需要注意$$\lambda_1, \lambda_2, \lambda_3$$要满足条件$$\lambda_1 + \lambda_2 + \lambda_3 = 1$$（即只有两个自由度）从而在每个三角形内部：

$$\int_{T} g \Delta f dA = 2A_T \int_0^1 \int_0^{1-\lambda_1} (\lambda_1 g_1 + \lambda_2 g_2 + (1-\lambda_1 - \lambda_2) g_3)(\lambda_1 \Delta f_1 + \lambda_2 \Delta f_2 + (1-\lambda_1 - \lambda_2) \Delta f_3) d\lambda_1 d\lambda_2 = \frac{A_T}{12} (\Delta f_1 \  \Delta f_2 \  \Delta f_3) \begin{pmatrix} 2 & 1 & 1 \\ 1 & 2 & 1 \\ 1 & 1 & 2 \\ \end{pmatrix} (g_1 \  g_2 \  g_3)^{\top}$$

> 注意，将积分变量从面积$$A$$变成$$\lambda_1, \lambda_2$$，所以引入了Jacobian，满足$$dA = 2A_T d\lambda_1 d\lambda_2$$。

从而$$\int_{\mathcal{M}} g \Delta f dA$$在每个三角形$$T$$上的积分值即为：

$$\int_{T} g \Delta f dA = \frac{A_T}{12} (\Delta f_1 \  \Delta f_2 \  \Delta f_3) \begin{pmatrix} 2 & 1 & 1 \\ 1 & 2 & 1 \\ 1 & 1 & 2 \\ \end{pmatrix} (g_1 \  g_2 \  g_3)^{\top} = (\Delta f_1 \  \Delta f_2 \  \Delta f_3) M_T (g_1 \  g_2 \  g_3)^{\top}$$

其中$$M_T$$是依赖于三角形$$T$$的质量矩阵，大小为$$3 \times 3$$，可以将其拓展为大小为$$\lvert \mathcal{V} \rvert$$的矩阵（补零），从而：

$$\int_{T} g \Delta f dA = \Delta \boldsymbol{ f}^{\top} M_T \boldsymbol{g}$$

其中$$\Delta \boldsymbol{ f}, \boldsymbol{g}$$是$$\Delta f, g$$在所有顶点上的函数值构成的向量。

对所有的三角形进行积分，即可得：

$$\int_{\mathcal{M}} g \Delta f dA = \sum_{T \in \mathcal{F}} \int_{T} g \Delta f dA = \sum_{T \in \mathcal{F}} \Delta \boldsymbol{ f}^{\top} M_T \boldsymbol{g} = \Delta \boldsymbol{ f}^{\top} (\sum_{T \in \mathcal{F}} M_T) \boldsymbol{g}$$

记$$M = \sum_{T \in \mathcal{F}} M_T$$，那么$$M$$就是整个三角网格的质量矩阵。

最后，比较上面Green第一恒等式左侧和右侧计算出来的结果，即可得：

$$\Delta \boldsymbol{ f}^{\top} M \boldsymbol{g} = \boldsymbol{f}^{\top} L \boldsymbol{g}$$

上面的结果对任意的函数$$g$$都成立，从而：

$$\Delta \boldsymbol{ f}^{\top} M = -\boldsymbol{f}^{\top} L$$

或者：

$$M \Delta \boldsymbol{ f} = -L\boldsymbol{f}$$

因为根据构造$$M,L$$都是对称矩阵。

### 4. 质量矩阵的选择

根据上面的证明过程，如果严格按照每个三角形内进行线性插值的方式，计算离散Laplace-Beltrami算子作用于函数$$f$$的结果，得到的是一个稠密的质量矩阵$$M$$，结合cotangent矩阵$$L$$，结果为：

$$M \Delta \boldsymbol{f} = -L\boldsymbol{f}$$

或者

$$\Delta \boldsymbol{f} = -M^{-1} L\boldsymbol{f}$$

从而计算矩阵$$-M^{-1}L$$的特征值和特征向量，就可以得到该离散Laplace-Beltrami算子的特征方程在各个顶点上的值。

但实际上，根据之前的结论，很多时候面积矩阵都是近似为一个对角矩阵，有如下几种近似方式：

* 一致质量矩阵（Consistent Mass Matrix）：这是精确的有限元质量矩阵，非对角，每个三角形贡献$$\frac{A_T}{12}\begin{pmatrix} 2 & 1 & 1 \\ 1 & 2 & 1 \\ 1 & 1 & 2 \\ \end{pmatrix}$$。它来自标准有限元理论，可以追溯到Strang & Fix 1973年的教科书 *An Analysis of the Finite Element Method*
* Barycentric面积（Row-sum lumping）：$$M_i = \frac{1}{3} \sum_{i \in T} A_T$$，即对于每个顶点$$i$$，其面积等于含有该顶点的所有三角形面积和的$$1/3$$。这等价于对一致质量矩阵做行求和（row lumping）。这是最简单的对角化方案，barycentric dual mesh 给出的面积恰好等于一致质量矩阵的行求和。文献中通常追溯到*Hinton, E. etal. 1976*，该文系统讨论了 lumping 技术。
* Voronoi面积：在每个非钝角三角形内，连接外接圆心到三条边的中点，将三角形分成三个区域，每个区域分配给最近的顶点。对于非钝角三角形，顶点的Voronoi区域面积可以用对边的cotangent权重表示。这个方案来自*Meyer et al. 2003*，他们介绍了使用Voronoi cells和混合有限元/有限体积方法来推导离散微分几何算子。
* Mixed Voronoi 面积：纯 Voronoi 面积在钝角三角形时有问题：外接圆心落在三角形外部，导致 Voronoi 区域不合理。*Meyer et al. 2003*提出了mixed area的修正方案，对非钝角三角形用Voronoi面积，对钝角三角形用其他策略（将钝角对面的中点代替外接圆心）。这是实际应用中最常用的方案，```libigl``` 中 ```MASSMATRIX_TYPE_VORONOI``` 实现的就是这个版本。

> Alec Jacobson在```https://alecjacobson.com/weblog/4666.html```提出了一种从混合有限元角度理解 mass lumping 的方式：用分段线性基函数离散位移，用定义在对偶网格（dual mesh）上的分段常数基函数离散速度。选择 Voronoi dual mesh 就自然得到 Voronoi 面积作为对角质量矩阵的元素，选择 barycentric dual mesh 就得到 barycentric 面积。这为不同的面积选择提供了统一的理论框架。


之所以对质量矩阵进行这样的近似，有如下几个原因：

* 求解效率：对角矩阵的逆就是逐元素取倒数。稠密矩阵的逆计算代价很大。但由于现在的算法提升和硬件提升，对于特征函数计算，这个优势其实不大。
* 逐顶点的局部公式：对角矩阵使得$$(\Delta f)_i = \frac{1}{M_i} (L \boldsymbol{f})_i$$，每个顶点的Laplacian-Beltrami算子计算后的函数的值只依赖于自身一环邻域的信息。这在很多应用中非常方便，比如曲率估计（$$\Delta \boldsymbol{p} = -2H \boldsymbol{n}$$，可以逐点计算）、Laplacian smoothing（逐顶点迭代更新）、以及任何需要局部计算$$\Delta f$$的场景。用稠密的面积矩阵，$$(\Delta f))_i$$依赖所有顶点的值，失去了局部性。
* 历史和惯例：cotangent Laplacian 在图形学中的广泛使用始于 Pinkall & Polthier (1993) 和 Meyer et al. (2003)，这些工作的重点是曲率估计和曲面流，需要的是逐顶点的局部公式，不是全局特征值问题。对角质量矩阵在这些场景下足够好，后续的大量工作沿用了这个惯例。

但对于特征函数计算，用完整的质量矩阵确实是更好的选择。广义特征值问题$$L\phi = \lambda M \phi$$中，稀疏的非对角矩阵$$M$$和系数的$$L$$在计算上几乎没有额外困难，而精度更高。


> 参考文献
> * $$\left[1 \right]$$ Strang, Gilbert, George J. Fix, and D. S. Griffin. "An analysis of the finite-element method." (1974): 62-62.
> * $$\left[2 \right]$$ Hinton, E., T. Rock, and O. C. Zienkiewicz. "A note on mass lumping and related processes in the finite element method." Earthquake Engineering & Structural Dynamics 4.3 (1976): 245-249.
> * $$\left[3 \right]$$ $$\left[3 \right]$$ Meyer, Mark, et al. "Discrete differential-geometry operators for triangulated 2-manifolds." Visualization and mathematics III. Berlin, Heidelberg: Springer Berlin Heidelberg, 2003. 35-57.

### 5. Laplace-Beltrami算子的应用

**1. mesh的per-vertex Global Point Signatures特征（GPS）**

先介绍一下等距变换的概念。

**等距变换（isometry transformation）**：根据$$\left[1 \right]$$等里的介绍，一个将surface $$S$$上的点映射到surface $$S^{'}$$上的点的映射，如果其满足映射后$$S^{'}$$上的任意弧长等于该弧上点在$$S$$上的原相构成的弧的长度，则称呼该映射是isometric或者length preserving的。注意$$S$$和$$S^{'}$$可以是同一个surface。

GPS embedding由论文$$\left[4\right]$$提出，是定义在mesh上的per-vertex feature，其希望能够构造一种per-vertex feature，不直接依赖于geodesic distance，从而可以对经过等距变换后的shape，保持其feature不变，且对一些局部的topological noise是鲁棒的（比如手指交叉引起的拓扑变换）。其是基于论文$$\left[3 \right]$$里的关于Laplace-Beltrami算子能够描述mesh的全局性质等结论，利用Laplace-Beltrami算子的eigenfunctions和eigenvalues构造出来的。

具体来说，对于mesh上一点$$p$$，其Global Point Signature，$$\text{GPS}(p)$$的定义为：

$$\text{GPS}(p) = ( \frac{1}{\sqrt{\lambda_1}} \phi_1(p), \frac{1}{\sqrt{\lambda_1}} \phi_1(p), \cdots )$$

其中$$\phi_i(p)$$表示eigenfunctions在点$$p$$上的取值。注意上述特征不包括$$\phi_0$$，因为其是一个常值向量。

$$\text{GPS}$$ embedding的定义实际上很类似于eigenfunctions，但其每一项除以了其对应的eigenvalue的根号，这是因为按照这样的定义方式，会和Green函数有关。具体来说：

$$G(p, q) = \text{GPS}(p) \cdot \text{GPS}(q)$$

其中$$G(\cdot, \cdot)$$是格林函数。



> 参考文献
> * $$\left[1 \right]$$ E. Kreyszig, Differential Geometry. Dover, 1991.
> * $$\left[2 \right]$$ ELAD A., KIMMEL R.: On bending invariant signatures for surfaces. IEEE Trans. Pattern Analysis and Machine Intelligence 25, 10 (2003), 1285–1295
> * $$\left[3\right]$$ Lévy, Bruno. "Laplace-beltrami eigenfunctions towards an algorithm that" understands" geometry." IEEE International Conference on Shape Modeling and Applications 2006 (SMI'06). IEEE, 2006.
> * $$\left[4\right]$$ Rustamov, Raif M. "Laplace-Beltrami eigenfunctions for deformation invariant shape representation." Symposium on geometry processing. Vol. 257. 2007.
