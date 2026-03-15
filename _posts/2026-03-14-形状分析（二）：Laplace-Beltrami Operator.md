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

首先，我们介绍一些基本概念。

**Banach空间**：给定一个流形 $$S$$，$$X$$表示定义在$$S$$上的实值函数空间，且定义该函数空间的范数$$\lVert \  \cdot \  \rVert$$。如果$$X$$里的任意柯西列都收敛到$$X$$里的某个函数，其中$$X$$里的柯西列表示一系列函数$$f_1, f_2, \cdots$$，且$$\lim_{n,m \rightarrow \infty} \lVert f_n - f_m \rVert = 0$$，那么该函数空间$$X$$是完备的。完备的向量空间称为Banach空间。

**Hilbert空间**：如果一个Banach空间里的范数是由内积定义的，即$$\lVert f \rVert = \sqrt{\langle f, f \rangle}$$，那这个空间就是Hilbert空间。

> 常见的内积可以定义为$$\langle f,g \rangle = \int_{S} f(x)g(x) dx$$，其定义的范数叫做$$L_2$$范数。

**算子（operator）**：在泛函分析里，算子被定义为函数的函数，比如将函数空间$$X$$映射到函数空间$$X$$，$$L: X \rightarrow X$$将$$f \in X$$映射到$$Lf \in X$$。一个算子$$L$$被称作线性，如果其对于所有的$$f \in X, \lambda \in \mathbb{R}$$，均满足$$L(\lambda f) = \lambda f$$。

**算子的特征函数（eigenfunctions）**：一个算子$$L$$的特征函数eigenfunctions，是满足$$Lf = \lambda f$$的函数$$f$$，其中和$$f$$对应的$$\lambda$$称为该特征函数$$f$$对应的特征值。也就是说，将算子$$L$$用在其特征函数上，等价于简单的缩放该函数。

**Hermitian算子**：一个算子如果满足对函数$$f,g \in X$$，均满足$$\langle Lf,g \rangle = \langle f, Lg \rangle$$，则称其为Hermitian算子，或者说该算子满足Hermitian symmetry。Hermitian算子的一个重要特性是其不同eigenvalues对应的eigenfunctions都相互垂直（$$\lambda \langle f,g \rangle = \langle \lambda f, g \rangle = \langle Lf,g \rangle = \langle f, Lg \rangle = \langle f, \mu g \rangle = \mu \langle f,g \rangle$$，且$$\lambda \neq \mu$$，说明$$\langle f,g \rangle=0$$）。

**Laplace-Beltrami算子**：$$\mathcal{M}$$是一个紧且联通（compact以及connected）的2维流形，$$L^2(\mathcal{M}) = \lbrace f: \mathcal{M} \rightarrow \mathbb{R} \vert \langle f, f \rangle_{\mathcal{M}} = \int_{\mathcal{M}} f^2(x) dx < \infty \rbrace$$表示的是在$$\mathcal{M}$$上定义的所有平方可积函数构成的函数空间。$$\mathcal{M}$$上的Laplace-Beltrami算子$$\Delta_{\mathcal{M}}: L^2(\mathcal{M}) \rightarrow L^2(\mathcal{M})$$定义为：$$\Delta_{\mathcal{M}} f = - \text{div}_{\mathcal{M}} (\nabla_{\mathcal{M}} f)。

> Laplace-Beltrami算子是Hermitian算子。

> Laplace-Beltrami算子的一个重要性质是，满足$$\Delta_{\mathcal{M}} \phi_i(x) = \lambda_i \phi_i(x), \forall x \in \mathcal{M}$$的所有eigenfunctions构成的集合$$\lbrace \phi_1, \phi_2, \cdots \rbrace$$构成函数空间$$L^2(\mathcal{M})$$的一组正交基，即任意$$f \in L^2(\mathcal{M})$$可以被表示为$$f(x) = \sum_{i=1}^{\infty} \langle f, \phi_i \rangle_{\mathcal{M}} \phi_i(x), \forall x \in \mathcal{M}$$。
