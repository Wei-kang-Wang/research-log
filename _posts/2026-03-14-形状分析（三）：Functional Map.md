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

Functional map在2012年由Ovsjanikov Maks等人于$$\left[1 \right]$$中提出，不同于之前的点与点之间的对应关系构成的map，而是用一种新型的方式来表示两个shapes之间的correspondence关系。

### 1. Functional map



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
