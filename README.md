HyperTaFOR: Task-adaptive Few-shot Open-set Recognition With Spatial-spectral Selective Transformer for Hyperspectral Imagery, TIP, 2025.
==
[Bobo Xi](https://b-xi.github.io/), [Wenjie Zhang](https://github.com/WenjieZhang-cyber), [Jiaojiao Li](https://scholar.google.com/citations?user=Ccu3-acAAAAJ&hl=zh-CN&oi=sra), [Rui Song](https://scholar.google.com/citations?user=_SKooBYAAAAJ&hl=zh-CN), and [Yunsong Li](https://dblp.uni-trier.de/pid/87/5840.html).
***

Code for the paper: [HyperTaFOR: Task-adaptive Few-shot Open-set Recognition With Spatial-spectral Selective Transformer for Hyperspectral Imagery](http://doi.org/10.1109/TIP.2025.3555069).

<div align=center><img src="/backbone.png" width="90%" height="90%"></div>
Fig. 1: The architecture of the proposed HyperTaFOR for HSI FSOSR. 

## Abstract
Open set recognition (OSR) aims to accurately classify known categories while effectively rejecting unknown negative samples. Existing methods for OSR in hyperspectral images (HSI) can be generally divided into two categories: reconstruction-based and distance-based methods. Reconstruction-based approaches focus on analyzing reconstruction errors during inference, whereas distance-based methods determine the rejection of unknown samples by measuring their distance to each prototype. However, these techniques often require a substantial amount of training data, which can be both time-consuming and expensive to gather, and they require manual threshold setting, which can be difficult for different tasks. Furthermore, effectively utilizing spectral-spatial information in HSI remains a significant challenge, particularly in open-set scenarios. 
To tackle these challenges, we introduce a few-shot OSR framework for HSI named HyperTaFOR, which incorporates a novel spatial-spectral selective transformer (S3Former). This framework employs a meta-learning strategy to implement a negative prototype generation module (NPGM) that generates task-adaptive rejection scores, allowing flexible categorization of samples into various known classes and anomalies for each task. Additionally, the S3Former is designed to extract spectral-spatial features, optimizing the use of central pixel information while reducing the impact of irrelevant spatial data. Comprehensive experiments conducted on three benchmark hyperspectral datasets show that our proposed method delivers competitive classification and detection performance in open-set environments when compared to state-of-the-art methods. 
The code will be available online at https://github.com/B-Xi/TIP_2025_HyperTaFOR.

## Training and Test Process
1. Please prepare the training and test data as operated in the paper. And the websites to access the datasets are also provided. The used OCBS band selection method is referred to https://github.com/tanmlh
2. Run "trainMetaDataProcess.py" to generate the meta-training data
3. Run the 'main.py' to reproduce the HyperTaFOR results on PC data set.

## References
--
If you find this code helpful, please kindly cite:

[1] Xi, B., Zhang, W., Li, J., Song, R., & Li, Y., "HyperTaFOR: Task-adaptive Few-shot Open-set Recognition With Spatial-spectral Selective Transformer for Hyperspectral Imagery" in IEEE Transactions on Image Processing, vol. 34, pp. 854-868, 2025, doi: 10.1109/TIP.2025.3555069.

Citation Details
--
BibTeX entry:
```
@ARTICLE{TIP_2025_TEFSL,
  author={Xi, Bobo and Zhang, Wenjie and Li, Jiaojiao and Song, Rui and Li, Yunsong},
  journal={IEEE Transactions on Image Processing}, 
  title={HyperTaFOR: Task-adaptive Few-shot Open-set Recognition With Spatial-spectral Selective Transformer for Hyperspectral Imagery}, 
  year={2025},
  volume={34},
  number={},
  pages={854-868},
  doi={10.1109/TIP.2025.3555069}}
```
 
Licensing
--
Copyright (C) 2025 Bobo Xi, Wenjie Zhang

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, version 3 of the License.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program.
