# SwinLSTM: A new recurrent cell for spatiotemporal modeling

This repository contains the official PyTorch implementation of the following paper:

SwinLSTM: Improving Spatiotemporal Prediction Accuracy using Swin Transformer and LSTM  **(ICCV 2023)**

Paper:http://arxiv.org/abs/2308.09891v2




## Introduction
![architecture](/architecture.png)
Integrating CNNs and RNNs to capture spatiotemporal dependencies is a prevalent strategy for spatiotemporal prediction tasks. However, the property of CNNs to learn local spatial information decreases their efficiency in capturing spatiotemporal dependencies, thereby limiting their prediction accuracy. In this paper, we propose a new recurrent cell, SwinLSTM, which integrates Swin Transformer blocks and the simplified LSTM, an extension that replaces the convolutional structure in ConvLSTM with the self-attention mechanism. Furthermore, we construct a network with SwinLSTM cell as the core for spatiotemporal prediction. Without using unique tricks, SwinLSTM outperforms state-of-the-art methods on  Moving MNIST, Human3.6m, TaxiBJ, and KTH datasets. In particular, it exhibits a significant improvement in prediction accuracy compared to ConvLSTM. Our competitive experimental results demonstrate that learning global spatial dependencies is more advantageous for models to capture spatiotemporal dependencies. We hope that SwinLSTM can serve as a solid baseline to promote the advancement of spatiotemporal prediction accuracy.

## Overview
- `Pretrained/` contains pretrained weights on MovingMNIST.
- `data/` contains the MNIST dataset and the MovingMNIST test set download link.
- `SwinLSTM_B.py` contains the model with a single SwinLSTM cell.
- `SwinLSTM_D.py` contains the model with a multiple SwinLSTM cell.
- `dataset.py` contains training and validation dataloaders.
- `functions.py` contains train and test functions.
- `train.py` is the core file for training pipeline.
- `test.py` is a file for a quick test.

## Requirements
- python >= 3.8
- torch == 1.11.0
- torchvision == 0.12.0
- numpy
- matplotlib
- skimage == 0.19.2
- timm == 0.4.12
- einops == 0.4.1

## Citation
If you find this work useful in your research, please cite the paper:
```
@inproceedings{tang2023swinlstm,
  title={SwinLSTM: Improving Spatiotemporal Prediction Accuracy using Swin Transformer and LSTM},
  author={Tang, Song and Li, Chuang and Zhang, Pu and Tang, RongNian},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={13470--13479},
  year={2023}
}

```

## Acknowledgment
These codes are based on [Swin Transformer](https://github.com/microsoft/Swin-Transformer). We extend our sincere appreciation for their valuable contributions.

