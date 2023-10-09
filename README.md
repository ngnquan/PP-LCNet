# PyTorch implementation of PP-LCNet

Reproduction of PP-LCNet architecture as described in [PP-LCNet: A Lightweight CPU Convolutional Neural Network](https://arxiv.org/pdf/2109.15099.pdf) by C. Cui. T. Gao, S. Wei *et al* (2021) with the [PyTorch](pytorch.org) framework. 

The official design is implemented with [Paddle](https://github.com/PaddlePaddle/Paddle) framework, the detail [here](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.3/ppcls/arch/backbone/legendary_models/pp_lcnet.py)

## TODO
- [ ] PPLCNetv2 from [this](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.5/docs/en/models/PP-LCNetV2_en.md)
- [ ] PPLCNetv3 from [this](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.7/ppocr/modeling/backbones/rec_lcnetv3.py)
- [ ] Convert weights from Paddle to Pytorch

## Models

| Architecture      | #Parameters | FLOPs | Top-1 Acc. (%) |
| ----------------- | :------------: | :------: | -------------------------- |
| PPLCNet_x0_25    | 1,522,960 | 18M |  |
| PPLCNet_x0_35    | 1,646,888 | 29M |  |
| PPLCNet_x0_5     | 1,881,864 | 47M |  |
| PPLCNet_x0_75    | 2,359,792 | 99M |  |
| PPLCNet_x1_0     | 2,955,816 | 161M |  |
| PPLCNet_x1_5     | 4,504,136 | 342M |  |
| PPLCNet_x2_0     | 6,526,824 | 590M |  |
| PPLCNet_x2_5     | 9,023,880 | 906M |  |

Stay tuned for ImageNet pre-trained weights.

## Acknowledgement

The implementation is heavily borrowed from [HBONet](https://github.com/d-li14/HBONet) or [MobileNetV2](https://github.com/d-li14/mobilenetv2.pytorch), please kindly consider citing the following

```
@InProceedings{Li_2019_ICCV,
author = {Li, Duo and Zhou, Aojun and Yao, Anbang},
title = {HBONet: Harmonious Bottleneck on Two Orthogonal Dimensions},
booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
month = {Oct},
year = {2019}
}
```
```
@InProceedings{Sandler_2018_CVPR,
author = {Sandler, Mark and Howard, Andrew and Zhu, Menglong and Zhmoginov, Andrey and Chen, Liang-Chieh},
title = {MobileNetV2: Inverted Residuals and Linear Bottlenecks},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2018}
}
```
