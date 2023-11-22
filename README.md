## M2PLNet: Multi-head multi-scale pixel localization network for crowd counting with highly dense and small-scale samples

This is the official implementation for paper: Multi-head multi-scale pixel localization network for crowd counting with highly dense and small-scale samples

![fig_1-网络结构3](https://mhyimg.oss-cn-hangzhou.aliyuncs.com/img/202311221546561.png)

M2PLNet consists of three parts: backbone, neck, and head. We design a novel neck, called centralized asymptotic feature pyramid (CAFP) module, which helps to capture feature information at different scales; thus improving the accuracy and robustness of counting. In the head part, we introduce a multi-head mechanism that contains four-pair heads, each of which handles feature maps with a special scale.  

## Comparison with state-of-the-art methods

M2PLNet achieves state-of-the-art performance on several challenging datasets with small sample densities.

| Methods | Venue     | SHTech Part A MAE | SHTech Part A MSE |
| ------- | --------- | ----------------- | ----------------- |
| MCNN    | CVPR'16   | 110.2             | -                 |
| CSRNet  | CVPR'18   | 68.2              | 115               |
| CAN     | CVPR'19   | 62.8              | 101.8             |
| SUA     | ICCV'21   | 68.5              | 121.9             |
| P2PNet  | ICCV'21   | 54.21             | **84.96**         |
| MAN     | CVPR'22   | 56.8              | 90.3              |
| GauNet  | CVPR'22   | 54.8              | 89.1              |
| CLTR    | ECCV'23   | 56.9              | 95.2              |
| DDC     | CVPR'23   | 52.87             | 85.62             |
| CHS-Net | ICASSP'23 | 59.2              | 97.8              |
| Ours    | -         | **50.86**         | 89.86             |

| Methods | Venue     | UCF_CC_50 MAE | UCF_CC_50 MSE |
| ------- | --------- | ------------- | ------------- |
| MCNN    | CVPR'16   | 377.6         | -             |
| CSRNet  | CVPR'18   | 266.1         | -             |
| CAN     | CVPR'19   | 212.2         | 243.7         |
| SUA     | ICCV'21   | -             | -             |
| P2PNet  | ICCV'21   | 172.72        | 256.18        |
| MAN     | CVPR'22   | -             | -             |
| GauNet  | CVPR'22   | 186.3         | 256.5         |
| CLTR    | ECCV'23   | -             | -             |
| DDC     | CVPR'23   | 157.12        | 220.59        |
| CHS-Net | ICASSP'23 | -             | -             |
| Ours    | -         | **142.56**    | **215.87**    |

| Methods   | P(%)   | R(%)   | F(%)   |
| --------- | ------ | ------ | ------ |
| LCFCN     | 43.30% | 26.00% | 32.50% |
| Method in | 34.90% | 20.70% | 25.90% |
| LSC-CNN   | 33.40% | 31.90% | 32.60% |
| TopoCount | 41.70% | 40.60% | 41.10% |
| CLTR      | 43.60% | 42.70% | 43.20% |
| Ours      | 44.20% | 43.10% | 43.60% |

## Ablation experiment

Ablation experiment on the head part.

| Prediction level | SHTech Part A MAE | SHTech Part A MSE |
| ---------------- | ----------------- | ----------------- |
| P2               | 56.42             | 93.19             |
| P3               | 51.84             | 92.48             |
| P4               | 55.13             | 90.14             |
| P5               | 53.17             | 90.15             |
| P2,P3,P4,P5      | 50.86             | 89.86             |

Ablation experiment on the neck part.

| Neck       | SHTech Part A MAE | SHTech Part A MSE |
| ---------- | ----------------- | ----------------- |
| FPN        | 72.19             | 112.18            |
| EVC+FPN    | 54.67             | 96.99             |
| AFPN       | 52.31             | 97.35             |
| CAFP(Ours) | 50.86             | 89.86             |

Ablation experiment on the loss funcation.

come soon.

## Run

1. ``pip install -r requirements.txt``
2. Configuring parameters in train.py
3. Run train.py