Code for "[Random resistive memory-based extreme point learning machine for unified visual processing.]([https://arxiv.org/abs/2312.09262](https://www.nature.com/articles/s41467-025-56079-3))"

## Abstract
Visual sensors, including 3D LiDAR, neuromorphic DVS, and conventional frame cameras, are increasingly integrated into edge-side intelligent machines. However, their data are heterogeneous, causing complexity in system development. Moreover, conventional digital hardware is limited by von Neumann bottleneck and the physical limit of transistor scaling. The computational demands of training ever-growing models further exacerbate these challenges. We propose a hardware-software co-designed random resistive memory-based deep extreme point learning machine. Data-wise, the multi-sensory data are unified as point set and processed universally. Software-wise, most weights are exempted from training. Hardware-wise, nanoscale resistive memory enables collocation of memory and processing, and leverages the inherent programming stochasticity for generating random weights. The co-design system is validated on 3D segmentation (ShapeNet), event recognition (DVS128 Gesture), and image classification (Fashion-MNIST) tasks, achieving accuracy comparable to conventional systems while delivering 6.78×/21.04×/15.79× energy efficiency improvements and reducing training costs by 70.12%/89.46%/85.61% training cost reduction. edge AI across various data modalities and tasks.

## Requirements
The codes are tested on Ubuntu 20.04, CUDA 12.0 with the following packages:

```bash
torch==2.0.0
numpy
tqdm
```

## Installation
You can install the required dependencies with the following code.

## Dataset

the Fashion-MNIST dataset can be downloaded automatically by torchvision. The preprocessed ShapeNet dataset can be downloaded from [here](https://drive.google.com/file/d/1ngKbWeDG6A9PopfYj_u8GZ8ONFjPZ3--/view?usp=sharing). The preprocessed DVS128 Gesture dataset can be downloaded from [here](https://drive.google.com/file/d/1bWRCtmnEBDc9-uOQRyswI2VhSdYqmw0g/view?usp=sharing).

datasets should be placed in the `data` folder.

## Run

For the 3D segmentation task, run the following code in the terminal:

```bash
bash run_seg.sh
```

For the DVS128 Gesture event classification task, run the following code in the terminal:

```bash
bash run_dvs.sh
```

FOr the Fashion-MNIST image classification task, run the following code in the terminal:

```bash
bash run_image.sh
```

## Acknowledgement

The codes for buiding point based model are build on the following repositories: [pointnet2](https://github.com/charlesq34/pointnet2), [Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch).

## Citation

Welcome to cite our paper if it help!

```
@article{wang2023random,
  title={Random resistive memory-based deep extreme point learning machine for unified visual processing},
  author={Wang, Shaocong and Gao, Yizhao and Li, Yi and Zhang, Woyu and Yu, Yifei and Wang, Bo and Lin, Ning and Chen, Hegan and Zhang, Yue and Jiang, Yang and others},
  journal={arXiv preprint arXiv:2312.09262},
  year={2023}
}
```
