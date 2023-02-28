# ProxyFormer: Proxy Alignment Assisted Point Cloud Completion with Missing Part Sensitive Transformer

Created by [Shanshan Li](https://github.com/MarkLiSS), [Pan Gao]()\*, [Xiaoyang Tan](), [Mingqiang Wei]()

[[arXiv]]() [[Dataset]](./DATASET.md) [[Models]](#pretrained-models) [[supp]](https://github.com/MarkLiSS/MyPapers/blob/main/Appendix_for_ProxyFormer.pdf)

This repository contains PyTorch implementation for __ProxyFormer: Proxy Alignment Assisted Point Cloud Completion with Missing Part Sensitive Transformer__ (Accepted by CVPR 2023).

Problems such as equipment defects or limited viewpoints will lead the captured point clouds to be incomplete. Therefore, recovering the complete point clouds from the partial ones plays an vital role in many practical tasks, and one of the keys lies in the prediction of the missing part. In this paper, we propose a novel point cloud completion approach namely ProxyFormer that divides point clouds into existing (input) and missing (to be predicted) parts and each part communicates information through its proxies. Specifically, we fuse information into point proxy via feature and position extractor, and generate features for missing point proxies from the features of existing point proxies. Then, in order to better perceive the position of missing points, we design a missing part sensitive transformer, which converts random normal distribution into reasonable position information, and uses proxy alignment  to refine the missing proxies. It makes the predicted point proxies more sensitive to the features and positions of the missing part, and thus make these proxies more suitable for subsequent coarse-to-fine processes. Experimental results show that our method outperforms state-of-the-art completion networks on several benchmark datasets and has the fastest inference speed.

<iframe src="https://github.com/I2-Multimedia-Lab/ProxyFormer/edit/main/README.md" style="width:800px; height:500px;" frameborder="0"></iframe>

## 🔥News
- **2023-02-28** ProxyFormer is accepted by CVPR 2023.
- **2022-11-05** ProxyFormer outperforms state-of-the-art completion networks on several benchmark datasets and has the fastest inference speed (PCN: CD 6.77 & DCD 0.577 etc.).

## Pretrained Models

We provide pretrained ProxyFormer models:
| dataset  | url| performance |
| --- | --- |  --- |
| ShapeNet-55 | [[[BaiDuYun](https://pan.baidu.com/s/1ZQf5XKgttZG0hZtDv9C_Vw)] (code:htzu) | CD = 0.93e-3| DCD = 0.588e-3|
| ShapeNet-34 | [[[BaiDuYun](https://pan.baidu.com/s/1sEo9F_UTrxBXyjD6pQ1Ydw)] (code:ocdl ) | CD = 1.42e-3| DCD = 0.583e-3|
| PCN |  [[[BaiDuYun](https://pan.baidu.com/s/1ISsmIkBYGNoJSnXKWAe_Ng)] (code:2ip6) | CD = 6.77e-3| DCD = 0.577e-3|

## Usage

### Requirements

- PyTorch >= 1.9.0
- python >= 3.7
- CUDA >= 11.0
- GCC >= 4.9 
- torchvision
- open3d
- plyfile
- tensorboardX

```
pip install -r requirements.txt
```

#### Building Pytorch Extensions for Chamfer Distance, PointNet++ and kNN

*NOTE:* PyTorch >= 1.9 and GCC >= 4.9 are required.

```
# Chamfer Distance
bash install.sh
```
The solution for a common bug in chamfer distance installation can be found in Issue [#6] of PoinTr (https://github.com/yuxumin/PoinTr/issues/6)


### Dataset

The details of ***ShapeNet-55/34*** datasets and other existing datasets can be found in [DATASET.md](./DATASET.md).

### Evaluation

To evaluate a pre-trained ProxyFormer model on the Three Dataset with single GPU, run:

```
bash ./scripts/test.sh <GPU_IDS>  \
    --ckpts <path> \
    --config <config> \
    --exp_name <name> \
    [--mode <easy/median/hard>]
```

####  Some examples:
Test the ProxyFormer pretrained model on the PCN benchmark:
```
bash ./scripts/test.sh 0 \
    --ckpts ./pretrained/ckpt-best.pth \
    --config ./cfgs/PCN_models/our_model.yaml \
    --exp_name example
```
Test the ProxyFormer pretrained model on ShapeNet55 benchmark (*easy* mode):
```
bash ./scripts/test.sh 0 \
    --ckpts ./pretrained/ckpt-best.pth \
    --config ./cfgs/ShapeNet55_models/our_model.yaml \
    --mode easy \
    --exp_name example
```

### Training

To train ProxyFormer from scratch, run:

```
# Use DistributedDataParallel (DDP)
bash ./scripts/dist_train.sh <NUM_GPU> <port> \
    --config <config> \
    --exp_name <name> \
    [--resume] \
    [--start_ckpts <path>] \
    [--val_freq <int>]
# or just use DataParallel (DP)
bash ./scripts/train.sh <GPUIDS> \
    --config <config> \
    --exp_name <name> \
    [--resume] \
    [--start_ckpts <path>] \
    [--val_freq <int>]
```
####  Some examples:
Train a ProxyFormer model with a single GPU:
```
bash ./scripts/train.sh 0 \
    --config ./cfgs/PCN_models/our_model.yaml \
    --exp_name example
```

### Completion Results on PCN and ShapeNet-55

![results](figs/PCN_Dataset_Result_Picture.pdf)
![results](figs/ShapeNet55_Results.pdf)

## Acknowledgements

Our code is inspired by [PoinTr](https://github.com/yuxumin/PoinTr) and [SeedFormer](https://github.com/hrzhou2/seedformer). Thanks for their excellent works.

## Citation
If you find our work useful in your research, please consider citing: 
```

```
