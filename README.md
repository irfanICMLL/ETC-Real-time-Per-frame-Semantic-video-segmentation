#  Structured Knowledge Distillation for Dense Prediction

This repository contains the demo evaluate code of our paper [Efficient Semantic Video Segmentation withPer-frame Inference](https://arxiv.org/pdf/).
Training code will coming soon.
## Sample results

Demo video for the PSPnet-18 on Cityscapes

ETC with mIoU 73.1, temporal consistncy 70.56 vs
baseline with mIoU 69.79, temporal consistncy 68.50:

![image](https://github.com/irfanICMLL/ETC-Real-time-Per-frame-Semantic-video-segmentation/blob/master/demo/val.mp4)

![image](https://github.com/irfanICMLL/ETC-Real-time-Per-frame-Semantic-video-segmentation/blob/master/demo/demo_seq.mp4)


## Performance on the Cityscape dataset
We apply the distillation method to training the [PSPNet](https://arxiv.org/abs/1612.01105). We used the dataset splits (train/val/test) provided [here](https://github.com/speedinghzl/pytorch-segmentation-toolbox). We trained the models at a resolution of 512x512.
Pi: Pixel-wise distillation PA: Pair-wise distillation HO: holistic distillation

| Model | mIoU |Temporal consitency
| -- | -- |
| baseline | 69.79 |68.50|
| +temporal loss | 71.72 |69.99 |
| +temporal loss + distillation | 73.06 |70.56 |

Note: Other chcekpoints can be obtained by email: yifan.liu04@adelaide.edu.au if needed.


## Requirement
python3.5 

pytorch >1.0.0

We recommend to use [Anaconda](https://conda.io/docs/user-guide/install/linux.html).

We have tested our code on Ubuntu 16.04.


## Quick start to test the model
1. download the [Cityscape dataset](https://www.cityscapes-dataset.com/)
2. python tool/test.py [you should change the data-dir to your own].  of VNL and FCOS using our checkpoints.

Please change the ckpt_path in config to compare the results with baseline models
## Train script
Coming soon.


## Acknowledgments
This code borrows heavily from [SPADE](https://github.com/hszhao/semseg).


















