#  Efficient Semantic Video Segmentation with Per-frame Inference
In semantic segmentation, most existing real-time deep
models trained with each frame independently may produce inconsistent results for a video sequence. 
Advanced methods take into considerations
the correlations in the video sequence,
e.g., by propagating the results to the neighboring frames using optical flow, or extracting the frame representations with other frames, which may lead to inaccurate results or unbalanced latency. In this work, we process efficient semantic video segmentation in a per-frame fashion during the inference process.

Different from previous per-frame models, we explicitly consider the temporal consistency among frames as extra constraints during the training process and embed the temporal consistency into the segmentation network. Therefore, in the inference process, we can process each frame independently with no latency, and improve the temporal consistency with no extra computational cost and post-processing. We employ compact models for real-time execution. To narrow the performance gap between compact models and large models, new knowledge distillation methods are designed. Our results outperform previous keyframe based methods with a better trade-off between the accuracy and the inference speed on popular benchmarks, including the Cityscapes and Camvid.
The temporal consistency is also improved compared with corresponding baselines which are trained  with each frame independently.

This repository contains the demo evaluate code of our paper [Efficient Semantic Video Segmentation with Per-frame Inference](https://arxiv.org/pdf/2002.11433.pdf). 
Training code will coming soon.
## Sample results

Demo video for the PSPnet-18 on Cityscapes

ETC with mIoU 73.1, temporal consistncy 70.56 vs
baseline with mIoU 69.79, temporal consistncy 68.50:

![image](https://github.com/irfanICMLL/ETC-Real-time-Per-frame-Semantic-video-segmentation/blob/master/demo/val.gif)

![image](https://github.com/irfanICMLL/ETC-Real-time-Per-frame-Semantic-video-segmentation/blob/master/demo/demo_seq.gif)


## Performance on the Cityscape dataset
We employ the temporal loss the temporal knowledge distillation methods to adapte single frame image segmentation methods for semantic video segmentation methods.

| Model | mIoU |Temporal consitency|
| -- | -- |--|
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
2. python tool/tdemo.py.

Please change the ckpt_path in config to compare the results with baseline models
## Train script
Coming soon.


## Acknowledgments
The test code borrows from [semseg](https://github.com/hszhao/semseg).


















