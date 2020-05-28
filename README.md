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
The flownet need to be compiled following [FlowNetV2](https://github.com/NVIDIA/flownet2-pytorch)
You can first clone the FlowNetV2, and compile it. 
Then copy the folder of flownet2-pytorch/networks/resample2d_package,correlation_package,channelnorm_package to OURS/flownet/
Download the weight of the [flownet](https://drive.google.com/file/d/1hF8vS6YeHkx3j2pfCeQqqZGwA_PJq_Da/view?usp=sharing), and place it in OURS/pretrained_model/

## Quick start to test the model
1. download the [Cityscape dataset](https://www.cityscapes-dataset.com/)
2. python tool/tdemo.py.
## Evaluation the Temporal Consistency
To evaluate the temporal consistency, you need to install the flownet first.
1. You need to download the video data of the [cityscapes](https://www.cityscapes-dataset.com/downloads/): leftImg8bit_sequence_trainvaltest.zip 
2. The download data should be placed in data/cityscapes/leftImg8bit/
3. Generate the results for the sampled frames, which need to be evaluated: python tool/gen_video.py
4. Evaluate the temporal consistency based on the warpping mIoU: python tool/eval_tc.py
Note that the first time you evaluate the TC, the code will save the flow automatically.
In our paper, we random sample ~20% of the validation set for testing the TC for all models for efficiency (lists are in 'data/list/cityscapes/val_sam').
If you want to evaluate with all the validation video clips, you can relpace the 'data/list/cityscapes/val_video_img_sam.lst' with 'data/list/cityscapes/val_video_img.lst', and replace the 'data/list/cityscapes/val_sam' with 'data/list/cityscapes/val'. The trendency of the TC are similar.

Please change the ckpt_path in config to compare the results with baseline models
## Train script
Coming soon.


## Acknowledgments
The test code borrows from [semseg](https://github.com/hszhao/semseg).


















