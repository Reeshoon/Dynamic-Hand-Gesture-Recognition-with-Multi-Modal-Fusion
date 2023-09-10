## Hand Gesture Recognition with Multi-Modal Fusion

Hand gestures represent spatiotemporal body language conveyed by various aspects of the hand, such as the palm, shape of the hand, and finger position, with the aim of conveying a particular message to the recipient. Computer Vision has different modalities of input, such as depth images, skeletal joint points, or RGB images. Raw-depth images are found to have poor contrast in the region of interest, which makes it difficult for the model to learn important information. Recently, in deep learning-based dynamic hand gesture recognition, researchers have attempted to combine different input modalities to improve recognition accuracy. 

### Our Approach

In this research project, we explore the use of depth-quantized image features and point clouds to recognize dynamic hand gestures (DHG). We investigate the impact of fusing depth-quantized features in Convolutional Neural Networks (CNN) and Recurrent Neural Networks (RNN) with point clouds in LSTM-based multi-modal fusion networks.

### Key Contributions

- Integration of depth-quantized image features and point clouds for DHG recognition.
- Comparative analysis of fusion techniques involving CNNs, RNNs, and LSTM-based multi-modal fusion networks.

This repository contains code and resources related to our research in dynamic hand gesture recognition using multi-modal fusion. 

## Score-Level Fusion Model

The `PointDepthScoreFusion` class, implemented in [score_fusion_model.py](./models/score_fusion_model.py), represents the score-level fusion model.

## Feature-Level Fusion Model

The `PointDepthFeatureFusion` class, implemented in [feature_fusion_model.py](./models/feature_fusion_model.py), represents the feature-level fusion model.

### Data Loader

We provide a data loader script, [dataloader.py](./dataloader.py), which loads the depth sequences and point cloud data for training and testing.

## Getting Started

To start training the recognition model, use the following command:

```bash
%run main.py
```

## Citations
- **Paper Title**: [A deep-learning-based multimodal depth-aware dynamic hand gesture recognition system](https://arxiv.org/pdf/2107.02543.pdf)
  - **Authors**: Hasan Mahmud, Mashrur M Morshed, Md Hasan
  - **Journal**: The Visual Computer
  - **Year**: 2023
    
- **Paper Title**: [An Efficient PointLSTM for Point Clouds-Based Gesture Recognition](https://openaccess.thecvf.com/content_CVPR_2020/papers/Min_An_Efficient_PointLSTM_for_Point_Clouds_Based_Gesture_Recognition_CVPR_2020_paper.pdf)
  - **Authors**: Yadan Min, Fuhai Liu, Yu Chen, Sen-Ching Samson Cheung
  - **Conference**: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)
  - **Year**: 2020




