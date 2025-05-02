# Gradient Origin Networks for Image Anomaly Detection

## Overview

Gradient Origin Networks (GON) is a novel type of implicit generative model that captures datasets without an explicit encoder and can generate samples in a single step. This project explores the application of GON to the task of anomaly detection in image data.

### Key Features

- **Streamlined Architecture**: Eliminates the need for separate encoders by using gradients to perform encoding, reducing overall model complexity
- **Efficient Learning**: Achieves faster convergence and lower reconstruction errors with fewer parameters
- **Applied to Anomaly Detection**: Uses reconstruction error as an anomaly signal to identify abnormal samples

## Framework Overview

### Gradient Origin Network Components

- **Latent Vector (z0)**: Initialized at zero
- **Coordinates (c)**: Input features (e.g., pixel positions)
- **Network (F)**: Single network for both encoding and decoding
- **Output (x^)**: Reconstructed data (e.g., image)

### Process

1. **Latent Initialization**: Start with z0 at zero
2. **Gradient Encoding**: Compute the gradient to encode data
3. **Reconstruction**: Network F combines gradient and coordinates c to generate the output x^

## Methodology

Unlike traditional VAEs which use separate encoder and decoder networks, GON uses a single network structure. The comparison is illustrated in the repository's visualization code:
- VAE: Uses encoder (E) and decoder (D) networks
- GON: Uses a single network (F) with gradient-based encoding

### Anomaly Detection Method

- The model is trained on normal (non-anomalous) data
- Anomalies are detected based on a dynamic threshold: `threshold = mean_loss + 2 * std_loss`
- Images with reconstruction losses above the threshold are classified as anomalies
- During evaluation, the reconstruction error is computed as the mean squared error between the generated image and the original image

## Experiments

### Datasets Used

1. **Corrupted MNIST**
   - Train: 60,000 non-defective images
   - Test: 30,000 images (2K each from 14 types of corruption and 2K from original MNIST test set)

2. **MVTec AD**
   - Train: 3,629 non-defective images
   - Test: 1,725 images (467 good, 1,258 defective)

### Models Compared

- Gradient Origin Network (GON)
- Convolutional Autoencoder (CAE)

## Results

The repository contains code to reproduce the following results:

1. **Reconstruction Performance**: GON provides high-quality image reconstruction with visual comparisons shown for MNIST and other datasets
2. **Loss Convergence**: Loss curves show rapid convergence for GON on both MNIST (around 30 epochs) and MVTec AD datasets (around 3000 iterations)
3. **Anomaly Detection Metrics**: 
   - **Confusion Matrices**: For MNIST, GON correctly classifies 1,913 normal images and 21,567 anomalies
   - **For MVTec AD**: GON identifies 313 normal images and 536 anomalies
   - **Threshold-Based Classification**: Uses `threshold = mean_loss + 2 * std_loss` to determine anomalies



## Getting Started

### Prerequisites

```
pytorch
numpy
matplotlib
scikit-learn
```

### Installation

```bash
git clone https://github.com/Mathio11/GON_for_Image_Anomaly_Detection.git
cd GON_for_Image_Anomaly_Detection
```



## References

- Bond-Taylor, S., & Willcocks, C. G. (2020). Gradient Origin Networks. [arXiv:2007.02798](https://arxiv.org/abs/2007.02798)


## Acknowledgments

- The authors of the original GON paper
- The authors of the MVTec AD and MNIST datasets
