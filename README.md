# 🧠 Gradient Origin Networks (GON) for Anomaly Detection

This repository implements **Gradient Origin Networks (GON)** for image anomaly detection tasks, comparing their performance to traditional Autoencoders. GONs provide a lightweight, encoder-free generative model ideal for efficient and effective anomaly detection.

---

## 📌 Problem Statement

Traditional generative models like VAEs and GANs involve separate encoder-decoder architectures, increasing complexity and training time. GONs streamline this process by:

- Eliminating explicit encoders.
- Using gradients for encoding.
- Enabling single-step image reconstruction.

---

## 🚀 Key Features

- ⚡ **Fast and Lightweight**: Encoder-free generative model.
- 📉 **Efficient Anomaly Detection**: Uses dynamic thresholds for reconstruction loss.
- 🧪 **Benchmarking**: Compared against traditional Autoencoder on **MVTec AD** and **Corrupted MNIST**.

---

## 🧪 Datasets

- **[MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad/)**: Industrial anomaly detection dataset.
- **Corrupted MNIST**: Modified version of MNIST with added noise.

---

## 🧰 Methodology

- **Latent Vector ($z_0$)**: Initialized to zero.
- **Coordinates ($c$)**: Pixel positions or similar features.
- **Network ($F$)**: A single model for encoding and decoding.
- **Output ($\hat{x}$)**: Reconstructed image.
- **Anomaly Score**: Computed as the mean squared error (MSE) between the reconstructed and original images.
- **Thresholding**: $threshold = \text{mean}_{loss} + 2 \cdot \text{std}_{loss}$

---

## 📊 Results

- GON achieves competitive reconstruction and anomaly detection accuracy.
- Dynamic thresholding identifies outliers based on reconstruction loss.
- Compared against a baseline Convolutional Autoencoder.

---

