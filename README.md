# unet-liver-segmentation

## U-Net Biomedical Image Segmentation

This repository contains the code to perform liver segmentation using [U-Net](https://arxiv.org/abs/1505.04597) and tensorflow.

![U-Net model](data/u-net.png) <br/>

## Data

The data are 2D medical image obtained from an abdominal CT-scan. Image processing from dicom to numpy, including HU and pixels normalization could be found at [this](https://github.com/vincentzossou/abdo-ctscan-processing) repository. Data can be download [here](https://drive.google.com/drive/folders/1PNAv7LGjPhw3cStAgAzbTUzuYodCcEmb?usp=sharing) and put in the data folder. The 4 phases of an abdominal scanner are used. <br/>

## Dependencies

- Python 3.10
- Tensorflow 2.7.0
- Keras 2.7.0
 <br/>

Run train.py
