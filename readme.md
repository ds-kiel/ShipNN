# Project Overview

This repository contains code for preparing audio datasets, generating spectrograms, and running deep learning experiments using MobileNet and ResNet architectures.

---

## Repository Structure

.
├── config  
│   └── mobilenet.yaml  
├── dataset  
│   ├── create_metadata.py  
│   ├── data_loader.py  
│   └── generate_spectrogram.py  
├── mobilenet.py  
├── resnet.py  
├── models.py  
├── run.py  
└── ReadMe.md  

---

## Dataset Preparation (Run Once)

Before running any experiments, you must generate the dataset metadata and spectrograms.

### Step 1: Configure the dataset
Update the configuration file according to your dataset and spectrogram requirements:

config/mobilenet.yaml

### Step 2: Create dataset metadata
Run the following script to generate the required metadata:

python dataset/create_metadata.py

### Step 3: Generate spectrograms
Run the following script to generate spectrograms for all audio files:

python dataset/generate_spectrogram.py

This step only needs to be executed once unless the dataset or spectrogram parameters change.

---

## Running Experiments

After completing dataset preparation, you can run training or evaluation experiments.

### Step 1: Update experiment configuration
Modify the configuration file to select the model, training parameters, and paths:

config/mobilenet.yaml

### Step 2: Run the experiment
Execute the main script:

python run.py

---
