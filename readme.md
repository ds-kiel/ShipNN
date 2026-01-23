# ShipNN  
**On-Device Ship Identification from Underwater Noise**

This repository accompanies the paper:

> **ShipNN: On-Device Ship Identification from Underwater Noise**  
> Kainat Altaf, Momin Ali, Laura Harms, Christian Renner, Olaf Landsiedel  
> *Proceedings of the ACM Symposium on Applied Computing (SAC), 2026*

The code is provided to support **reproducibility, transparency, and further research** in embedded and edge-based underwater acoustic monitoring.

---

## Overview

**ShipNN** implements the complete experimental pipeline used in our paper for ship identification from underwater audio signals under resource-constrained settings.

The repository includes:

- Audio dataset preprocessing and metadata generation  
- Time–frequency feature extraction using spectrogram representations  
- Deep learning models adapted for underwater acoustic data  
- Training and evaluation pipelines aligned with the experimental setup in the paper  

The focus is on **lightweight CNN architectures** suitable for **on-device inference**, enabling deployment on embedded platforms such as microcontrollers and low-power edge devices.

---

## Scope and Intended Use

This repository is intended as a **research codebase** to:

- Reproduce the results reported in the paper  
- Support ablation studies and architectural comparisons  
- Serve as a reference implementation for embedded ship classification  

It is **not** designed as a general-purpose or production-ready framework.

---

## Repository Structure

```
.
├── config/
│   └── mobilenet.yaml         # Experiment configuration used in the paper
│
├── dataset/
│   ├── create_metadata.py     # Audio metadata generation
│   ├── data_loader.py         # Dataset and DataLoader definitions
│   └── generate_spectrogram.py# Spectrogram extraction utilities
│
├── mobilenet.py               # MobileNet-based model architecture
├── resnet.py                  # ResNet-based baseline architecture
├── models.py                  # Shared model components
├── run.py                     # Main entry point for training and evaluation
└── ReadMe.md                  # Documentation
```

---

## Experimental Pipeline

The workflow follows the experimental methodology described in the paper:

1. **Metadata Generation**  
   Raw underwater audio recordings are indexed and labeled using scripts in `dataset/`.

2. **Feature Extraction**  
   Audio signals are transformed into time–frequency representations suitable for CNN-based learning.

3. **Model Configuration**  
   Model architecture, preprocessing parameters, and training settings are defined via YAML configuration files.

4. **Training and Evaluation**  
   Experiments are executed through a unified pipeline using `run.py`.

---

## Usage

To run an experiment using the configuration reported in the paper:

```bash
python run.py --config config/mobilenet.yaml
```

All key hyperparameters and preprocessing settings are defined in the configuration file.

---

## Reproducibility Notes

- Results may vary slightly depending on hardware, random seeds, and software versions.
- The repository focuses on the **training and evaluation pipeline**; embedded deployment and latency measurements are discussed in the paper.

---

## Citation

If you use this code in your research, please cite the following paper:

```bibtex
@inproceedings{altaf2026shipnn,
  title     = {ShipNN: On-Device Ship Identification from Underwater Noise},
  author    = {Altaf, Kainat and Ali, Momin and Harms, Laura and Renner, Christian and Landsiedel, Olaf},
  booktitle = {Proceedings of the ACM Symposium on Applied Computing (SAC)},
  year      = {2026}
}
```

---

## License

This code is released for **academic and research use**.  
Please refer to the repository for license details.
