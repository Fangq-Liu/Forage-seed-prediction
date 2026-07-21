# Forage Seed Prediction: A Deep Learning Pipeline for Multispectral Imaging

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

> **Official Implementation** for the paper:  
> *"A deep learning pipeline enables robust phenotype measurement of small forage seeds via multispectral imaging"* (**Measurement**, IF: 6.0).  
> 🔗 [Link to the paper](<https://doi.org/10.1016/j.measurement.2025.119820>)

## 📖 Introduction
This repository provides a robust deep learning framework for the rapid and accurate classification of forage seeds with highly similar phenotypes. 

### ✨ Key Features
*   **Feature Optimization:** Integrates normalized Canonical Discriminant Analysis (**nCDA**) with ResNet-18 to effectively reduce multispectral feature redundancy.
*   **Uncertainty Quantification:** Employs **MC Dropout** for robust model evaluation and anti-noise performance.
*   **Explainable AI:** Utilizes **SHAP** attribution analysis to identify key decision wavelengths (e.g., 630 nm), providing biological interpretability for the deep learning predictions.

# Forage-seed-prediction
ResNet-18(or CNN) based deep learning model for rapid classification of forage seed with tiny difference.

## File Structure
```
your_repo/
├── model/                  # Pretrained models
├── training_sample/        # Training dataset
├── predicting_sample/      # Sample images for prediction
├── predict.py              # Prediction script
├── train.py                # Training script
├── evaluate.py             # Evaluation script
├── model_CNN.py 
└── model_resnet18.py       # Model architecture
```

## Environment Setup
Python 3.7.2 
```bash
# Install dependencies
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow pillow
```

## Quick Start
### Single Image Prediction
```bash
python predict_resnet18.py predicting_sample/Dough.png --show
```
Loading and Result：

![Single Prediction](./picture/single_seed.png)

Show：

![Single Prediction](./picture/single_seed_show.png)

### Batch Prediction
```bash
python predict_resnet18.py predicting_sample/
```
Loading：

![Single Prediction](./picture/load.png)

Result：

![Single Prediction](./picture/result.png)


## Model Training

Suitable for simulating training on personal computers, this case uses Intel (R) Core (TM) i7-14700HX 2.10 GHz CPU for testing

The image data used for simulating training is provided in training_Sample. zip

1. Prepare data:
```bash
unzip training_sample.zip -d ./training_sample
```

2. Start training:
```bash
python training_resnet18.py --data_dir ./training_sample
```

## Model Evaluation
```bash
python evaluate_resnet18.py
```
Output includes:
- 📊 Confusion Matrix: `./eval_results/confusion_matrix.png`
- 📈 ROC Curve: `./eval_results/roc_curve.png`
- ✅ Metrics: `./eval_results/metrics.txt`

Example of Prediction Results：

![Evaluation Results](./picture/90predict_result.png)

## Customization
### Training Parameters
Modify in `training_resnet18.py`:
```python
# Hyperparameters
EPOCHS = 30
BATCH_SIZE = 32
IMG_SIZE = 224  # Match image resizing
```

### Model Path
Update in `predict_resnet18.py`:
```python
model_path = './model/your_custom_model'  # Default: './model/pretrained_model'
```

## FAQ
❓ **Path errors**  
Use relative paths `./predicting_sample/` or absolute paths

❓ **Dependency conflicts**  
Recommend using virtual environment

❓ **Visualization issues**  
Ensure GUI support when using `--show` flag
