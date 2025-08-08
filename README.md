# Brain-to-Text '25 Competition

![Competition Status](https://img.shields.io/badge/Status-Open%20~4%20months%20left-green)
![License](https://img.shields.io/badge/License-MIT-blue)
![Prize Pool](https://img.shields.io/badge/Prize%20Pool-$9,000-gold)

## Table of Contents

- [Competition Overview](#competition-overview)
- [Data Description](#data-description)
- [Evaluation Metric](#evaluation-metric)
- [Baseline Solution](#baseline-solution)
- [Environment Setup](#environment-setup)
- [Reproducing Results](#reproducing-results)
- [Project Structure](#project-structure)
- [Tips & Tricks](#tips--tricks)
- [References & Acknowledgments](#references--acknowledgments)

## Competition Overview

This Kaggle competition challenges participants to decode intracortical neural activity recorded 
during attempted speech into corresponding words. Using microelectrode array recordings from a 
speech-impaired participant, teams must develop models that translate brain signals into text, 
advancing brain-computer interface technology for communication restoration.

**Key Dates:**
- **Start Date:** April 8, 2025
- **Submission Deadline:** August 7, 2025 (23:59 UTC)
- **Prize Pool:** US $9,000 distributed among top 3 teams

**Organizers:** Neural Interface Lab & Kaggle Research Competitions  
**Background:** Builds upon the [2024 Brain-to-Text Benchmark](https://doi.org/10.1038/s41593-024-01619-1)

## Data Description

The dataset contains ~12 GB of intracortical recordings across 132 files, capturing neural 
activity from 256-channel Utah arrays implanted in speech motor cortex.

### Dataset Structure
```
brain-to-text-25/
├── train/
│   ├── participant_01/
│   │   ├── session_001.npz  # Neural data + ground truth
│   │   ├── session_002.npz
│   │   └── ...
│   └── metadata.json         # Recording parameters
├── test/
│   └── participant_01/
│       ├── session_101.npz  # Neural data only
│       └── ...
└── sample_submission.csv
```

### Data Format
- **Sampling Rate:** 30 kHz (downsampled to 1 kHz for competition)
- **Channels:** 256 electrodes (96 active after quality filtering)
- **Features:** Spike band power (300-1000 Hz), LFP bands (0-200 Hz)
- **Labels:** Word-level transcriptions with timing information

### Loading Example
```python
import numpy as np
import pandas as pd

# Load a training session
data = np.load('train/participant_01/session_001.npz')
neural_features = data['neural']  # Shape: (timestamps, channels)
transcript = data['transcript']   # Ground truth words
timestamps = data['timestamps']   # Time in seconds

print(f"Recording duration: {timestamps[-1]:.1f} seconds")
print(f"Neural shape: {neural_features.shape}")
print(f"Transcript: {' '.join(transcript[:10])}...")
```

## Evaluation Metric

Submissions are evaluated using **Word Error Rate (WER)**, which measures the edit distance 
between predicted and true transcripts normalized by reference length.

### Formula

$$\text{WER} = \frac{S + D + I}{N} \times 100\%$$

Where:
- **S** = Number of substitutions
- **D** = Number of deletions  
- **I** = Number of insertions
- **N** = Number of words in reference

### Leaderboard Split
- **Public Score:** 30% of test data (shown during competition)
- **Private Score:** 70% of test data (revealed after deadline)

💡 **Note:** Lower WER is better. State-of-art baselines achieve ~15% WER.

## Baseline Solution

The official starter notebook implements an RNN encoder-decoder architecture achieving ~28% WER
on the validation set. The model uses bidirectional LSTM layers to encode neural features and 
an attention-based decoder to generate word sequences.

**Key Components:**
- Feature extraction: Multi-scale temporal convolutions
- Encoder: 3-layer BiLSTM (512 hidden units)
- Decoder: LSTM with Bahdanau attention
- Output: 5000-word vocabulary with beam search

**Resources:**
- [Official Starter Notebook](https://www.kaggle.com/code/competitions/brain-to-text-25/starter)
- [EDA & Visualization](https://www.kaggle.com/code/neuralninja/brain-signals-eda)

## Environment Setup

### Requirements
```txt
python==3.10.12
torch==2.1.0
transformers==4.36.0
numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.0
scipy==1.11.1
matplotlib==3.7.2
tqdm==4.65.0
kaggle==1.5.16
```

### Installation
```bash
# Using conda
conda create -n brain2text python=3.10
conda activate brain2text
pip install -r requirements.txt

# Using pip only
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Kaggle API Setup
```bash
# 1. Download API token from kaggle.com/account
mkdir ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# 2. Test connection
kaggle competitions list
```

## Reproducing Results

### Complete Pipeline
```bash
# 1. Clone repository and navigate
git clone https://github.com/yourusername/brain-to-text-25.git
cd brain-to-text-25

# 2. Download competition data
kaggle competitions download -c brain-to-text-25
unzip brain-to-text-25.zip -d data/

# 3. Train baseline model
python src/train_baseline.py \
    --data_dir data/train \
    --epochs 50 \
    --batch_size 32 \
    --learning_rate 0.001

# 4. Generate predictions
python src/predict.py \
    --model_path models/baseline_best.pth \
    --test_dir data/test \
    --output submissions/baseline_submission.csv

# 5. Submit to Kaggle
kaggle competitions submit \
    -c brain-to-text-25 \
    -f submissions/baseline_submission.csv \
    -m "Baseline RNN submission"
```

### Expected Runtime
- **Training:** ~6 hours on NVIDIA V100, ~18 hours on CPU
- **Inference:** ~30 minutes on GPU, ~90 minutes on CPU  
- **Memory:** 16 GB RAM minimum, 32 GB recommended

## Project Structure

```
brain-to-text-25/
├── data/
│   ├── train/              # Training recordings
│   ├── test/               # Test recordings (no labels)
│   └── processed/          # Preprocessed features
├── src/
│   ├── models/
│   │   ├── baseline.py    # RNN encoder-decoder
│   │   ├── transformer.py # Transformer variant
│   │   └── ensemble.py    # Model ensembling
│   ├── utils/
│   │   ├── data_loader.py # Data processing pipeline
│   │   ├── metrics.py     # WER calculation
│   │   └── augmentation.py # Data augmentation
│   ├── train_baseline.py
│   └── predict.py
├── notebooks/
│   ├── 01_eda.ipynb       # Exploratory analysis
│   ├── 02_baseline.ipynb  # Baseline walkthrough
│   └── 03_advanced.ipynb  # Advanced techniques
├── submissions/            # CSV predictions
├── models/                 # Saved model checkpoints
├── requirements.txt
└── README.md
```

## Tips & Tricks

### Model Improvements
1. **Ensemble Methods:** Combine RNN, Transformer, and CNN architectures
2. **Phoneme Objectives:** Train auxiliary task for phoneme prediction
3. **Language Model Rescoring:** Use GPT-2/BERT to rerank beam search outputs
4. **Data Augmentation:** Time warping, noise injection, channel dropout

### Feature Engineering
- Extract high-gamma power (70-200 Hz) for speech onset detection
- Compute inter-electrode coherence for network dynamics
- Apply Kalman filtering for neural state estimation

### Training Strategies
- **Curriculum Learning:** Start with short sequences, gradually increase
- **Mixed Precision:** Use fp16 training for 2x speedup
- **Learning Rate Schedule:** Cosine annealing with warm restarts

⚠️ **Warning:** Test set has domain shift from training (different session dates). 
Consider domain adaptation techniques.

## References & Acknowledgments

- [Competition Page](https://www.kaggle.com/competitions/brain-to-text-25)
- [Brain-to-Text Benchmark 2024](https://doi.org/10.1038/s41593-024-01619-1)
- [Neural Speech Decoding Review](https://doi.org/10.1146/annurev-neuro-111020-101837)
- [Kaggle Discussion Forums](https://www.kaggle.com/competitions/brain-to-text-25/discussion)

Special thanks to the research participant and clinical team enabling this dataset, and to the
Kaggle Research Competitions team for hosting.

---
*Last updated: August 2025*
