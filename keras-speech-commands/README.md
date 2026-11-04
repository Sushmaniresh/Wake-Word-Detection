## Wake Word Detection using CNN — “Yes” Wake Word

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Librosa](https://img.shields.io/badge/Audio-Librosa-yellow.svg)](https://librosa.org/)

**Offline Wake Word Detection using Deep Learning and MFCC Audio Features**

This project implements a **Convolutional Neural Network (CNN)**–based **wake word detection system**, trained to recognize the **single wake word “Yes”** from the **Google Speech Commands dataset**.  
The system runs fully **offline**, processing microphone input locally using MFCC-based features and TensorFlow inference — without relying on any cloud APIs.

---

##  Features

*  **CNN-based audio classifier** using Mel-Frequency Cepstral Coefficients (MFCCs)  
*  **Offline wake word detection** — no cloud dependency  
*  **Real-time microphone input** using `sounddevice`  
*  **Comprehensive model evaluation** (accuracy/loss curves, confusion matrix, ROC, PR curve)  
*  **Trained on Google Speech Commands v0.02** for reproducible results  
*  **Low latency and fast inference** for edge or embedded systems  

---

##  Project Architecture

```

Microphone Input
↓
Preprocessing (MFCC Extraction)
↓
CNN Model (Wake vs Non-Wake Classification)
↓
Threshold Decision
↓
Wake Word Detection Trigger (“Yes”)

```

---

##  Technical Overview

| Component                  | Description                                                |
| -------------------------- | ---------------------------------------------------------- |
| **Model**                  | Convolutional Neural Network (TensorFlow / Keras)          |
| **Input Features**         | MFCCs (Mel-Frequency Cepstral Coefficients)                |
| **Training Data**          | Google Speech Commands v0.02                               |
| **Wake Word**              | “Yes”                                                     |
| **Detection Method**       | Real-time sliding window (1.0 s) with confidence threshold |
| **Accuracy (Clean Data)**  | ≈ 99 %                                                   |
| **AUC**                    | ≈ 1.00                                                    |
| **Average Precision (PR)** | ≈ 0.99                                                    |

---

##  Directory Structure

```

wakeword_stage1/
│
├── data/
│   ├── processed/                 # Preprocessed MFCC feature arrays
│   └── speech_commands_v0.02/     # Google Speech Commands dataset
│
├── model/
│   ├── wakeword_cnn_stage1.h5     # Trained CNN model
│   └── training_history.json
│
├── src/
│   ├── feature_extraction.py
│   ├── build_dataset.py
│   ├── model_cnn.py
│   ├── train_model.py
│   ├── evaluate_model.py
│
├── visuals/                       # Visualization scripts (accuracy, ROC, etc.)
│
├── detect.py                      # Real-time microphone detection for “Yes”
├── train.py / test.py
└── requirements.txt

````

---

##  Results Summary

| Metric               | Score  |
| -------------------- | ------ |
| **Accuracy**         | 99 % + |
| **Precision (Wake)** | 0.97   |
| **Recall (Wake)**    | 0.92   |
| **F1-Score**         | 0.95   |
| **AUC**              | 1.00   |
| **AP (PR Curve)**    | 0.99   |

---

##  Project Description

> This stage demonstrates the baseline wake word detection system using a CNN model trained to recognize the **word “Yes”**.  
> The audio data is preprocessed into MFCC features, allowing the CNN to efficiently classify wake vs. non-wake inputs.  
> This single-word model forms the foundation for **Stage 2**, which introduces **compound wake phrases** for enhanced reliability.

---

##  Tech Stack

* **Python 3.9+**  
* **TensorFlow / Keras** — Model training and inference  
* **Librosa** — Audio feature extraction (MFCCs, spectrograms)  
* **SoundDevice** — Real-time microphone input  
* **Matplotlib / Seaborn / Scikit-learn** — Model evaluation and metrics visualization  

---

##  How to Run

1. **Build Dataset**

   ```bash
   python src/build_dataset.py
 ``

2. **Train Model**

   ```bash
   python train.py
  ``

3. **Evaluate Model**

   ```bash
   python src/evaluate_model.py
   ```

4. **Run Real-time Detection**

   ```bash
   python detect.py
   ```

Say **“Yes”** clearly — and the system will trigger a detection event 

---

##  Future Enhancements

* Add **background noise augmentation** for robust detection
* Extend to **multi-wake word** or **user-defined wake phrases**
* Integrate with **offline voice assistant modules**
* Explore **CNN-LSTM hybrid** for temporal sequence modeling

---

##  License

This project is licensed under the **MIT License** — feel free to modify and build upon it for research or educational purposes.

---

###  Author

Developed by **Chinmay Karve**
A Deep Learning–based Offline Voice Trigger Detection Project

```

---

✅ **This README is tailored for your “YES” wake word project (Stage 1)** — the base model before your “Cat Bed” extension.

Would you like me to give you a **slightly shortened version** (optimized for a portfolio showcase page, ~350 words, with bold section dividers and emojis for GitHub’s sidebar)?
```
