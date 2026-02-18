# Fine-Tuning Whisper (Medium) for Swahili ASR 🇹🇿🎙️  
**Automatic Speech Recognition (ASR) adaptation using Mozilla Common Voice v17**

> **Project Type:** Small Machine Learning Experiment (Fine-tuning + Evaluation)  
> **Model:** `openai/whisper-medium` (764M parameters)  
> **Dataset:** Mozilla Common Voice 17.0 (Swahili)  
> **Core Metrics:** Word Error Rate (WER), Real-Time Factor (RTF), Peak Memory

---

## 📌 Project Overview

This project presents an end-to-end machine learning experiment for fine-tuning OpenAI’s **Whisper-medium** model on the **Swahili** language using the **Mozilla Common Voice v17** dataset.

The goal is to improve Whisper’s transcription quality for Swahili speech by adapting the pretrained multilingual ASR model using supervised fine-tuning. The pipeline covers:

- Dataset loading and preprocessing
- Audio resampling and tokenization
- Fine-tuning Whisper using HuggingFace Trainer
- Evaluation using WER
- Inference and demonstration on test samples
- Reporting of latency and compute resource metrics (RTF and peak memory)

This repo is designed to be **reproducible**, **clean**, and **reviewer-friendly**.

---

## 🎯 Why This Project Matters

Swahili is one of Africa’s most spoken languages, yet it remains underrepresented in many speech recognition systems.

By fine-tuning Whisper on Swahili speech data, this work demonstrates how modern foundation models can be adapted to support low-resource languages, improving:

- accessibility tools (voice assistants, captioning)
- local language technology development
- speech-driven education systems
- inclusive AI systems for underrepresented communities

---

## 🧠 Model Description

Whisper is a multilingual encoder-decoder transformer model trained for speech recognition and translation.

- **Model used:** `openai/whisper-medium`
- **Parameters:** ~764 million
- **Architecture:** Transformer encoder-decoder
- **Framework:** HuggingFace Transformers

---

## 📂 Dataset

We use the **Swahili subset** of the Mozilla Common Voice 17.0 dataset.

**Dataset properties**
- Open-source crowd-sourced voice recordings
- Contains audio clips and their reference transcripts
- Highly diverse speaker population

---

## 🔧 Project Structure

```bash
.
├── README.md
├── requirements.txt
├── notebooks/
│   ├── 1_DATA_PREPROCESSING_TOKENIZATION.ipynb
│   ├── 2_FINETUNING_MODELLING.ipynb
│   └── 3_INFERENCING_DEMO.ipynb
├── scripts/
│   ├── preprocess.py
│   ├── train.py
│   └── infer.py
├── data/
│   ├── sample/
│   └── README_data.md
└── results/
    └── run_report.md
```

---

## 🚀 Quick Start (Reviewer-Friendly)

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/yourusername/whisper-swahili-finetuning.git
cd whisper-swahili-finetuning
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run Inference Demo (Fast)

Open the notebook:

📌 `notebooks/3_INFERENCING_DEMO.ipynb`

This notebook runs a small inference test and computes WER on a small sample dataset.

---

## 🏋️ Full Training Pipeline (Reproducibility)

### Step 1: Data Preprocessing

📌 Notebook:  
`notebooks/1_DATA_PREPROCESSING_TOKENIZATION.ipynb`

This notebook performs:
- dataset loading from Common Voice
- audio resampling to 16kHz
- transcript cleaning
- tokenization using Whisper tokenizer
- creation of train/validation splits

---

### Step 2: Fine-Tuning Whisper

📌 Notebook:  
`notebooks/2_FINETUNING_MODELLING.ipynb`

Training is performed using HuggingFace `Seq2SeqTrainer`.

**Training hyperparameters**
- Model: `openai/whisper-medium`
- Epochs: `3`
- Learning rate: `1e-5`
- Warmup steps: `500`
- Batch size: `16`
- Gradient accumulation: `2`
- Mixed precision: `fp16=True`
- Evaluation metric: WER

---

### Step 3: Inference + Evaluation

📌 Notebook:  
`notebooks/3_INFERENCING_DEMO.ipynb`

This notebook performs:
- loading trained checkpoint
- transcription on validation/test audio
- computation of WER
- runtime and memory analysis

---

## 📊 Evaluation Metrics

### ✅ Word Error Rate (WER)

WER is computed as:

\[
WER = \frac{S + D + I}{N}
\]

Where:
- **S** = substitutions  
- **D** = deletions  
- **I** = insertions  
- **N** = number of words in reference transcript  

Lower WER indicates better transcription performance.

---

### ⚡ Real-Time Factor (RTF)

RTF measures inference speed:

\[
RTF = \frac{Inference \ Time}{Audio \ Duration}
\]

- RTF < 1 means faster-than-real-time transcription.

---

### 🧠 Peak Memory Usage

Peak memory usage is measured to evaluate feasibility for deployment under limited GPU environments.

---

## 💻 Hardware Requirements

### Recommended
- NVIDIA GPU (T4, L4, A100, RTX series)
- 16GB+ VRAM preferred for Whisper-medium fine-tuning
- Python 3.9+

### Notes
- Fine-tuning is compute-intensive.
- Inference notebooks can be run on CPU (slower but works).

---

## 🧪 Experiment Settings

| Component | Value |
|----------|-------|
| Model | Whisper-medium |
| Dataset | Common Voice v17 Swahili |
| Optimizer | AdamW |
| Learning Rate | 1e-5 |
| Epochs | 3 |
| Batch Size | 16 |
| Gradient Accumulation | 2 |
| Precision | FP16 |
| Metric | WER |

---

## 📌 Key Contributions

This repository demonstrates:

✅ Full fine-tuning pipeline for Swahili ASR  
✅ Clean preprocessing and tokenization workflow  
✅ Whisper fine-tuning with Seq2SeqTrainer  
✅ Evaluation using WER  
✅ Inference demo notebook  
✅ Performance reporting (RTF + memory)  
✅ Clear documentation and reproducibility

---

## 📜 Reproducibility Notes

- All preprocessing and training steps are documented in notebooks.
- Hyperparameters and evaluation methodology are clearly stated.
- Results can be reproduced by running notebooks sequentially.
- A small sample dataset is included for quick evaluation.

---

## 🛡️ Relevance to AI Safety

This project is aligned with AI safety evaluation principles because it focuses on:

- **measuring model performance** rigorously (WER)
- **detecting failure modes** in low-resource language transcription
- reporting computational constraints (latency and memory)
- improving fairness and accessibility for underrepresented languages

Reliable speech recognition reduces risks of harmful mis-transcription in sensitive applications such as healthcare, education, and legal transcription.

---

## 📌 Future Work

Possible improvements include:

- adding audio augmentation (noise injection, speed perturbation)
- training Whisper-large for comparison
- experimenting with LoRA / PEFT to reduce compute cost
- evaluating robustness to accents and background noise
- building a lightweight Swahili speech benchmark

---

## 🙌 Acknowledgements

- OpenAI Whisper model
- HuggingFace Transformers & Datasets
- Mozilla Common Voice contributors
- Evaluation metrics implemented using HuggingFace `evaluate`

---

## 📎 References

- OpenAI Whisper: https://github.com/openai/whisper  
- HuggingFace Whisper Fine-Tuning Guide: https://huggingface.co/blog/fine-tune-whisper  
- Mozilla Common Voice: https://commonvoice.mozilla.org/en/datasets  

---

## 📄 License

This project is released under the MIT License (or specify your preferred license).

---

## 👤 Author

**Adegoke Adedolapo Israel**  
Artificial Intelligence Engineer | All Tech  
📧 Email: adegisrael198@gmail.com  

---
