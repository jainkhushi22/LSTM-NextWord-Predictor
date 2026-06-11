# LSTM-NextWord-Predictor

A deep learning-based Natural Language Processing project that predicts the next word in a sequence using a stacked LSTM architecture. The model is trained on Shakespeare's *Hamlet* corpus and deployed as an interactive web application with Streamlit.

**Live Application:** https://nextwordpredictor-lstm.streamlit.app/

---

## Overview

Language modeling is a fundamental NLP task that enables machines to understand contextual relationships between words. This project focuses on next-word prediction, where the model learns sequential patterns from text and predicts the most probable word that follows a given input sequence.

The entire pipeline includes text preprocessing, sequence generation, tokenization, embedding representation, LSTM-based learning, and deployment through Streamlit.

---

## Features

* Predicts the next word from a given text sequence
* Built using a stacked LSTM neural network
* Trained on Shakespeare's *Hamlet* corpus
* Interactive web interface using Streamlit
* End-to-end NLP workflow implementation
* Model and tokenizer persistence for inference

---

## Project Structure

```text
NextWord-LSTM/
│
├── app.py
├── experiments.ipynb
├── hamlet.txt
├── next_word_LSTM.h5
├── tokenizer.pickle
├── requirements.txt
└── README.md
```

---

## Dataset

The model is trained on the *Hamlet* text available through the NLTK Gutenberg Corpus.

Dataset Source:
https://www.nltk.org/book/ch02.html

---

## Model Architecture

```text
Embedding Layer
        ↓
LSTM (150 Units)
        ↓
Dropout (0.2)
        ↓
LSTM (100 Units)
        ↓
Dense Layer (Softmax)
```

### Training Configuration

| Parameter     | Value                    |
| ------------- | ------------------------ |
| Optimizer     | Adam                     |
| Loss Function | Categorical Crossentropy |
| Epochs        | 100                      |
| Architecture  | Stacked LSTM             |
| Framework     | TensorFlow / Keras       |

---

## Workflow

### 1. Text Collection

The Hamlet corpus is loaded from the Gutenberg dataset and prepared for training.

### 2. Text Preprocessing

* Cleaning and formatting text
* Tokenization using Keras Tokenizer
* Vocabulary creation
* Sequence generation

### 3. Sequence Preparation

N-gram sequences are generated and padded to ensure consistent input length.

### 4. Feature Engineering

The dataset is divided into:

* Input Sequence (X)
* Target Word (y)

Target labels are converted using one-hot encoding.

### 5. Model Training

A stacked LSTM network learns contextual dependencies between words and captures sequential information from the corpus.

### 6. Inference

The trained model predicts the most probable next word for a given input sequence.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/jainkhushi22/LSTM-NextWord-Predictor.git
```

Move into the project directory:

```bash
cd LSTM-NextWord-Predictor
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Application

Launch the Streamlit application:

```bash
streamlit run app.py
```

The application will be available at:

```text
http://localhost:8501
```

---

## Sample Prediction

| Input Sequence           | Predicted Word |
| ---------------------    | -------------- |
| To be or not to be       | sweet          |
| Mar. Is it not like the  | King           |
| My lord                  |  you           |

Predictions may vary depending on the trained model version.

---

## Skills Demonstrated

* Natural Language Processing
* Text Preprocessing
* Tokenization
* Language Modeling
* Sequence Modeling
* Deep Learning
* LSTM Networks
* TensorFlow & Keras
* Streamlit Deployment

---

## Future Enhancements

* Train on larger Shakespeare corpora
* Add top-k word predictions
* Extend to multi-word text generation
* Experiment with Transformer-based models

---

## Tech Stack

* Python
* TensorFlow
* Keras
* NumPy
* NLTK
* Streamlit

---

## Connect With Me

LinkedIn: https://www.linkedin.com/in/khushi-jain-009836307

GitHub: https://github.com/jainkhushi22

---

If you found this project interesting, consider starring the repository and sharing feedback.
