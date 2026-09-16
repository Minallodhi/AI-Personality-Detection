# AI Personality Detection

## Overview

AI Personality Detection is a Python machine learning project that uses Natural Language Processing (NLP) to classify text into different personality traits.

The project uses **TF-IDF** to convert text into numerical features and **Logistic Regression** for classification.

## Personality Traits

The prototype classifies text into:

* Openness
* Conscientiousness
* Extraversion
* Neuroticism
* Agreeableness

## How It Works

```text
Input Text
    ↓
TF-IDF Vectorization
    ↓
Logistic Regression
    ↓
Personality Trait Prediction
```

## Technologies Used

* Python
* Pandas
* Scikit-learn
* TF-IDF
* Logistic Regression
* Joblib

## Project Structure

```text
AI-Personality-Detection/
│
├── train_model.py
├── requirements.txt
└── README.md
```

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Minallodhi/AI-Personality-Detection.git
```

### 2. Open the Project Folder

```bash
cd AI-Personality-Detection
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Run the Project

Run the training script:

```bash
python train_model.py
```

The trained model and TF-IDF vectorizer are saved in the `models/` directory.

## Example

Example text:

```text
I enjoy socializing with friends.
```

The trained model can use the text to predict one of the personality traits.

## Note

This is a learning prototype using a small sample dataset to demonstrate NLP-based text classification. It is not intended to be used as a validated psychological assessment tool.

## Author

**Minal Lodhi**

# Example text input
text = ["I enjoy socializing with friends."]

X = vectorizer.transform(text)
prediction = model.predict(X)

print(prediction)
