# Text-Sentiment-Analysis
# Sentiment Analysis on IMDB Reviews

## Introduction
This project implements a **sentiment analysis model** using the **IMDB movie reviews dataset**. The goal is to classify reviews as either positive or negative based on their textual content. The workflow includes text preprocessing, feature engineering, model training, and evaluation.

## Dataset
The dataset used is the **IMDB Reviews Dataset**, which consists of **50,000 movie reviews** labeled as positive or negative. The dataset was obtained from the [Stanford AI Lab](https://ai.stanford.edu/~amaas/data/sentiment/).

## Steps Implemented

### 1. Text Preprocessing
To prepare the text data, the following preprocessing steps were applied:
- **Tokenization**: Splitting text into individual words.
- **Stopword Removal**: Removing common words that do not contribute to sentiment (e.g., "the", "is").
- **Lemmatization**: Converting words to their base forms (e.g., "running" → "run").

### 2. Feature Engineering
To convert text into numerical format for model training, the **TF-IDF (Term Frequency-Inverse Document Frequency)** approach was used. This method represents text based on word importance in a document while reducing common-word influence.

### 3. Model Training
A **Logistic Regression** classifier was trained to predict sentiment labels. The dataset was split into **80% training and 20% testing** to evaluate the model's generalization ability.

### 4. Model Evaluation
The trained model was evaluated using the following metrics:
- **Precision**: Measures how many predicted positive instances were actually positive.
- **Recall**: Measures how many actual positive instances were correctly identified.
- **F1-Score**: Harmonic mean of precision and recall, providing an overall performance measure.

## Results
The model achieved good performance on the test set, demonstrating the effectiveness of **TF-IDF with Logistic Regression** for sentiment analysis.

## Future Improvements
- Implement **word embeddings** like Word2Vec or BERT for improved feature representation.
- Train **deep learning models** (e.g., LSTMs or transformers) for better accuracy.
- Deploy the model using a **Flask or FastAPI** web application.

## Conclusion
This project demonstrates a **machine learning-based approach to sentiment analysis** using text preprocessing, TF-IDF vectorization, and Logistic Regression. Future improvements can enhance performance using deep learning and advanced embeddings.

---
Feel free to contribute or suggest improvements to this project!

