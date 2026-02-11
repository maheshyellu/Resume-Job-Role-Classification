# Resume Job Role Classification
## Project Overview

This project builds an end-to-end Resume Classification System that automatically predicts the most relevant job role based on resume content.

The system uses Natural Language Processing (NLP) for text preprocessing and feature extraction, followed by multiple machine learning models for classification. The best-performing model is deployed using Streamlit for real-time prediction.

## Objective

To classify resumes into predefined job roles using textual content analysis and supervised machine learning techniques.

## Dataset Information

- Total resumes: 78

- Number of job roles: 4

- Text-based classification problem

- Best achieved accuracy: 93%

Due to the small dataset size, evaluation results should be interpreted carefully.

## Project Pipeline
### 1️.Text Preprocessing

Performed the following NLP steps:

- Lowercasing

- Removal of punctuation and special characters

- Tokenization

- Stopword removal

- Lemmatization

This ensures clean and normalized textual input for modeling.

### 2️.Feature Extraction

Used TF-IDF (Term Frequency – Inverse Document Frequency) to convert resume text into numerical vectors.

### Why TF-IDF?

- Captures word importance within documents

- Reduces dominance of frequent but less meaningful words

- Suitable baseline for text classification problems

### 3️.Model Training & Evaluation

The following models were trained and compared:

- Logistic Regression

- Support Vector Machine (SVM)

- Naive Bayes

- Random Forest

Models were evaluated using accuracy on the test dataset.
The best-performing model achieved 93% accuracy and was selected for deployment.

### 4️.Model Deployment (Streamlit)

A Streamlit web application was developed to:

- Accept resume text (paste or upload file)

- Automatically preprocess text

- Transform input using saved TF-IDF vectorizer

- Predict job role using trained model

- Display classification result instantly

Supported formats:

- .txt

- .pdf

- .docx

## Technologies Used

- Python

- Pandas

- NumPy

- Scikit-learn

- NLTK

- Streamlit

- Jupyter Notebook
