## Data Exploration

- Loaded and explored a real-world fake news dataset
- Fixed corrupted CSV rows and label inconsistencies
- Ensured clean binary labels (0 = fake, 1 = real)
- Observed that real news articles are longer and more structured
- Fake news articles tend to be shorter and emotionally charged
- Minimal preprocessing chosen to preserve semantic context
- Saved a cleaned dataset for modeling 

## Text Classification Baseline

A classical machine learning baseline is built using TF-IDF features and
Logistic Regression for fake news classification.

### Notebook
`notebooks/text_baseline.ipynb`

### Steps performed
- Loaded the cleaned fake news dataset
- Split data into training and test sets
- Converted raw text into TF-IDF feature vectors
- Trained a Logistic Regression classifier with class balancing
- Evaluated the model using accuracy, precision, recall, F1-score, and confusion matrix
- Achieved strong baseline performance (~94% accuracy)
- Saved the trained model and vectorizer for reuse

### Saved artifacts
The trained model and vectorizer are stored for inference:


## Inference Demo
A trained TF-IDF + Logistic Regression model is loaded from disk and used
to classify unseen news text with confidence scores.

## Explainability

The fake news classifier is made interpretable using the linear nature of
TF-IDF features combined with Logistic Regression.

### Global Explainability
- Extracted word-level coefficients from the trained model
- Identified top words that generally indicate fake and real news

### Local Explainability
- Explained individual predictions by computing word-level contributions
- Shows how specific words influenced a given prediction

Notebook:
`notebooks/explainability.ipynb`

## Model Comparison

Classical machine learning models were compared using identical
TF-IDF features to justify baseline selection.

Models:
- Logistic Regression
- Naive Bayes
- Linear SVM

Notebook:
`notebooks/model_comparison.ipynb`

## Error Analysis & Model Selection

Performed systematic error analysis on the baseline TF-IDF + Logistic Regression
model to identify misclassification patterns and limitations.

Key observations from error analysis were used to compare classical models
with transformer-based approaches and guide final model selection.

Notebook:
`notebooks/error_analysis.ipynb`

