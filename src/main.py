import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from nltk.corpus import stopwords
from scipy.special import softmax
import nltk
import re
import pickle
import json
import os
from tqdm import tqdm
import time

# Constants
CONFIDENCE_THRESHOLD = 0.3

# Download required NLTK data
print("Downloading NLTK data...")
nltk.download('stopwords', quiet=True)
STOPWORDS = set(stopwords.words('english'))

# Class mapping for better understanding
class_mapping = {
    0: 'hateful',
    1: 'offensive',
    2: 'neutral'
}

def preprocess(text):
    """Improved preprocessing pipeline"""
    # Convert to lowercase
    text = str(text).lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove user @ references and '#' from tweet
    text = re.sub(r'\@\w+|\#\w+', '', text)
    # Remove punctuation except for specific ones that might indicate sentiment
    text = re.sub(r'[^\w\s!?]', '', text)
    # Tokenize
    tokens = text.split()
    # Remove stopwords
    tokens = [word for word in tokens if word not in STOPWORDS]
    # Join tokens back
    return ' '.join(tokens)

def normalize_probabilities(probabilities):
    """Apply softmax to normalize probabilities"""
    return softmax(probabilities)

def test_classification(text, tfidf_vectorizer, nb_model, lr_model):
    """Test classification for a single text input"""
    processed = preprocess(text)
    
    if not processed.strip():
        return "neutral", 1.0, "Text contains only stopwords or no meaningful content"
    
    vector = tfidf_vectorizer.transform([processed]).toarray()
    
    # Get predictions from both models
    nb_probs = normalize_probabilities(nb_model.predict_proba(vector)[0])
    lr_probs = normalize_probabilities(lr_model.predict_proba(vector)[0])
    
    nb_pred = nb_model.predict(vector)[0]
    lr_pred = lr_model.predict(vector)[0]
    
    nb_conf = nb_probs[nb_pred]
    lr_conf = lr_probs[lr_pred]
    
    # Use the model with higher confidence
    if nb_conf > lr_conf:
        prediction = nb_pred
        confidence = nb_conf
    else:
        prediction = lr_pred
        confidence = lr_conf
    
    # If confidence is too low, return neutral
    if confidence < CONFIDENCE_THRESHOLD:
        return "neutral", confidence, f"Low confidence ({confidence:.1%})"
    
    return class_mapping[prediction], confidence, f"Confidence: {confidence:.1%}"

def preprocess_with_progress(texts):
    """Preprocess texts with progress bar"""
    processed_texts = []
    for text in tqdm(texts, desc="Preprocessing texts", unit="text"):
        processed = preprocess(text)
        processed_texts.append(processed)
    return processed_texts

def main():
    print("\n=== Starting Training Process ===\n")
    
    print("Step 1/7: Loading data...")
    try:
        data = pd.read_csv('data/labeled_data.csv')
        # Rename columns if needed
        if 'tweet' in data.columns:
            data = data.rename(columns={'tweet': 'text'})
        print(f"✓ Loaded {len(data)} samples")
        
        print("\nClass distribution before balancing:")
        print(data['class'].value_counts())
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return
    
    print("\nStep 2/7: Preprocessing data...")
    try:
        # Convert to list for progress bar
        texts = data['text'].tolist()
        processed_texts = preprocess_with_progress(texts)
        
        # Split into training and testing sets
        print("\nStep 3/7: Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(
            processed_texts, 
            data['class'],
            test_size=0.2,
            random_state=42,
            stratify=data['class']
        )
        print(f"✓ Training set: {len(X_train)} samples")
        print(f"✓ Testing set: {len(X_test)} samples")
    except Exception as e:
        print(f"Error in preprocessing: {str(e)}")
        return
    
    print("\nStep 4/7: Extracting features...")
    try:
        # Create and fit TF-IDF vectorizer
        tfidf = TfidfVectorizer(max_features=10000)
        print("- Fitting TF-IDF vectorizer...")
        X_train_tfidf = tfidf.fit_transform(X_train)
        X_test_tfidf = tfidf.transform(X_test)
        print(f"✓ Extracted {X_train_tfidf.shape[1]} features")
    except Exception as e:
        print(f"Error in feature extraction: {str(e)}")
        return
    
    print("\nStep 5/7: Applying SMOTE for class balancing...")
    try:
        smote = SMOTE(random_state=42)
        with tqdm(total=100, desc="Balancing classes") as pbar:
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train_tfidf, y_train)
            pbar.update(100)
        
        print("\nClass distribution after balancing:")
        print(pd.Series(y_train_balanced).value_counts())
    except Exception as e:
        print(f"Error in SMOTE: {str(e)}")
        return
    
    print("\nStep 6/7: Training models...")
    try:
        # Train Naive Bayes
        print("\nTraining Naive Bayes...")
        nb_model = MultinomialNB()
        with tqdm(total=100, desc="Training Naive Bayes") as pbar:
            nb_model.fit(X_train_balanced, y_train_balanced)
            pbar.update(100)
        
        # Train Logistic Regression
        print("\nTraining Logistic Regression...")
        lr_model = LogisticRegression(
            class_weight='balanced',
            max_iter=2000,
            C=0.3,
            solver='lbfgs',
            multi_class='multinomial',
            random_state=42
        )
        with tqdm(total=100, desc="Training Logistic Regression") as pbar:
            lr_model.fit(X_train_balanced, y_train_balanced)
            pbar.update(100)
    except Exception as e:
        print(f"Error in model training: {str(e)}")
        return
    
    print("\nStep 7/7: Evaluating and saving models...")
    try:
        # Evaluate models
        y_pred_nb = nb_model.predict(X_test_tfidf)
        y_pred_lr = lr_model.predict(X_test_tfidf)
        
        print("\nNaive Bayes Results:")
        print("Classification Report:")
        print(classification_report(y_test, y_pred_nb, target_names=['Hateful', 'Offensive', 'Neutral']))
        
        print("\nLogistic Regression Results:")
        print("Classification Report:")
        print(classification_report(y_test, y_pred_lr, target_names=['Hateful', 'Offensive', 'Neutral']))
        
        # Save models and configuration
        print("\nSaving models and configuration...")
        if not os.path.exists('models'):
            os.makedirs('models')
        
        with tqdm(total=4, desc="Saving files") as pbar:
            # Save models
            with open('models/naive_bayes.pkl', 'wb') as f:
                pickle.dump(nb_model, f)
                pbar.update(1)
            
            with open('models/logistic_regression.pkl', 'wb') as f:
                pickle.dump(lr_model, f)
                pbar.update(1)
            
            with open('models/tfidf_vectorizer.pkl', 'wb') as f:
                pickle.dump(tfidf, f)
                pbar.update(1)
            
            # Save configuration
            config = {
                "class_mapping": {
                    "0": "hateful",
                    "1": "offensive",
                    "2": "neutral"
                },
                "confidence_threshold": CONFIDENCE_THRESHOLD
            }
            with open('models/config.json', 'w') as f:
                json.dump(config, f, indent=4)
                pbar.update(1)
        
        print("\n✓ Training completed successfully!")
        print("\nModel files saved in 'models/' directory:")
        print("- naive_bayes.pkl")
        print("- logistic_regression.pkl")
        print("- tfidf_vectorizer.pkl")
        print("- config.json")
        
    except Exception as e:
        print(f"Error in evaluation and saving: {str(e)}")
        return

if __name__ == '__main__':
    main()
