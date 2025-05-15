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
CONFIDENCE_THRESHOLD = 0.4  # Increased from 0.3
HIGH_CONFIDENCE_THRESHOLD = 0.8

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

# Word sets for classification
PROTECTED_GROUPS = {
    'muslim', 'jew', 'christian', 'black', 'white', 'asian',
    'gay', 'lesbian', 'trans', 'queer', 'immigrant', 'foreigner',
    'muslims', 'jews', 'christians', 'blacks', 'whites', 'asians',
    'immigrants', 'foreigners', 'lgbt', 'hispanic', 'arab', 'chinese'
}

HATE_SPEECH_INDICATORS = {
    'die', 'kill', 'murder', 'eliminate', 'destroy', 'hate',
    'terrorist', 'terrorists', 'animals', 'vermin', 'cockroach',
    'scum', 'filth', 'trash', 'garbage', 'disease', 'plague',
    'exterminate', 'deport', 'ban', 'evil', 'death', 'dead'
}

OFFENSIVE_WORDS = {
    'terrible', 'stupid', 'idiot', 'ugly', 'dumb', 'fool',
    'fuck', 'fucking', 'shit', 'damn', 'ass', 'asshole',
    'crap', 'suck', 'sucks', 'bad', 'worst', 'hate'
}

def preprocess(text):
    """Improved preprocessing pipeline with better word handling"""
    # Convert to lowercase
    text = str(text).lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove user @ references and preserve hashtag content
    text = re.sub(r'\@\w+', '', text)
    text = re.sub(r'\#(\w+)', r'\1', text)
    
    # Handle common offensive word variations
    text = text.replace('f*ck', 'fuck')
    text = text.replace('f**k', 'fuck')
    text = text.replace('sh*t', 'shit')
    text = text.replace('b*tch', 'bitch')
    text = text.replace('a**', 'ass')
    
    # Remove punctuation except for specific ones that might indicate sentiment
    text = re.sub(r'[^\w\s!?]', '', text)
    
    # Tokenize
    tokens = text.split()
    
    # Remove stopwords but keep important negation words
    important_words = {'no', 'not', 'nor', 'none', 'never', 'neither'}
    tokens = [word for word in tokens if word not in STOPWORDS or word in important_words]
    
    # Handle common offensive words that aren't hate speech
    offensive_replacements = {
        'terrible': 'bad',
        'stupid': 'bad',
        'idiot': 'bad',
        'ugly': 'bad',
        'dumb': 'bad',
        'fool': 'bad'
    }
    
    tokens = [offensive_replacements.get(word, word) for word in tokens]
    
    # Join tokens back
    return ' '.join(tokens)

def normalize_probabilities(probabilities):
    """Apply softmax to normalize probabilities"""
    return softmax(probabilities)

def classify_text(text, words):
    """Helper function to determine text category based on word presence"""
    has_protected_group = bool(words.intersection(PROTECTED_GROUPS))
    has_hate_indicator = bool(words.intersection(HATE_SPEECH_INDICATORS))
    has_offensive = bool(words.intersection(OFFENSIVE_WORDS))
    
    # More robust hate speech detection
    if has_protected_group and has_hate_indicator:
        return 0, 0.9  # Hateful with high confidence
    elif "death to" in text.lower() or "kill all" in text.lower():
        return 0, 0.9  # Explicit death threats are always hateful
    elif has_offensive:
        return 1, 0.8  # Offensive with good confidence
    return None, None  # Let the model decide

def classify_text_with_models(text, tfidf_vectorizer, nb_model, lr_model):
    """Enhanced classification with rule-based verification"""
    if not text or not str(text).strip():
        return "neutral", 1.0, "Empty or meaningless content"
    
    # Check for explicit hate speech phrases before preprocessing
    text_lower = str(text).lower()
    if any(phrase in text_lower for phrase in ["death to", "kill all", "die all"]):
        return "hateful", 0.9, "Rule-based classification with 90.0% confidence"
    
    # Process the text
    processed = preprocess(text)
    if not processed.strip():
        return "neutral", 1.0, "Empty or meaningless content"
    
    # Check rule-based classification
    words = set(processed.lower().split())
    rule_pred, rule_conf = classify_text(processed, words)
    
    # Use rule-based result if available and confident
    if rule_pred is not None:
        return class_mapping[rule_pred], rule_conf, f"Rule-based classification with {rule_conf:.1%} confidence"
    
    # Get model predictions
    vector = tfidf_vectorizer.transform([processed]).toarray()
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
    
    # Additional verification for hate speech predictions
    if prediction == 0:  # If predicted hateful
        if not (bool(words.intersection(PROTECTED_GROUPS)) or bool(words.intersection(HATE_SPEECH_INDICATORS))):
            # If no protected groups or hate indicators, downgrade to offensive
            prediction = 1
            confidence = max(confidence, 0.7)  # Boost confidence for clear offensive content
    
    return class_mapping[prediction], confidence, f"Model classification with {confidence:.1%} confidence"

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
        texts = data['text'].tolist()
        processed_texts = preprocess_with_progress(texts)
        
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
        # Train Naive Bayes with adjusted priors
        print("\nTraining Naive Bayes...")
        nb_model = MultinomialNB(class_prior=[0.2, 0.4, 0.4])  # Adjusted priors
        with tqdm(total=100, desc="Training Naive Bayes") as pbar:
            nb_model.fit(X_train_balanced, y_train_balanced)
            pbar.update(100)
        
        # Train Logistic Regression with adjusted parameters
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
        
        # Test examples
        print("\n=== Testing with Example Texts ===\n")
        test_examples = [
            "I love everyone and wish them the best!",
            "I hate all Muslims and they should die!",
            "You're a f*cking idiot!",
            "The weather is nice today",
            "Kill all immigrants",
            "This movie was terrible",
            "Death to all Jews",
            "@user I really enjoyed your post about #AI",
            "You're stupid and ugly",
            "This is a neutral statement about technology",
            "The service at this restaurant was awful",
            "These people are cockroaches and should be eliminated",
            "Great job on the presentation!",
            "I disagree with your opinion",
            "All [protected_group] should be deported"
        ]
        
        print("Testing classification on example texts:")
        for text in test_examples:
            result, conf, explanation = classify_text_with_models(text, tfidf, nb_model, lr_model)
            print(f"\nText: {text}")
            print(f"Classification: {result}")
            print(f"Confidence: {conf:.1%}")
            print(f"Explanation: {explanation}")
        
    except Exception as e:
        print(f"Error in evaluation and saving: {str(e)}")
        return

if __name__ == '__main__':
    main() 