"""
Hate Speech Detection System
---------------------------
A hybrid ML and rule-based system for classifying text as hateful, offensive, or neutral.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from scipy.special import softmax
import nltk
import re
import pickle
import json
import os
from tqdm import tqdm
import time

# Initialize NLTK components
print("Downloading NLTK data...")
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
STOPWORDS = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Classification Constants
CONFIDENCE_THRESHOLD = 0.5  # Increased from 0.45
HIGH_CONFIDENCE_THRESHOLD = 0.8
SEMANTIC_SIMILARITY_THRESHOLD = 0.7

# Class Mapping
class_mapping = {
    0: 'hateful',
    1: 'offensive',
    2: 'neutral'
}

# Word Sets for Rule-Based Classification
PROTECTED_GROUPS = {
    # Religious groups
    'muslim', 'jew', 'christian', 'buddhist', 'hindu', 'sikh',
    'muslims', 'jews', 'christians', 'buddhists', 'hindus', 'sikhs',
    
    # Racial/Ethnic groups
    'black', 'white', 'asian', 'hispanic', 'arab', 'native',
    'blacks', 'whites', 'asians', 'hispanics', 'arabs', 'natives',
    'chinese', 'japanese', 'korean', 'mexican', 'indian', 'african',
    'latinos', 'latinas', 'latino', 'latina',
    
    # Immigration status
    'immigrant', 'refugee', 'foreigner', 'asylum',
    'immigrants', 'refugees', 'foreigners', 'asylees',
    
    # LGBTQ+
    'gay', 'lesbian', 'trans', 'queer', 'lgbt', 'lgbtq',
    'transgender', 'nonbinary', 'bisexual', 'homosexual',
    
    # Gender
    'woman', 'man', 'girl', 'boy', 'female', 'male',
    'women', 'men', 'girls', 'boys', 'females', 'males',
    
    # Age groups
    'old', 'young', 'elderly', 'teen', 'boomer', 'millennial',
    'senior', 'youth', 'kid', 'child', 'adult',
    'seniors', 'youths', 'kids', 'children', 'adults',
    'boomers', 'millennials', 'generation',
    
    # Disability
    'disabled', 'handicapped', 'impaired', 'disability',
    'disabilities', 'handicap', 'disorder', 'illness',
    
    # Political
    'liberal', 'conservative', 'democrat', 'republican',
    'leftist', 'rightist', 'socialist', 'communist',
    'liberals', 'conservatives', 'democrats', 'republicans',
    
    # General protected terms
    'minority', 'minorities', 'community', 'group',
    'people', 'person', 'human', 'citizen', 'individual',
    'these', 'those', 'they', 'them', 'their',
    
    # Additional gender terms
    'gender', 'sex', 'sexuality', 'identity', 'orientation',
    'genders', 'sexes', 'sexualities', 'identities', 'orientations',
    
    # Additional disability terms
    'disabled', 'handicapped', 'impaired', 'disability', 'differently abled',
    'disabilities', 'handicap', 'disorder', 'illness', 'condition',
    'mental', 'physical', 'cognitive', 'developmental', 'neurological',
    'special needs', 'wheelchair', 'blind', 'deaf', 'autistic',
    
    # Additional workplace terms
    'worker', 'employee', 'staff', 'colleague', 'coworker',
    'workers', 'employees', 'staffers', 'colleagues', 'coworkers',
    
    # Additional cultural terms
    'culture', 'tradition', 'heritage', 'background', 'community',
    'cultures', 'traditions', 'backgrounds', 'communities', 'societies'
}

HATE_SPEECH_INDICATORS = {
    'die', 'kill', 'murder', 'eliminate', 'destroy', 'hate',
    'terrorist', 'terrorists', 'animals', 'vermin', 'cockroach',
    'scum', 'filth', 'trash', 'garbage', 'disease', 'plague',
    'exterminate', 'deport', 'ban', 'evil', 'death', 'dead',
    'burden', 'remove', 'go away', 'get out', 'leave', 'dont belong',
    'inferior', 'subhuman', 'worthless', 'parasites', 'criminals'
}

OFFENSIVE_WORDS = {
    'terrible', 'stupid', 'idiot', 'ugly', 'dumb', 'fool',
    'fuck', 'fucking', 'shit', 'damn', 'ass', 'asshole',
    'crap', 'suck', 'sucks', 'bad', 'worst', 'hate'
}

# Disclaimer Prefixes that Often Mask Hate Speech
SUSPICIOUS_DISCLAIMERS = {
    "i'm not racist but",
    "not to be racist but",
    "i'm not sexist but",
    "i'm not homophobic but",
    "i'm not transphobic but",
    "i'm not prejudiced but",
    "i'm not discriminating but",
    "i'm not biased but",
    "no offense but",
    "don't take this wrong but",
    "with all due respect but",
    "some of my best friends are",
    "i have friends who are",
    "i know people who are",
    "i work with people who are",
    "i have nothing against",
    "i don't hate them but",
    "i respect them but",
    "they're fine people but",
    "i understand their culture but",
    "i support their rights but"
}

# Complex Hate Speech Pattern Templates
COMPLEX_PATTERNS = [
    # Direct hate speech with protected groups
    r"\b(all|these|those)?\s?({groups})[^\n]*?(should|must|need to|have to|going to)?\s?(die|be killed|be eliminated|go away|be removed|be exterminated|be destroyed|be wiped out)\b",
    
    # Dehumanizing comparisons (expanded)
    r"\b(these|those|they|the|all)?\s*({groups})?\s*(are|is|like|basically|just)?\s*(cockroaches|vermin|scum|trash|animals|parasites|disease|plague|rats|pests|garbage|filth|dirt|burden|problem|threat|cancer|infestation)\b",
    
    # Rights/existence denial (expanded)
    r"\b({groups})\s+(don't|do not|shouldn't|should not|can't|cannot|won't|will not)\s+(deserve|have|get|be allowed|be permitted|exist|live|stay|remain|survive|work|participate)\s*(here|there|anywhere|in|at)?\s*(society|country|nation|community|workplace|world|life)?\b",
    
    # General hate patterns
    r"\b(should just die|are worthless|are trash|shouldn't exist|don't belong|go back|send them back|get out)\b",
    
    # Workplace discrimination (expanded)
    r"\b({groups})\s*(don't|do not|shouldn't|should not|can't|cannot|won't|will not|have no place|aren't suited|aren't fit)\s*(work|belong|be|exist|participate|succeed|thrive|lead|manage|handle|understand)\s*(in|at|near|around)?\s*(the|this|our|any)?\s*(workplace|office|job|company|business|industry|profession|position|role|career|field|sector)\b",
    
    # Existence denial (expanded)
    r"\b({groups})\s*(shouldn't|should not|can't|cannot|must not|mustn't|don't deserve to|have no right to|aren't meant to|weren't meant to)\s*(exist|be|live|survive|be here|be alive|be accepted|be recognized|be respected|be treated|be considered|be seen as|identify as)\b",
    
    # Resource competition (expanded)
    r"\b({groups})\s*(are|is|keep|always)?\s*(stealing|taking|getting|receiving|demanding|draining|wasting|abusing|exploiting)\s*(our|all the|the|precious|limited|valuable)?\s*(jobs|benefits|resources|opportunities|money|aid|support|welfare|healthcare|education|services)\b",
    
    # Cultural threat (expanded)
    r"\b(our|the|traditional|real|true)\s*(culture|values|way of life|society|country|nation|identity|heritage|traditions)\s*(needs?|must|should|has to|is being|are being)\s*(be protected|be defended|be saved|be preserved|be maintained)\s*(from|against|due to)\s*({groups})\b",
    
    # Subtle workplace bias
    r"\b({groups})\s*(just|simply|naturally|typically|generally|usually|always|often)?\s*(aren't|are not|can't|cannot|don't|do not)\s*(qualified|capable|suitable|fit|right|appropriate|professional|competent|reliable|trustworthy|productive)\s*(enough|for|in)?\s*(the|this|our|any)?\s*(workplace|job|position|role|industry|field)\b",
    
    # Capability denial
    r"\b({groups})\s*(lack|don't have|cannot handle|can't manage|aren't capable of|are incapable of|will never understand|cannot grasp|don't understand)\s*(the|proper|necessary|required|basic|fundamental)?\s*(skills|abilities|qualities|competencies|capabilities|requirements|qualifications|intelligence|capacity|aptitude)\b",
    
    # Disclaimer-prefixed hate (new)
    r"\b(i'?m not (?:racist|sexist|homophobic|transphobic|prejudiced|biased) but|some of my best friends are|i have (?:friends|colleagues) who are|with all due respect but|no offense but|don't take this wrong but)\s*({groups})\s*(?:are|should|must|need to|have to|always|never|don't|can't|shouldn't)\b",
    
    # False acceptance (new)
    r"\b(i (?:respect|accept|support|understand|tolerate) {groups} but|{groups} are fine (?:people|individuals) but|i have nothing against {groups} but)\s*(?:they|we|society|everyone)\s*(?:should|must|need to|have to|can't|shouldn't)\b",
    
    # Subtle exclusion (expanded)
    r"\b({groups})\s*(just|simply|naturally|obviously|clearly)?\s*(don't|do not|cannot|can't|won't|will never)\s*(understand|handle|manage|cope with|deal with|fit into|adapt to|belong in|work in|succeed in|thrive in)\s*(our|the|this|any)?\s*(culture|society|workplace|environment|system|standards|values|norms|community|country)\b",
    
    # Resource resentment (new)
    r"\b(why|how come|since when)\s*(do|should|must|can)\s*({groups})\s*(get|receive|have|deserve|demand|expect|take|claim)\s*(special|preferential|extra|additional|more|better)?\s*(treatment|rights|benefits|privileges|advantages|opportunities|consideration|accommodation|support|help)\b",
    
    # Conditional acceptance (new)
    r"\b({groups})\s*(can|could|should|would|might|may)\s*(be|exist|work|live|stay)\s*(here|there|anywhere)\s*(only|just|if|as long as|provided that|assuming)\s*(they|we|society|everyone)?\s*(don't|do not|won't|will not|never|avoid|stop)\b"
]

# Context Disclaimers (Expanded)
CONTEXT_DISCLAIMERS = {
    'in minecraft', 'in game', 'in the game', 'metaphorically',
    'joking', 'sarcastically', 'just kidding', '/s', 'video game',
    'roleplay', 'rp', 'quoted', 'quoting', 'parody', 'satire',
    'movie quote', 'game quote', 'song lyrics', 'not serious',
    'in jest', 'for fun', 'playing around'
}

def preprocess(text):
    """
    Preprocessing Pipeline
    ---------------------
    - Lowercase normalization
    - URL/mention removal
    - Offensive word normalization
    - Stopword removal (preserving negations)
    - Lemmatization
    - Offensive term mapping
    """
    # Convert to lowercase
    text = str(text).lower()
    
    # Remove URLs and social media elements
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+', '', text)
    text = re.sub(r'\#(\w+)', r'\1', text)
    
    # Normalize offensive words
    text = text.replace('f*ck', 'fuck')
    text = text.replace('f**k', 'fuck')
    text = text.replace('sh*t', 'shit')
    text = text.replace('b*tch', 'bitch')
    text = text.replace('a**', 'ass')
    
    # Remove punctuation except sentiment indicators
    text = re.sub(r'[^\w\s!?]', '', text)
    
    # Tokenize
    tokens = text.split()
    
    # Remove stopwords but keep negations and important context words
    important_words = {'no', 'not', 'nor', 'none', 'never', 'neither', 'all', 'every', 'should', 'must'}
    tokens = [word for word in tokens if word not in STOPWORDS or word in important_words]
    
    # Apply lemmatization
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    # Map offensive terms
    offensive_replacements = {
        'terrible': 'bad',
        'stupid': 'bad',
        'idiot': 'bad',
        'ugly': 'bad',
        'dumb': 'bad',
        'fool': 'bad'
    }
    tokens = [offensive_replacements.get(word, word) for word in tokens]
    
    return ' '.join(tokens)

def normalize_probabilities(probabilities):
    """Apply softmax to normalize model probabilities."""
    return softmax(probabilities)

def check_hate_speech_patterns(text, protected_groups):
    """
    Enhanced pattern matching for hate speech detection.
    Returns (is_hate_speech, confidence, pattern_type) tuple.
    """
    text = text.lower()
    
    # Format protected groups for regex
    groups_pattern = '|'.join(protected_groups)
    
    # Check each pattern
    for pattern in COMPLEX_PATTERNS:
        # Replace {groups} placeholder with actual groups pattern
        pattern = pattern.format(groups=groups_pattern)
        if re.search(pattern, text, re.IGNORECASE):
            # Determine confidence based on pattern strength
            if "should" in text or "must" in text or "kill" in text or "die" in text:
                return True, 0.9, "explicit_hate"
            elif any(word in text for word in ["cockroach", "vermin", "trash", "burden", "cancer", "threat"]):
                return True, 0.85, "dehumanizing"
            elif any(phrase in text for phrase in ["don't deserve", "shouldn't exist", "don't belong", "taking our", "protect from"]):
                return True, 0.85, "rights_denial"
            elif any(word in text for word in ["workplace", "job", "industry", "profession"]):
                return True, 0.85, "workplace_exclusion"
            else:
                return True, 0.8, "subtle_hate"
    
    return False, None, None

def check_context_disclaimers(text):
    """Enhanced context disclaimer detection."""
    text = text.lower()
    
    # Direct matches
    if any(disclaimer in text for disclaimer in CONTEXT_DISCLAIMERS):
        return True, "explicit_disclaimer"
    
    # Check for quoted text
    if text.count('"') >= 2 or text.count("'") >= 2:
        return True, "quoted_text"
    
    # Check for common prefixes that indicate non-literal meaning
    context_prefixes = ['imagine', 'pretend', 'suppose', 'what if']
    if any(text.startswith(prefix) for prefix in context_prefixes):
        return True, "hypothetical"
    
    return False, None

def check_suspicious_disclaimers(text):
    """Check for suspicious disclaimers that often precede hate speech."""
    text_lower = text.lower()
    
    for disclaimer in SUSPICIOUS_DISCLAIMERS:
        if disclaimer in text_lower:
            # Look for hate speech indicators after the disclaimer
            disclaimer_pos = text_lower.find(disclaimer) + len(disclaimer)
            text_after = text_lower[disclaimer_pos:]
            
            # Check if the text after the disclaimer contains hate speech indicators
            words_after = set(text_after.split())
            if (words_after.intersection(PROTECTED_GROUPS) and 
                (words_after.intersection(HATE_SPEECH_INDICATORS) or 
                 any(word in text_after for word in ["should", "must", "always", "never", "don't", "can't"]))):
                return True, "disclaimer_hate"
    
    return False, None

def classify_text(text, words):
    """
    Enhanced rule-based classification component.
    Returns (prediction, confidence, pattern_type) or (None, None, None) if no rules match.
    """
    text_lower = text.lower()
    
    # Check for suspicious disclaimers first
    is_disclaimer_hate, disclaimer_type = check_suspicious_disclaimers(text_lower)
    if is_disclaimer_hate:
        return 0, 0.85, disclaimer_type
    
    # Check for context disclaimers
    has_disclaimer, disclaimer_type = check_context_disclaimers(text_lower)
    if has_disclaimer:
        return 2, 0.7, disclaimer_type
    
    # Check complex hate speech patterns
    is_hate_pattern, pattern_conf, pattern_type = check_hate_speech_patterns(text_lower, PROTECTED_GROUPS)
    if is_hate_pattern:
        return 0, pattern_conf, pattern_type
    
    has_protected_group = bool(words.intersection(PROTECTED_GROUPS))
    has_hate_indicator = bool(words.intersection(HATE_SPEECH_INDICATORS))
    has_offensive = bool(words.intersection(OFFENSIVE_WORDS))
    
    # Enhanced hate speech detection
    if has_protected_group:
        if has_hate_indicator:
            return 0, 0.9, "protected_hate"  # Hateful with high confidence
        elif "should" in words or "must" in words:
            return 0, 0.85, "protected_action"  # Likely hateful with good confidence
        elif any(word in words for word in ["workplace", "job", "work", "industry", "profession"]):
            return 0, 0.85, "workplace_discrimination"  # Workplace discrimination
        elif any(word in words for word in ["exist", "real", "normal", "natural", "right"]):
            return 0, 0.85, "existence_denial"  # Existence denial
    
    # Check for explicit death threats or elimination
    death_phrases = ["death to", "kill all", "die all", "eliminate all", "remove all"]
    if any(phrase in text_lower for phrase in death_phrases):
        return 0, 0.9, "death_threat"  # Explicit death threats are hateful
    
    # Check for offensive content
    if has_offensive:
        if has_protected_group:
            return 0, 0.8, "offensive_protected"  # Offensive + protected group = likely hate speech
        return 1, 0.8, "offensive_only"  # Just offensive with good confidence
    
    return None, None, None  # Let ML models decide

def classify_text_with_models(text, tfidf_vectorizer, nb_model, lr_model):
    """
    Enhanced hybrid classification combining rule-based and ML approaches.
    Returns (classification, confidence, explanation, metadata).
    """
    metadata = {
        "is_rule_based": False,
        "flagged_context": None,
        "semantic_alert": False,
        "pattern_type": None
    }
    
    if not text or not str(text).strip():
        return "neutral", 1.0, "Empty or meaningless content", metadata
    
    text_lower = str(text).lower()
    
    # Check for context disclaimers
    has_disclaimer, disclaimer_type = check_context_disclaimers(text_lower)
    if has_disclaimer:
        metadata["flagged_context"] = disclaimer_type
        return "neutral", 0.7, f"Context disclaimer detected: {disclaimer_type}", metadata
    
    # Process the text
    processed = preprocess(text)
    if not processed.strip():
        return "neutral", 1.0, "Empty or meaningless content", metadata
    
    # Rule-based classification
    words = set(processed.lower().split())
    rule_pred, rule_conf, pattern_type = classify_text(processed, words)
    
    if rule_pred is not None:
        metadata["is_rule_based"] = True
        metadata["pattern_type"] = pattern_type
        return class_mapping[rule_pred], rule_conf, f"Rule-based classification ({pattern_type}) with {rule_conf:.1%} confidence", metadata
    
    # ML classification
    vector = tfidf_vectorizer.transform([processed]).toarray()
    nb_probs = normalize_probabilities(nb_model.predict_proba(vector)[0])
    lr_probs = normalize_probabilities(lr_model.predict_proba(vector)[0])
    
    nb_pred = nb_model.predict(vector)[0]
    lr_pred = lr_model.predict(vector)[0]
    
    nb_conf = nb_probs[nb_pred]
    lr_conf = lr_probs[lr_pred]
    
    # Use model with higher confidence
    prediction = nb_pred if nb_conf > lr_conf else lr_pred
    confidence = max(nb_conf, lr_conf)
    
    # Confidence adjustments
    if confidence < CONFIDENCE_THRESHOLD:
        return "neutral", confidence, f"Low confidence ({confidence:.1%})", metadata
    
    # Boost confidence for hate speech predictions with strong indicators
    if prediction == 0:
        has_protected = bool(words.intersection(PROTECTED_GROUPS))
        has_hate = bool(words.intersection(HATE_SPEECH_INDICATORS))
        
        if has_protected and has_hate and confidence >= 0.5:
            confidence = max(confidence, 0.75)
            metadata["semantic_alert"] = True
    
    return class_mapping[prediction], confidence, f"Model classification with {confidence:.1%} confidence", metadata

def main():
    """
    Main training pipeline:
    1. Load and preprocess data
    2. Extract features
    3. Apply SMOTE
    4. Train models
    5. Evaluate and save
    """
    print("\n=== Starting Training Process ===\n")
    
    # Load data
    print("Step 1/7: Loading data...")
    try:
        data = pd.read_csv('data/labeled_data.csv')
        if 'tweet' in data.columns:
            data = data.rename(columns={'tweet': 'text'})
        print(f"✓ Loaded {len(data)} samples")
        
        print("\nClass distribution before balancing:")
        print(data['class'].value_counts())
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return
    
    # Preprocess
    print("\nStep 2/7: Preprocessing data...")
    try:
        texts = data['text'].tolist()
        processed_texts = []
        for text in tqdm(texts, desc="Preprocessing texts", unit="text"):
            processed = preprocess(text)
            processed_texts.append(processed)
        
        # Split data
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
    
    # Extract features
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
    
    # Apply SMOTE
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
    
    # Train models
    print("\nStep 6/7: Training models...")
    try:
        # Train Naive Bayes
        print("\nTraining Naive Bayes...")
        nb_model = MultinomialNB(class_prior=[0.2, 0.4, 0.4])
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
    
    # Evaluate and save
    print("\nStep 7/7: Evaluating and saving models...")
    try:
        # Evaluate
        y_pred_nb = nb_model.predict(X_test_tfidf)
        y_pred_lr = lr_model.predict(X_test_tfidf)
        
        print("\nNaive Bayes Results:")
        print("Classification Report:")
        print(classification_report(y_test, y_pred_nb, target_names=['Hateful', 'Offensive', 'Neutral']))
        
        print("\nLogistic Regression Results:")
        print("Classification Report:")
        print(classification_report(y_test, y_pred_lr, target_names=['Hateful', 'Offensive', 'Neutral']))
        
        # Save models
        print("\nSaving models and configuration...")
        if not os.path.exists('models'):
            os.makedirs('models')
        
        with tqdm(total=4, desc="Saving files") as pbar:
            with open('models/naive_bayes.pkl', 'wb') as f:
                pickle.dump(nb_model, f)
                pbar.update(1)
            
            with open('models/logistic_regression.pkl', 'wb') as f:
                pickle.dump(lr_model, f)
                pbar.update(1)
            
            with open('models/tfidf_vectorizer.pkl', 'wb') as f:
                pickle.dump(tfidf, f)
                pbar.update(1)
            
            config = {
                "class_mapping": class_mapping,
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
            # Neutral Examples
            "I love everyone and wish them the best!",
            "The weather is nice today",
            "This is a neutral statement about technology",
            "Great job on the presentation!",
            "I disagree with your opinion",
            "The movie was released in 2024",
            "Let's discuss this topic respectfully",
            "I prefer tea over coffee",
            
            # Offensive Examples (Non-hate speech)
            "You're a f*cking idiot!",
            "This movie was terrible",
            "You're stupid and ugly",
            "The service at this restaurant was awful",
            "Go to hell you moron",
            "This is complete garbage",
            "You're absolutely worthless",
            "What a dumb idea",
            
            # Hateful Examples (Protected Groups)
            "I hate all Muslims and they should die!",
            "Kill all immigrants",
            "Death to all Jews",
            "All [protected_group] should be deported",
            "These people are cockroaches and should be eliminated",
            "All mexicans are criminals and should be killed",
            "Gay people don't deserve to live",
            "Black people are inferior and should be eliminated",
            
            # Edge Cases
            "@user I really enjoyed your post about #AI",
            "!@#$%^&*()",
            "",
            "   ",
            "a" * 1000,  # Very long text
            "死亡",  # Non-English text
            
            # Ambiguous Cases
            "These people should leave our country",  # No explicit protected group
            "They are ruining everything",  # Ambiguous reference
            "We need to get rid of them all",  # Ambiguous threat
            "I can't stand these kinds of people",  # Ambiguous dislike
            
            # Mixed Content
            "The weather is nice but I hate all immigrants",  # Mixed neutral and hate
            "You're an idiot and Muslims should die",  # Mixed offensive and hate
            "Great presentation but you're stupid",  # Mixed neutral and offensive
            
            # Context-Dependent
            "They should all die (in the video game)",
            "Kill them all (in minecraft)",
            "You're dead to me (metaphorically)",
            
            # Subtle Hate Speech
            "We need to protect our culture from them",
            "They're taking all our jobs",
            "These people don't belong here",
            "Send them back where they came from",
            
            # Offensive Without Slurs
            "Your intelligence is questionable",
            "Your contribution is worthless",
            "Nobody likes you",
            "You should quit your job",
            
            # Complex Cases
            "While I respect everyone, immigrants are destroying our country",
            "I'm not racist but all [protected_group] are criminals",
            "Some of my best friends are gay but they shouldn't have rights",
            "I don't hate them, I just think they should leave",
            
            # Multiple Protected Groups
            "Muslims and Jews should all be eliminated",
            "I hate both immigrants and blacks",
            "Gay people and trans people don't deserve rights",
            
            # Sarcasm/Irony (challenging cases)
            "Oh yeah, you're *really* smart",
            "What a *brilliant* contribution",
            "You must be the *best* at everything",
            
            # Religious/Political
            "All Christians are evil and should die",
            "Buddhists don't deserve to live here",
            "Conservative people should be eliminated",
            "Liberal people should all be killed",
            
            # Gender-based
            "Women don't belong in the workplace",
            "Men are all trash and should die",
            "Trans people shouldn't exist",
            
            # Disability-related
            "Disabled people are a burden and should be removed",
            "People with disabilities don't deserve jobs",
            "Mental illness is just an excuse",
            
            # Age-related
            "Old people should just die already",
            "Young people are destroying everything",
            "Boomers should be eliminated",
            
            # Mixed languages
            "You're stupid pendejo",
            "Hate all les étrangers",
            "Die you バカ",
        ]
        
        print("Testing classification on example texts:")
        print("\nFormat: Classification (Confidence) - Explanation")
        print("-" * 50)
        for text in test_examples:
            result, conf, explanation, metadata = classify_text_with_models(text, tfidf, nb_model, lr_model)
            print(f"\nText: {text}")
            print(f"Result: {result.upper()} ({conf:.1%})")
            print(f"Explanation: {explanation}")
            print("-" * 50)
        
    except Exception as e:
        print(f"Error in evaluation and saving: {str(e)}")
        return

if __name__ == '__main__':
    main() 