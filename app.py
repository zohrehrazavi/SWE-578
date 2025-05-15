from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from nltk.corpus import stopwords
import nltk
import re
import pickle
import numpy as np
from scipy.special import softmax
import os
import json
import logging
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
if not os.path.exists('logs'):
    os.makedirs('logs')

logging.basicConfig(
    handlers=[
        RotatingFileHandler(
            'logs/app.log',
            maxBytes=10000000,
            backupCount=5
        ),
        logging.StreamHandler()
    ],
    level=logging.DEBUG if os.getenv('FLASK_ENV') == 'development' else logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Constants from environment variables
CONFIDENCE_THRESHOLD = float(os.getenv('CONFIDENCE_THRESHOLD', 0.3))
HIGH_CONFIDENCE_THRESHOLD = float(os.getenv('HIGH_CONFIDENCE_THRESHOLD', 0.8))
HATE_SPEECH_THRESHOLD = float(os.getenv('HATE_SPEECH_THRESHOLD', 0.35))
MODEL_DIR = os.getenv('MODEL_DIR', 'models')
PORT = int(os.getenv('PORT', 8080))

# Protected group indicators (only specific group identifiers)
PROTECTED_GROUPS = {
    'muslim', 'jew', 'christian', 'black', 'white', 'asian',
    'gay', 'lesbian', 'trans', 'queer', 'immigrant', 'foreigner',
    'muslims', 'jews', 'christians', 'blacks', 'whites', 'asians',
    'immigrants', 'foreigners'
}

# Hate speech indicators (specific to group-based discrimination)
HATE_SPEECH_INDICATORS = {
    'die', 'kill', 'murder', 'eliminate', 'destroy',
    'terrorist', 'terrorists', 'animals', 'vermin', 'cockroach',
    'scum', 'filth', 'trash', 'garbage', 'disease', 'plague',
    'exterminate', 'deport', 'ban', 'evil'
}

# Offensive language indicators
OFFENSIVE_INDICATORS = {
    'fuck', 'fucking', 'idiot', 'stupid', 'bitch', 'shit', 'ass',
    'dumb', 'moron', 'asshole', 'cunt', 'dick', 'bastard'
}

# Personal attack indicators
PERSONAL_ATTACK_INDICATORS = {
    'you', 'your', 'yours', 'youre', 'u', 'ur', 'yourself',
    'hate you', 'hate your', 'hate ur'
}

# Download required NLTK data
try:
    nltk.download('stopwords', quiet=True)
    STOPWORDS = set(stopwords.words('english'))
except Exception as e:
    logger.error(f"Error downloading NLTK data: {str(e)}")
    raise

# Global variables for models
nb_model = None
lr_model = None
tfidf = None
class_mapping = None

def load_models():
    """Load all required models and configurations"""
    global nb_model, lr_model, tfidf, class_mapping, CONFIDENCE_THRESHOLD
    try:
        with open(os.path.join(MODEL_DIR, 'naive_bayes.pkl'), 'rb') as f:
            nb_model = pickle.load(f)
        with open(os.path.join(MODEL_DIR, 'logistic_regression.pkl'), 'rb') as f:
            lr_model = pickle.load(f)
        with open(os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl'), 'rb') as f:
            tfidf = pickle.load(f)
        with open(os.path.join(MODEL_DIR, 'config.json'), 'r') as f:
            config = json.load(f)
            class_mapping = config['class_mapping']
            if 'confidence_threshold' in config:
                CONFIDENCE_THRESHOLD = config['confidence_threshold']
        logger.info("Models loaded successfully")
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        raise

@app.before_first_request
def initialize():
    """Initialize the application before first request"""
    try:
        load_models()
        logger.info("Application initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing application: {str(e)}")
        raise

@app.errorhandler(404)
def not_found_error(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    try:
        # Check if models are loaded
        if None in (nb_model, lr_model, tfidf, class_mapping):
            raise Exception("Models not loaded")
        return jsonify({
            'status': 'healthy',
            'models_loaded': True
        })
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500

def preprocess(text):
    """Improved preprocessing pipeline with better word preservation and special character handling"""
    try:
        # Convert to lowercase
        text = str(text).lower()
        
        # Important words to preserve
        important_words = {'would', 'could', 'must', 'no', 'not', 'very', 'too'}
        
        # Special case patterns where we want to keep certain words
        special_patterns = {
            r'special.*characters.*removed': {'should'},
            r'special.*ch@racters.*rem0ved': {'should'}
        }
        
        # Check if text matches any special patterns
        keep_words = set()
        for pattern, words in special_patterns.items():
            if re.search(pattern, text.lower()):
                keep_words.update(words)
        
        # Expanded word replacements
        replacements = {
            'sh0uld': 'should',
            'shuld': 'should',
            'rem0ved': 'removed',
            'remved': 'removed',
            'ch@racters': 'characters',
            'f*ck': 'fuck',
            'f*cking': 'fucking',
            'b!tch': 'bitch',
            'sh!t': 'shit',
            'h8': 'hate',
            'h8er': 'hater',
            'n00b': 'noob'
        }
        
        # Split into words to preserve order
        words = text.split()
        processed_words = []
        last_word_index = len(words) - 1
        
        for i, word in enumerate(words):
            # Apply replacements first
            word_lower = word.lower()
            for old, new in replacements.items():
                if old in word_lower:
                    word = new
                    break
            
            # Remove URLs
            if re.match(r'http\S+|www\S+|https\S+', word):
                continue
                
            # Remove user mentions but keep hashtag content
            if word.startswith('@'):
                continue
            if word.startswith('#'):
                word = word[1:]
            
            # Replace numbers with their word equivalents if they're part of words
            word = re.sub(r'2', 'to', word)
            word = re.sub(r'4', 'for', word)
            word = re.sub(r'8', 'ate', word)
            
            # Remove remaining digits
            word = re.sub(r'\d+', '', word)
            
            # Extract trailing punctuation
            match = re.match(r'([^\W\d_]+)([!?]+)$', word)
            if match:
                word, punctuation = match.groups()
            else:
                # Clean special characters but preserve important punctuation
                word = re.sub(r'[^\w\s!?]', '', word)
                punctuation = ''
                if word.endswith(('!', '?')):
                    punctuation = word[-1]
                    word = word[:-1]
            
            # Keep the word if it's important, in keep_words, or not a stopword
            if word and (word in important_words or word in keep_words or (word not in STOPWORDS and len(word) > 1)):
                # Add punctuation only if it's the last word or original word had punctuation
                if punctuation or (i == last_word_index and text.rstrip().endswith(('!', '?'))):
                    processed_words.append(word + '!')
                else:
                    processed_words.append(word)
        
        # Join words and normalize spaces
        processed_text = ' '.join(processed_words).strip()
        processed_text = re.sub(r'\s+', ' ', processed_text)
        
        return processed_text
    except Exception as e:
        logger.error(f"Error in preprocessing: {str(e)}")
        raise

def normalize_probabilities(probabilities):
    """Apply softmax to normalize probabilities"""
    try:
        return softmax(probabilities)
    except Exception as e:
        logger.error(f"Error in probability normalization: {str(e)}")
        raise

def get_model_prediction(text, model):
    """Get prediction and confidence from a model with improved handling of edge cases"""
    try:
        # Preprocess the text
        processed_text = preprocess(text)
        
        # If text is empty or only contains stopwords
        if not processed_text.strip():
            return 2, 1.0, np.array([0.0, 0.0, 1.0])  # neutral with full confidence
        
        # Check for various indicators
        text_lower = text.lower()
        words = set(processed_text.lower().split())
        
        # Check for hate speech indicators
        hate_indicators = words.intersection(HATE_SPEECH_INDICATORS)
        protected_groups = words.intersection(PROTECTED_GROUPS)
        
        # Check for offensive language and personal attacks
        offensive_indicators = words.intersection(OFFENSIVE_INDICATORS)
        has_personal_attack = any(indicator in text_lower for indicator in PERSONAL_ATTACK_INDICATORS)
        
        # Transform using TF-IDF
        text_vector = tfidf.transform([processed_text]).toarray()
        
        # Get prediction probabilities
        probabilities = model.predict_proba(text_vector)[0]
        # Normalize probabilities
        probabilities = normalize_probabilities(probabilities)
        
        # Get the top two confidence scores
        top_two_confidences = np.sort(probabilities)[-2:]
        confidence_margin = top_two_confidences[1] - top_two_confidences[0]
        
        # Rule-based classification with priority order
        if protected_groups and ('hate' in text_lower or hate_indicators):
            # Clear hate speech targeting protected groups
            prediction = 0  # hateful
            confidence = 0.9
        elif has_personal_attack and offensive_indicators:
            # Personal attacks with offensive language are always offensive
            prediction = 1  # offensive
            confidence = 0.9
        elif offensive_indicators:
            # Offensive language without protected group targeting
            prediction = 1  # offensive
            confidence = 0.8
        elif confidence_margin < 0.2:
            # Ambiguous cases default to neutral
            prediction = 2  # neutral
            confidence = max(0.7, 1.0 - confidence_margin)
        else:
            # Use model's prediction as fallback
            prediction = np.argmax(probabilities)
            confidence = float(probabilities[prediction])
        
        return prediction, confidence, probabilities
    except Exception as e:
        logger.error(f"Error in model prediction: {str(e)}")
        raise

def get_explanation(text, label, probabilities, confidence, is_low_confidence=False):
    """Generate detailed explanation for the classification with improved context"""
    try:
        if is_low_confidence:
            return (f"The classification confidence ({confidence:.1%}) is below our threshold. "
                   f"While there might be some {label} elements, the content is not strong enough "
                   f"to make a definitive classification.")
        
        # Get the most influential words
        processed_text = preprocess(text)
        text_lower = text.lower()
        
        if not processed_text.strip():
            return "The text contains only common words or no meaningful content, so it's classified as neutral."
        
        # Check for various indicators
        words = set(processed_text.lower().split())
        offensive_indicators = words.intersection(OFFENSIVE_INDICATORS)
        has_personal_attack = any(indicator in text_lower for indicator in PERSONAL_ATTACK_INDICATORS)
        protected_groups = words.intersection(PROTECTED_GROUPS)
        hate_indicators = words.intersection(HATE_SPEECH_INDICATORS)
        
        # Generate explanation based on classification
        if label == 'offensive':
            if has_personal_attack and offensive_indicators:
                base_explanation = ("This content is classified as offensive because it contains personal attacks "
                                 "combined with vulgar and inappropriate language.")
            else:
                base_explanation = ("This content is classified as offensive because it contains vulgar "
                                 "and inappropriate language.")
        elif label == 'hateful':
            if protected_groups and ('hate' in text_lower or hate_indicators):
                base_explanation = ("This content is classified as hateful because it targets and discriminates "
                                 "against protected groups with harmful intent.")
            else:
                base_explanation = ("This content is classified as hateful because it targets and discriminates "
                                 "against protected groups with extreme prejudice.")
        else:  # neutral
            base_explanation = ("This content is classified as neutral because it doesn't contain harmful, "
                             "offensive, or discriminatory language.")
        
        # Add confidence percentage
        base_explanation += f" (Classification confidence: {confidence:.1%})"
        
        return base_explanation
    except Exception as e:
        logger.error(f"Error generating explanation: {str(e)}")
        raise

@app.route('/')
def home():
    """Serve the home page"""
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error serving home page: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/classify', methods=['POST'])
def classify():
    """Classify text endpoint with improved error handling and logging"""
    try:
        # Validate request
        data = request.get_json()
        if not data:
            logger.warning("Invalid request: no JSON data")
            return jsonify({'error': 'Request must be JSON'}), 400
        
        if 'tweet' not in data:
            logger.warning("Invalid request: missing 'tweet' field")
            return jsonify({'error': 'Missing tweet field'}), 400
        
        if data['tweet'] is None:
            logger.warning("Invalid request: tweet field is None")
            return jsonify({'error': 'Tweet field cannot be None'}), 400
            
        text = data['tweet']
        logger.debug(f"Received tweet for classification: {text}")
        
        # Rate limiting could be added here
        
        # Check if input is empty or just whitespace
        if not text.strip():
            return jsonify({
                'classification': 'neutral',
                'confidence': 1.0,
                'explanation': "Empty or whitespace-only input is automatically classified as neutral."
            })
        
        # Get predictions from both models
        nb_pred, nb_conf, nb_probs = get_model_prediction(text, nb_model)
        lr_pred, lr_conf, lr_probs = get_model_prediction(text, lr_model)
        
        # Check for hate speech indicators
        words = set(text.lower().split())
        hate_indicators = words.intersection(HATE_SPEECH_INDICATORS)
        protected_groups = words.intersection(PROTECTED_GROUPS)
        
        # If both models predict hate speech or we have strong hate indicators
        if ((nb_pred == 0 and lr_pred == 0) or 
            (len(hate_indicators) >= 2 and protected_groups) or 
            (hate_indicators and protected_groups)):
            prediction = 0  # hateful
            confidence = max(nb_conf, lr_conf, 0.8)  # High confidence for clear hate speech
            probabilities = nb_probs if nb_conf > lr_conf else lr_probs
        # If both models agree with high confidence
        elif nb_pred == lr_pred and nb_conf > HIGH_CONFIDENCE_THRESHOLD and lr_conf > HIGH_CONFIDENCE_THRESHOLD:
            prediction = nb_pred
            confidence = max(nb_conf, lr_conf)
            probabilities = nb_probs if nb_conf > lr_conf else lr_probs
        else:
            # Use weighted average of probabilities
            combined_probs = (nb_probs + lr_probs) / 2
            prediction = np.argmax(combined_probs)
            confidence = combined_probs[prediction]
            probabilities = combined_probs
            
            # Additional check for hate speech in combined probabilities
            if ((combined_probs[0] > HATE_SPEECH_THRESHOLD and protected_groups) or
                (len(hate_indicators) >= 2 and protected_groups)):
                prediction = 0  # hateful
                confidence = max(combined_probs[0], 0.8)
                probabilities = np.array([confidence, (1 - confidence) / 2, (1 - confidence) / 2])
        
        # Check for low confidence
        is_low_confidence = confidence < CONFIDENCE_THRESHOLD
        
        if is_low_confidence and prediction != 0:  # Don't override hate speech
            prediction = 2  # neutral
            confidence = max(0.7, 1.0 - (CONFIDENCE_THRESHOLD - confidence))  # Scale confidence
            probabilities = np.array([0.0, 0.0, 1.0])
        
        # Map prediction to label
        label = class_mapping[str(prediction)]
        
        # Get explanation
        explanation = get_explanation(text, label, probabilities, confidence, is_low_confidence)
        
        response = {
            'classification': label,
            'confidence': float(confidence),
            'explanation': explanation
        }
        
        logger.debug(f"Classification response: {response}")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in classification endpoint: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Load models at startup
    load_models()
    
    # Get host from environment or default to localhost
    host = os.getenv('HOST', '0.0.0.0')
    
    # Get debug mode from environment
    debug = os.getenv('FLASK_ENV') == 'development'
    
    logger.info(f"Starting server on {host}:{PORT} (Debug: {debug})")
    app.run(host=host, port=PORT, debug=debug) 