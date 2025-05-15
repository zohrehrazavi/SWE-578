from flask import Flask, request, jsonify, render_template
import pickle
import os
from src.main import classify_text_with_models

app = Flask(__name__)

# Load the models
def load_models():
    try:
        with open('models/naive_bayes.pkl', 'rb') as f:
            nb_model = pickle.load(f)
        with open('models/logistic_regression.pkl', 'rb') as f:
            lr_model = pickle.load(f)
        with open('models/tfidf_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        return vectorizer, nb_model, lr_model
    except FileNotFoundError:
        print("Error: Model files not found. Please run 'python src/main.py' first to train the models.")
        return None, None, None

vectorizer, nb_model, lr_model = load_models()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    # Check if models are loaded
    if None in (vectorizer, nb_model, lr_model):
        return jsonify({
            'error': 'Models not loaded. Please train models first using python src/main.py'
        }), 500

    # Get the text from the request
    if request.is_json:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({
                'error': 'No text provided. Please send a JSON with a text field.'
            }), 400
        text = data['text']
    else:
        text = request.form.get('text')
        if not text:
            return jsonify({
                'error': 'No text provided.'
            }), 400

    if not text.strip():
        return jsonify({
            'error': 'Empty text provided.'
        }), 400

    # Classify the text
    try:
        result, confidence, explanation = classify_text_with_models(
            text,
            vectorizer,
            nb_model,
            lr_model
        )

        response_data = {
            'text': text,
            'classification': result,
            'confidence': f'{confidence:.1%}',
            'explanation': explanation
        }

        # If it's a form submission, render the template with results
        if not request.is_json:
            return render_template('index.html', result=response_data)
        
        # If it's an API call, return JSON
        return jsonify(response_data)

    except Exception as e:
        return jsonify({
            'error': f'Classification failed: {str(e)}'
        }), 500

if __name__ == '__main__':
    # Create required directories if they don't exist
    os.makedirs('static', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    
    app.run(host='0.0.0.0', port=8080, debug=True) 