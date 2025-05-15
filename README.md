# Hate Speech and Offensive Language Classification System

This project implements a machine learning-based system for classifying text into three categories: hateful, offensive, and neutral content. The system combines rule-based classification with machine learning models (Naive Bayes and Logistic Regression) to provide accurate and explainable classifications.

## Features

- Text classification into three categories: hateful, offensive, and neutral
- Combination of rule-based and machine learning approaches
- Detailed explanations for classifications
- REST API endpoint for classification
- Web interface for easy testing
- Comprehensive test suite
- Handling of edge cases and ambiguous content

## Prerequisites

- Python 3.9 or higher
- pip (Python package installer)
- Virtual environment (recommended)
- Training dataset (see Training Data section)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Create and activate a virtual environment (recommended):
```bash
# On macOS/Linux
python3 -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
.\venv\Scripts\activate
```

3. Install the required packages:
```bash
# On macOS/Linux
python3 -m pip install -r requirements.txt

# On Windows
python -m pip install -r requirements.txt
```

4. Download NLTK data (required for text processing):
```bash
# Start Python interpreter
python3
```
```python
>>> import nltk
>>> nltk.download('stopwords')
>>> exit()
```

## Project Structure

```
.
├── app.py              # Flask application with classification endpoints
├── src/
│   └── main.py        # Model training and evaluation script
├── models/            # Directory for saved models
│   ├── naive_bayes.pkl
│   ├── logistic_regression.pkl
│   ├── tfidf_vectorizer.pkl
│   └── config.json
├── templates/         # HTML templates
│   └── index.html    # Web interface
├── test_app.py       # Test suite
└── requirements.txt  # Project dependencies
```

## Training Data

The system requires a training dataset in CSV format with the following columns:
- `text`: The input text to classify
- `label`: The classification label (0 for hateful, 1 for offensive, 2 for neutral)

Place your training data file in the `data/` directory before running the training script.

Example training data format:
```csv
text,label
"I hate all people from that country!",0
"You're such an idiot",1
"Have a nice day everyone",2
```

## Training the Models

1. Create the data directory and add your training data:
```bash
mkdir -p data
# Add your training.csv file to the data/ directory
```

2. Run the training script:
```bash
# On macOS/Linux
python3 src/main.py

# On Windows
python src/main.py
```

This will:
- Preprocess the training data
- Train both Naive Bayes and Logistic Regression models
- Apply SMOTE for handling imbalanced classes
- Save the trained models in the `models/` directory
- Generate a classification report with model performance metrics

## Running the Application

1. Start the Flask application:
```bash
# On macOS/Linux
python3 app.py

# On Windows
python app.py
```

The application will:
- Load the trained models
- Start a web server on `http://localhost:8080`
- Provide both a web interface and REST API endpoint

2. Access the web interface:
- Open your browser and navigate to `http://localhost:8080`
- Enter text in the input field and click "Classify" to get results

3. Use the REST API:
```bash
curl -X POST http://localhost:8080/classify \
     -H "Content-Type: application/json" \
     -d '{"tweet": "Your text here"}'
```

## Running Tests

The project includes a comprehensive test suite covering various aspects of the system:

1. Run all tests:
```bash
# On macOS/Linux
python3 -m pytest test_app.py -v

# On Windows
python -m pytest test_app.py -v
```

2. Run specific test categories:
```bash
# Run only preprocessing tests
python3 -m pytest test_app.py -v -k "test_preprocess"

# Run only classification tests
python3 -m pytest test_app.py -v -k "test_classification"

# Run only explanation tests
python3 -m pytest test_app.py -v -k "test_explanation"
```

## Troubleshooting

Common issues and solutions:

1. **Command not found: python**
   - Use `python3` instead of `python` on macOS/Linux
   - Ensure Python is added to your PATH on Windows

2. **ModuleNotFoundError**
   - Make sure you've activated the virtual environment
   - Verify all dependencies are installed: `python3 -m pip list`

3. **Model loading errors**
   - Ensure you've run the training script before starting the app
   - Check that all model files exist in the `models/` directory

4. **NLTK data errors**
   - Run the NLTK download command mentioned in the installation section

## API Documentation

### POST /classify

Classifies the provided text into one of three categories.

**Request Format:**
```json
{
    "tweet": "Text to classify"
}
```

**Response Format:**
```json
{
    "classification": "hateful|offensive|neutral",
    "confidence": 0.95,
    "explanation": "Detailed explanation of the classification"
}
```

**Status Codes:**
- 200: Successful classification
- 400: Invalid request (missing or null tweet field)
- 500: Server error

## Classification Logic

The system uses a combination of approaches for classification:

1. Rule-based Classification:
   - Checks for protected groups and hate speech indicators
   - Identifies personal attacks and offensive language
   - Handles special cases and common patterns

2. Machine Learning Models:
   - Naive Bayes for baseline classification
   - Logistic Regression for improved accuracy
   - TF-IDF vectorization for feature extraction

3. Confidence Thresholds:
   - High confidence threshold (0.8) for definitive classifications
   - Lower threshold (0.3) for general classification
   - Special handling for hate speech detection

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run the test suite
5. Submit a pull request

## License

[Your License Here]

## Contact

[Your Contact Information]
