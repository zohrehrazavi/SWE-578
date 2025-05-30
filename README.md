# Hate Speech Detection Web Application

This application combines machine learning and rule-based approaches to detect and classify text as hateful, offensive, or neutral content.

# Project: Hate Speech Detection System

## 🧠 Algorithm Overview

This system classifies social media text into three categories: **hateful**, **offensive**, or **neutral** using a hybrid rule-based and machine learning approach.

### 🚀 Libraries Used:
- **Scikit-learn (sklearn):** For TF-IDF feature extraction, model training (`MultinomialNB`, `LogisticRegression`), evaluation (`classification_report`, `confusion_matrix`).
- **Imbalanced-learn (SMOTE):** To handle class imbalance by oversampling minority classes.
- **NLTK:** For stopword removal and lemmatization during preprocessing.
- **TQDM:** To show progress bars for long-running preprocessing or training steps.
- **SciPy (softmax):** To normalize model probabilities.
- **Flask:** To serve a web interface and API endpoint for real-time classification.

## 🧹 Preprocessing Pipeline
Implemented with `nltk` and custom regex logic:
- Lowercase normalization
- Removal of URLs, mentions, hashtags
- Offensive word normalization (`f*ck` → `fuck`)
- Stopword removal (preserving negations like "not", "never")
- Lemmatization (handling plural forms, e.g., "immigrants" → "immigrant")
- Offensive term soft-mapping (e.g., `idiot` → `bad`)

### Protected Groups Coverage
The system recognizes various forms of protected groups including:
- Religious groups (Muslims, Jews, Christians, etc.)
- Ethnic and racial groups (Black, Asian, Hispanic, etc.)
- Gender identities (Women, Transgender, Non-binary)
- Sexual orientations (Gay, Lesbian, Queer)
- Immigration status (Immigrants, Refugees)
- Disabilities
- Minorities

All terms are lemmatized to catch both singular and plural forms.

## 🧪 Machine Learning Models

Two classifiers are trained and compared:
- **Multinomial Naive Bayes** (with prior tuning)
- **Logistic Regression** (with `class_weight='balanced'`, `C=0.3`, 2000 max iterations)

Feature extraction is done using `TfidfVectorizer` with a vocabulary limit of 10,000.

To combat class imbalance:
- **SMOTE** is applied on the training set.

## 🛡️ Hybrid Rule-Based + ML Classification Logic

Before using the models, the system performs **rule-based checks**:
- If protected group + hate indicators → classify as *hateful* directly with 90% confidence.
- Explicit phrases like "kill all", "death to" → directly *hateful*
- If rules don't trigger, fall back to model prediction
- If model confidence < 40%, label as *neutral*
- Final confidence is explained in output.

## 🌐 Flask Interface

Includes:
- **HTML Form Submission** for user testing
- **REST API** at `/classify` endpoint that accepts JSON input
- Renders classification result, confidence score, and explanation
- Error handling for missing/empty input

## 🧪 Unit Tests

Full suite of `unittest` cases test:
- Preprocessing accuracy
- Classification correctness (rule-based + ML)
- Flask endpoints
- Model consistency and edge cases (`None`, empty text, special characters)

## 🔍 Example Use

```json
POST /classify
{
  "text": "I hate all immigrants and they should die!"
}
→ {
  "classification": "hateful",
  "confidence": "90.0%",
  "explanation": "Rule-based classification with 90.0% confidence"
}
```

## Features

- Text classification into three categories: hateful, offensive, and neutral
- Combines ML models (Naive Bayes and Logistic Regression) with rule-based classification
- Provides confidence scores and explanations for classifications
- Web interface and API endpoints
- Comprehensive test coverage

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
```

2. Activate the virtual environment:
- On Windows:
```bash
venv\Scripts\activate
```
- On macOS/Linux:
```bash
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Training the Model

The model needs to be trained before running the web application. The training process:

1. Loads and preprocesses the labeled data
2. Performs text vectorization using TF-IDF
3. Applies SMOTE for class balancing
4. Trains two models:
   - Naive Bayes (with adjusted class priors)
   - Logistic Regression (with balanced class weights)

To train the model, run:
```bash
# Make sure you're in the project root directory
python3 src/main.py
```

Note: The command must be run from the project root directory, as the script expects to find the data directory and will save models in the correct location.

This will:
- Load the training data from `data/labeled_data.csv`
- Preprocess the texts
- Train the models
- Save the following files in the `models/` directory:
  - `naive_bayes.pkl`
  - `logistic_regression.pkl`
  - `tfidf_vectorizer.pkl`
  - `config.json`

The training process includes:
1. Data preprocessing (removing URLs, handling special characters, etc.)
2. Feature extraction using TF-IDF vectorization
3. Class balancing using SMOTE
4. Model training with optimized parameters
5. Evaluation on test set
6. Example classifications to verify performance

## Running the Web Application

After training the models, run the Flask application:

```bash
python3 app.py
```

The application will be available at:
- Web Interface: http://localhost:8081
- API Endpoint: http://localhost:8081/classify

### API Usage

Send POST requests to `/classify` with JSON data:

```json
{
    "text": "Your text to classify"
}
```

Response format:
```json
{
    "text": "Your text to classify",
    "classification": "neutral",
    "confidence": "85.0%",
    "explanation": "Model classification with 85.0% confidence"
}
```

## Testing

The application includes comprehensive test coverage:

```bash
python -m pytest tests/
```

Tests cover:
- Text preprocessing
- Classification accuracy
- Model consistency
- Edge cases
- API endpoints
- Rule-based classification logic

## Model Details

The classification system combines:

1. Rule-based Classification:
   - Detects explicit hate speech using predefined word sets
   - Identifies protected groups and hate speech indicators
   - Handles offensive language patterns

2. Machine Learning Models:
   - Naive Bayes with adjusted class priors
   - Logistic Regression with balanced class weights
   - TF-IDF vectorization for feature extraction

3. Confidence Scoring:
   - Combines model probabilities
   - Applies threshold-based decision making
   - Provides explanations for classifications

## Security Note

The application includes protection against:
- Empty or malformed inputs
- Special character manipulation
- Very long texts
- Non-English content

## Prerequisites

- Python 3.9 or higher
- pip (Python package installer)
- Virtual environment (recommended)

## Project Structure

```
.
├── app.py              # Flask application with classification endpoints
├── models/            # Directory for saved models
│   ├── naive_bayes.pkl
│   ├── logistic_regression.pkl
│   ├── tfidf_vectorizer.pkl
│   └── config.json
└── requirements.txt   # Project dependencies
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

## 🐳 Docker Deployment

### Using Docker Compose (Recommended)

1. Make sure you have Docker and Docker Compose installed
2. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

3. Build and start the containers:
```bash
docker-compose up --build
```

The application will be available at `http://localhost:8082`

### Using Docker Directly

1. Build the Docker image:
```bash
docker build -t hate-speech-detection .
```

2. Run the container:
```bash
docker run -p 8082:8082 -v $(pwd)/models:/app/models -v $(pwd)/data:/app/data hate-speech-detection
```

### Environment Variables

The following environment variables can be configured:

- `FLASK_ENV`: Set to 'production' or 'development' (default: production)
- `PORT`: Application port (default: 8082)

### Docker Volumes

The application uses two persistent volumes:
- `./models`: For storing trained ML models
- `./data`: For storing training and test data

### Health Checks

The Docker container includes health checks that:
- Monitor the application every 30 seconds
- Timeout after 10 seconds
- Retry up to 3 times before marking container as unhealthy

## 🚀 Deployment Best Practices

1. Always use the production environment in deployment
2. Ensure models are trained before deploying
3. Monitor container health status
4. Back up the models directory regularly
5. Use proper logging and monitoring in production