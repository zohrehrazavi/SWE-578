import unittest
import sys
import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

# Add the src directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.main import preprocess, classify_text_with_models, normalize_probabilities, classify_text

class TestHateSpeechDetection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that will be used for all tests"""
        # Create a small test dataset with more explicit examples
        cls.test_texts = [
            "I love everyone!",  # neutral
            "You are an idiot",  # offensive
            "Kill all immigrants",  # hateful
            "The weather is nice",  # neutral
            "Muslims should die",  # hateful
            "This movie sucks",  # offensive
            "Death to all Jews",  # hateful
            "You're stupid",  # offensive
            "Have a great day",  # neutral
            "All immigrants are vermin",  # hateful
        ]
        cls.test_labels = [2, 1, 0, 2, 0, 1, 0, 1, 2, 0]  # 0: hateful, 1: offensive, 2: neutral
        
        # Train a small model for testing
        # Vectorize
        cls.vectorizer = TfidfVectorizer(max_features=1000)
        X = cls.vectorizer.fit_transform(cls.test_texts)
        
        # Train models
        cls.nb_model = MultinomialNB(class_prior=[0.2, 0.4, 0.4])  # Adjusted priors
        cls.lr_model = LogisticRegression(max_iter=1000, class_weight='balanced')
        cls.nb_model.fit(X, cls.test_labels)
        cls.lr_model.fit(X, cls.test_labels)

    def test_preprocessing(self):
        """Test text preprocessing functionality"""
        test_cases = [
            ("Hello World!", "hello world!"),  # Keeps exclamation marks
            ("@user how are you?", "you?"),  # Removes stopwords
            ("http://test.com visit me", "visit"),  # Removes stopwords
            ("f*ck this sh*t", "fuck shit"),  # Normalizes censored words
            ("I am NOT happy", "not happy"),  # Preserves negation
            ("This is a #hashtag", "hashtag")  # Removes stopwords, keeps content
        ]
        
        for input_text, expected in test_cases:
            processed = preprocess(input_text)
            self.assertEqual(processed.strip(), expected.strip(),
                           f"Preprocessing failed for '{input_text}'")

    def test_probability_normalization(self):
        """Test probability normalization function"""
        test_probs = np.array([1.0, 2.0, 3.0])
        normalized = normalize_probabilities(test_probs)
        
        self.assertTrue(np.allclose(np.sum(normalized), 1.0),
                       "Probabilities should sum to 1")
        self.assertTrue(all(0 <= p <= 1 for p in normalized),
                       "All probabilities should be between 0 and 1")

    def test_rule_based_classification(self):
        """Test rule-based classification logic"""
        test_cases = [
            ("Muslims should die", 0, 0.9),  # Should detect as hateful
            ("You're an idiot", 1, 0.8),  # Should detect as offensive
            ("The weather is nice", None, None),  # Should defer to model
            ("Kill all immigrants", 0, 0.9),  # Should detect as hateful
            ("This is terrible", 1, 0.8),  # Should detect as offensive
        ]
        
        for text, expected_class, expected_conf in test_cases:
            words = set(preprocess(text).split())
            pred_class, pred_conf = classify_text(text, words)
            self.assertEqual(pred_class, expected_class,
                           f"Rule-based classification failed for '{text}'")
            if expected_conf is not None:
                self.assertEqual(pred_conf, expected_conf,
                               f"Confidence score mismatch for '{text}'")

    def test_end_to_end_classification(self):
        """Test the complete classification pipeline"""
        test_cases = [
            {
                "text": "I love everyone!",
                "expected_class": "neutral",
                "min_confidence": 0.3
            },
            {
                "text": "Death to all Muslims!",  # More explicit hate speech
                "expected_class": "hateful",
                "min_confidence": 0.8
            },
            {
                "text": "You're stupid and ugly",  # Changed to a clearer offensive example
                "expected_class": "offensive",
                "min_confidence": 0.7
            },
            {
                "text": "The weather is nice today",
                "expected_class": "neutral",
                "min_confidence": 0.3
            }
        ]
        
        for case in test_cases:
            result, confidence, _ = classify_text_with_models(
                case["text"],
                self.vectorizer,
                self.nb_model,
                self.lr_model
            )
            
            self.assertEqual(
                result,
                case["expected_class"],
                f"Classification failed for '{case['text']}'"
            )
            self.assertGreaterEqual(
                confidence,
                case["min_confidence"],
                f"Confidence too low for '{case['text']}'"
            )

    def test_model_consistency(self):
        """Test if model predictions are consistent"""
        text = "This is a test message"
        
        # Get multiple predictions
        results = []
        for _ in range(5):
            result, conf, _ = classify_text_with_models(
                text,
                self.vectorizer,
                self.nb_model,
                self.lr_model
            )
            results.append((result, conf))
        
        # Check if all predictions are the same
        self.assertEqual(
            len(set(r[0] for r in results)),
            1,
            "Model predictions should be consistent for the same input"
        )
        
        # Check if confidence scores are consistent
        self.assertEqual(
            len(set(r[1] for r in results)),
            1,
            "Confidence scores should be consistent for the same input"
        )

    def test_edge_cases(self):
        """Test edge cases and potential error conditions"""
        edge_cases = [
            "",  # Empty string
            " ",  # Just whitespace
            "!@#$%^&*()",  # Only special characters
            "a" * 1000,  # Very long text
            "短文",  # Non-English text
            None,  # None input
        ]
        
        for text in edge_cases:
            try:
                result, conf, explanation = classify_text_with_models(
                    text,
                    self.vectorizer,
                    self.nb_model,
                    self.lr_model
                )
                # Should handle all cases without raising exceptions
                self.assertIsInstance(result, str)
                self.assertIsInstance(conf, float)
                self.assertIsInstance(explanation, str)
            except Exception as e:
                self.fail(f"Failed to handle edge case '{text}': {str(e)}")

if __name__ == '__main__':
    unittest.main() 