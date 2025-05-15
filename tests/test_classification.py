import unittest
from src.main import classify_text_with_models
import pickle
import os

class TestClassification(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Load models before running tests."""
        try:
            with open('models/naive_bayes.pkl', 'rb') as f:
                cls.nb_model = pickle.load(f)
            with open('models/logistic_regression.pkl', 'rb') as f:
                cls.lr_model = pickle.load(f)
            with open('models/tfidf_vectorizer.pkl', 'rb') as f:
                cls.vectorizer = pickle.load(f)
        except FileNotFoundError:
            raise unittest.SkipTest("Model files not found. Run training first.")

    def test_neutral_classification(self):
        """Test classification of neutral text."""
        text = "The weather is beautiful today"
        result, confidence, explanation, metadata = classify_text_with_models(
            text,
            self.vectorizer,
            self.nb_model,
            self.lr_model
        )
        self.assertEqual(result, 'neutral')
        self.assertIsInstance(confidence, float)
        self.assertGreater(confidence, 0)
        self.assertLess(confidence, 1)
        self.assertIsInstance(explanation, str)
        self.assertIsInstance(metadata, dict)

    def test_hateful_classification(self):
        """Test classification of hateful text."""
        text = "death to all [protected_group]"  # Using explicit phrase that triggers rule-based classification
        result, confidence, explanation, metadata = classify_text_with_models(
            text,
            self.vectorizer,
            self.nb_model,
            self.lr_model
        )
        self.assertEqual(result, 'hateful')
        self.assertIsInstance(confidence, float)
        self.assertGreater(confidence, 0)
        self.assertLess(confidence, 1)
        self.assertIsInstance(explanation, str)
        self.assertIsInstance(metadata, dict)

    def test_offensive_classification(self):
        """Test classification of offensive text."""
        text = "You're a complete idiot"
        result, confidence, explanation, metadata = classify_text_with_models(
            text,
            self.vectorizer,
            self.nb_model,
            self.lr_model
        )
        self.assertEqual(result, 'offensive')
        self.assertIsInstance(confidence, float)
        self.assertGreater(confidence, 0)
        self.assertLess(confidence, 1)
        self.assertIsInstance(explanation, str)
        self.assertIsInstance(metadata, dict)

    def test_empty_text(self):
        """Test classification with empty text."""
        text = ""
        result, confidence, explanation, metadata = classify_text_with_models(
            text,
            self.vectorizer,
            self.nb_model,
            self.lr_model
        )
        self.assertEqual(result, 'neutral')
        self.assertEqual(confidence, 1.0)
        self.assertEqual(explanation, "Empty or meaningless content")
        self.assertIsInstance(metadata, dict)

    def test_none_text(self):
        """Test classification with None text."""
        text = None
        result, confidence, explanation, metadata = classify_text_with_models(
            text,
            self.vectorizer,
            self.nb_model,
            self.lr_model
        )
        self.assertEqual(result, 'neutral')
        self.assertEqual(confidence, 1.0)
        self.assertEqual(explanation, "Empty or meaningless content")
        self.assertIsInstance(metadata, dict)

    def test_special_characters(self):
        """Test classification with special characters."""
        text = "!@#$%^&*()"
        result, confidence, explanation, metadata = classify_text_with_models(
            text,
            self.vectorizer,
            self.nb_model,
            self.lr_model
        )
        self.assertEqual(result, 'neutral')
        self.assertIsInstance(confidence, float)
        self.assertGreater(confidence, 0)
        self.assertLess(confidence, 1.1)  # Allow for floating point imprecision
        self.assertIn("confidence", explanation.lower())
        self.assertIsInstance(metadata, dict)

if __name__ == '__main__':
    unittest.main() 