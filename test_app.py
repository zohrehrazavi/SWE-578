import unittest
from app import app, preprocess, get_model_prediction, get_explanation, CONFIDENCE_THRESHOLD
import json
import numpy as np

class TestTweetClassifier(unittest.TestCase):
    def setUp(self):
        # Set up the Flask test client
        self.app = app.test_client()
        self.app.testing = True
    
    def test_preprocess(self):
        # Test text preprocessing
        test_cases = [
            {
                "input": "Hello World! This is a TEST tweet with HTTP://example.com",
                "expected": "hello world! test tweet"
            },
            {
                "input": "Special ch@racters sh0uld be rem0ved!!!",
                "expected": "special characters should removed!"
            },
            {
                "input": "Multiple    spaces   should    be   normalized",
                "expected": "multiple spaces normalized"
            },
            {
                "input": "",
                "expected": ""
            },
            {
                "input": "   Leading and trailing spaces   ",
                "expected": "leading trailing spaces"
            }
        ]
        
        for test_case in test_cases:
            result = preprocess(test_case["input"])
            self.assertEqual(result, test_case["expected"])
    
    def test_home_page(self):
        # Test if home page loads correctly
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
    
    def test_classification_endpoint(self):
        # Test the classification endpoint with different types of tweets
        test_cases = [
            {
                "tweet": "I love everyone and wish them the best!",
                "expected_class": "neutral",
                "expected_status": 200
            },
            {
                "tweet": "I hate all Muslims and they should die!",  # Strong hate speech
                "expected_class": "hateful",
                "expected_status": 200
            },
            {
                "tweet": "",  # Empty tweet
                "expected_class": "neutral",
                "expected_status": 200
            },
            {
                "tweet": "You're a f*cking idiot and I hate you!",  # Strong offensive language
                "expected_class": "offensive",
                "expected_status": 200
            }
        ]
        
        for test_case in test_cases:
            response = self.app.post('/classify',
                                   data=json.dumps({"tweet": test_case["tweet"]}),
                                   content_type='application/json')
            
            self.assertEqual(response.status_code, test_case["expected_status"])
            
            if response.status_code == 200:
                data = json.loads(response.data)
                self.assertIn('classification', data)
                self.assertIn('confidence', data)
                self.assertIn('explanation', data)
                self.assertIn(data['classification'], ['hateful', 'offensive', 'neutral'])
                
                # For empty tweets, verify neutral classification with high confidence
                if not test_case["tweet"].strip():
                    self.assertEqual(data['classification'], 'neutral')
                    self.assertGreaterEqual(data['confidence'], 0.7)
    
    def test_confidence_threshold(self):
        # Test that low confidence predictions default to neutral
        response = self.app.post('/classify',
                               data=json.dumps({"tweet": "a b c d e"}),  # Meaningless text
                               content_type='application/json')
        
        data = json.loads(response.data)
        self.assertEqual(data['classification'], 'neutral')
        self.assertGreaterEqual(data['confidence'], 0.7)
    
    def test_explanation_generation(self):
        # Test explanation generation for different cases
        test_cases = [
            {
                "tweet": "I hate all Muslims and they should die! They are ruining everything!",  # Strong hate speech
                "expected_type": "hateful",
                "should_contain": ["targets", "discriminates", "protected groups"]
            },
            {
                "tweet": "You're a f*cking idiot! I hate you so much!",  # Strong offensive language
                "expected_type": "offensive",
                "should_contain": ["vulgar", "inappropriate"]
            },
            {
                "tweet": "Have a nice day everyone! The weather is beautiful!",  # Clearly neutral
                "expected_type": "neutral",
                "should_contain": ["doesn't contain harmful"]
            }
        ]
        
        for test_case in test_cases:
            response = self.app.post('/classify',
                                   data=json.dumps({"tweet": test_case["tweet"]}),
                                   content_type='application/json')
            data = json.loads(response.data)
            
            self.assertEqual(data['classification'], test_case['expected_type'])
            for phrase in test_case['should_contain']:
                self.assertIn(phrase.lower(), data['explanation'].lower())
    
    def test_invalid_requests(self):
        # Test various invalid request formats
        test_cases = [
            {
                "data": {"invalid_key": "some text"},
                "expected_status": 400
            },
            {
                "data": {},
                "expected_status": 400
            },
            {
                "data": {"tweet": None},
                "expected_status": 400
            }
        ]
        
        for test_case in test_cases:
            response = self.app.post('/classify',
                                   data=json.dumps(test_case["data"]),
                                   content_type='application/json')
            self.assertEqual(response.status_code, test_case["expected_status"])
    
    def test_model_prediction_consistency(self):
        # Test that model predictions are consistent
        tweet = "This is a test tweet"
        response1 = self.app.post('/classify',
                                data=json.dumps({"tweet": tweet}),
                                content_type='application/json')
        response2 = self.app.post('/classify',
                                data=json.dumps({"tweet": tweet}),
                                content_type='application/json')
        
        data1 = json.loads(response1.data)
        data2 = json.loads(response2.data)
        
        self.assertEqual(data1['classification'], data2['classification'])
        self.assertEqual(data1['confidence'], data2['confidence'])

if __name__ == '__main__':
    unittest.main() 