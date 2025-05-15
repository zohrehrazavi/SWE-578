import unittest
from app import app
import json

class TestApp(unittest.TestCase):
    def setUp(self):
        """Set up test client before each test."""
        self.app = app.test_client()
        self.app.testing = True

    def test_home_page(self):
        """Test that home page loads correctly."""
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Hate Speech Detection System', response.data)

    def test_classify_endpoint_json(self):
        """Test classification endpoint with JSON data."""
        test_data = {'text': 'This is a test message'}
        response = self.app.post('/classify',
                               data=json.dumps(test_data),
                               content_type='application/json')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('classification', data)
        self.assertIn('confidence', data)
        self.assertIn('explanation', data)

    def test_classify_endpoint_form(self):
        """Test classification endpoint with form data."""
        test_data = {'text': 'This is a test message'}
        response = self.app.post('/classify',
                               data=test_data)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Classification Result', response.data)

    def test_empty_text(self):
        """Test classification with empty text."""
        test_data = {'text': ''}
        response = self.app.post('/classify',
                               data=json.dumps(test_data),
                               content_type='application/json')
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('error', data)

    def test_missing_text(self):
        """Test classification with missing text field."""
        test_data = {}
        response = self.app.post('/classify',
                               data=json.dumps(test_data),
                               content_type='application/json')
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('error', data)

    def test_hateful_content(self):
        """Test classification of hateful content."""
        test_data = {'text': 'I hate all Muslims and they should die!'}
        response = self.app.post('/classify',
                               data=json.dumps(test_data),
                               content_type='application/json')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['classification'], 'hateful')

    def test_neutral_content(self):
        """Test classification of neutral content."""
        test_data = {'text': 'The weather is nice today'}
        response = self.app.post('/classify',
                               data=json.dumps(test_data),
                               content_type='application/json')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['classification'], 'neutral')

if __name__ == '__main__':
    unittest.main() 