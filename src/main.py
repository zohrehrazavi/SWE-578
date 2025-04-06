import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix

# 1. Load dataset
df = pd.read_csv("../data/labeled_data.csv")  # Replace with your actual filename

# Optional: Map labels to binary or categorical
# Example: Hate=0, Offensive=1, Neither=2
df = df[df['class'].isin([0, 1])]  # Filter only hate/offensive for binary
df['label'] = df['class']  # Assuming column name is 'class'

# 2. Preprocess
df['text'] = df['tweet'].str.lower().str.replace(r"http\S+|[^a-zA-Z\s]", "", regex=True)

# 3. Feature extraction
tfidf = TfidfVectorizer(max_features=1000)
X = tfidf.fit_transform(df['text']).toarray()
y = df['label']

# 4. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train model
clf = DecisionTreeClassifier(max_depth=10, random_state=42)
clf.fit(X_train, y_train)

# 6. Evaluation
y_pred = clf.predict(X_test)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
