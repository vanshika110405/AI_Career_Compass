from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
import pandas as pd

# Load data
df = pd.read_csv('data/ai_career_compass_dataset.csv')

# Example preprocessing (you’ll do more as needed)
X = df['required_skills'] + " " + df['domain_industries']  # combined text features
y = df['role']  # target to predict

# Vectorize
tfidf = TfidfVectorizer()
X_tfidf = tfidf.fit_transform(X)

# Encode output labels
le = LabelEncoder()
y_enc = le.fit_transform(y)

# Train model
model = RandomForestClassifier()
model.fit(X_tfidf, y_enc)

# Save model
import joblib
joblib.dump((model, tfidf, le), 'model/career_predictor.pkl')
