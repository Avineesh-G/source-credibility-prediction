# train_model.py

import nltk
nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('stopwords')

import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from src.data_preprocessing import DataPreprocessor
from src.traditional_ml_models import TraditionalMLModels

df = pd.read_csv("data/news_dataset.csv")  # Ensure your dataset exists with 'text' and 'label'

preprocessor = DataPreprocessor()
df['cleaned'] = df['text'].apply(preprocessor.clean_text)

vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['cleaned'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_wrapper = TraditionalMLModels(model_type="random_forest")
model_wrapper.train(X_train, y_train)

save_obj = {
    "model": model_wrapper.model,
    "vectorizer": vectorizer
}

with open("models/credibility_model.pkl", "wb") as f:
    pickle.dump(save_obj, f)

print("Model and vectorizer saved to models/credibility_model.pkl")
