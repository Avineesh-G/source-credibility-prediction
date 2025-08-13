from flask import Flask, request, render_template
from src.data_preprocessing import DataPreprocessor
from src.factcheck_api import GoogleFactChecker
import pickle

# ----------------- CONFIG -----------------
FACTCHECK_API_KEY = "AIzaSyBw77nSIozzzoZDtweNEbNS1SnEccVIk_Q"
MODEL_PATH = "models/credibility_model.pkl"  # Path to saved model+vectorizer
# -------------------------------------------

# Flask app
app = Flask(__name__)

# Preprocessing
preprocessor = DataPreprocessor()

# Google Fact Checker
factchecker = GoogleFactChecker(api_key=FACTCHECK_API_KEY)

# Load trained model + vectorizer from pickle
with open(MODEL_PATH, "rb") as f:
    saved_data = pickle.load(f)
    model = saved_data["model"]         # Trained ML model
    vectorizer = saved_data["vectorizer"]  # TF-IDF vectorizer

@app.route("/", methods=["GET", "POST"])
def home():
    fact_results = None
    model_result = None
    user_text = None

    if request.method == "POST":
        user_text = request.form.get("user_text")
        if user_text and user_text.strip():
            # 1️⃣ Preprocess text
            cleaned = preprocessor.clean_text(user_text)
            features = vectorizer.transform([cleaned])

            # 2️⃣ ML prediction
            pred = model.predict(features)[0]
            model_result = "Credible ✅" if pred == 1 else "Not Credible ❌"

            # 3️⃣ Google Fact Check API
            fact_results = factchecker.search_claim(user_text)

    return render_template(
        "index.html",
        model_result=model_result,
        fact_results=fact_results,
        user_text=user_text
    )

if __name__ == "__main__":
    app.run(debug=True)
