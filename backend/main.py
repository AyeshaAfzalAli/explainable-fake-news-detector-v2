from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from pathlib import Path
app = FastAPI(title="Fake News Detection API")


BASE_DIR = Path(__file__).resolve().parent.parent

model = joblib.load(
    BASE_DIR / "data" / "logreg_tfidf_model.pkl"
)

vectorizer = joblib.load(
    BASE_DIR / "data" / "tfidf_vectorizer.pkl"
)

class NewsRequest(BaseModel):
    text: str

@app.post("/predict")
def predict_news(request: NewsRequest):
    # Convert text to vector
    text_vector = vectorizer.transform([request.text])

    # Predict label
    prediction = model.predict(text_vector)[0]

    # Predict confidence
    confidence = model.predict_proba(text_vector).max()

    label = "FAKE" if prediction == 1 else "REAL"

    return {
        "prediction": label,
        "confidence": round(confidence, 3)
    }



