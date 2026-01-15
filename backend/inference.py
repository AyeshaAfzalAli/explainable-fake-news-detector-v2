import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
# Decide device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "/Users/ayeshaafzalali/Desktop/explainable-fake-news-detector/ML_MODELS/bert_finetuned"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(device)
model.eval()

def predict_text(text: str):
    """
    Takes raw text and returns predicted label and confidence.
    """

    # Tokenize input text
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    # Move tensors to same device as model
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Disable gradient calculation (inference only)
    with torch.no_grad():
        outputs = model(**inputs)

    # Convert logits to probabilities
    probs = torch.softmax(outputs.logits, dim=1)

    # Get predicted class
    pred_class = torch.argmax(probs, dim=1).item()
    confidence = probs[0][pred_class].item()

    label = "REAL" if pred_class == 1 else "FAKE"

    return {
        "label": label,
        "confidence": round(confidence, 4)
    }

if __name__ == "__main__":
    result = predict_text("Government announces new economic reforms")
    print(result)



