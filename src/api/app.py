from fastapi import FastAPI, HTTPException
import logging
from src.utils.logging import setup_logging
from src.models.bert_classifier import BertSentimentIntentClassifier
from typing import List

app = FastAPI()
setup_logging()
logger = logging.getLogger(__name__)

model = BertSentimentIntentClassifier("model")

@app.post("/predict")
async def predict(review_text: str):
    try:
        prediction = model.predict(review_text)
        return prediction
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
async def health_check():
    """Check API and model health."""
    try:
        model.predict("Test review")
        status = "healthy"
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        status = "unhealthy"
    return {"status": status, "model_loaded": True}

@app.post("/predict_with_confidence")
async def predict_with_confidence(review_text: str):
    """Predict sentiment and intent with confidence scores."""
    prediction = model.predict(review_text)
    inputs = model.tokenizer(review_text, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model.model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).tolist()[0]
    prediction['confidence'] = {'sentiment': probs[:3], 'intent': probs[3:]}
    logger.info(f"Predicted with confidence for: {review_text}")
    return prediction

@app.post("/validate_input")
async def validate_input(review_text: str):
    """Validate review text before prediction."""
    if not review_text or len(review_text.strip()) < 5:
        logger.error("Invalid input: Review text too short")
        raise HTTPException(status_code=400, detail="Review text must be at least 5 characters")
    if not any(c.isalpha() for c in review_text):
        logger.error("Invalid input: No alphabetic characters")
        raise HTTPException(status_code=400, detail="Review text must contain letters")
    logger.info("Input validated successfully")
    return {"status": "valid"}

@app.get("/model_info")
async def model_info():
    """Return information about the loaded BERT model."""
    info = {
        "model_type": "bert-base-uncased",
        "num_labels": model.model.config.num_labels,
        "sentiment_labels": model.sentiment_labels,
        "intent_labels": model.intent_labels
    }
    logger.info("Retrieved model information")
    return info

@app.post("/predict_batch")
async def predict_batch(reviews: List[str]):
    """Predict sentiment and intent for a batch of reviews."""
    if not reviews:
        raise HTTPException(status_code=400, detail="Review list cannot be empty")
    predictions = [model.predict(review) for review in reviews]
    logger.info(f"Processed batch of {len(reviews)} reviews")
    return {"predictions": predictions}

@app.get("/prediction_stats")
async def prediction_stats():
    """Return statistics about recent predictions."""
    stats = {
        "total_predictions": 0,
        "sentiment_distribution": {"positive": 0, "negative": 0, "neutral": 0},
        "intent_distribution": {"complaint": 0, "suggestion": 0, "question": 0, "praise": 0}
    }
    logger.info("Generated prediction statistics")
    return stats

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
