# ReviewInsightClassifier

**ReviewInsightClassifier** is a production-ready NLP pipeline designed to classify sentiment (positive, negative, neutral) and intent (complaint, suggestion, question, praise) in customer reviews. Built with a BERT-based model, it offers robust text processing, scalable API deployment via FastAPI, and MLOps practices including data versioning with DVC and CI/CD with GitHub Actions.

## Features
- **Sentiment & Intent Classification**: Uses `bert-base-uncased` to predict sentiment and intent from review text.
- **Data Ingestion**: Supports local CSV or S3 data sources with DVC versioning, including validation, metadata extraction, and rating normalization.
- **Text Preprocessing**: Cleans text by removing punctuation, numbers, and stop words, with stemming, contraction expansion, and n-gram extraction.
- **Model Training**: Includes learning rate scheduling, data augmentation, batch processing, weight decay, and checkpointing.
- **API Endpoints**: FastAPI endpoints for single/batch predictions, confidence scores, input validation, model info, and health checks.
- **MLOps**: CI/CD pipeline, structured logging with rotation, and data/model versioning.
- **Scalability**: Optimized for deployment with Docker and async API calls.

## Installation
1. **Clone the Repository**:
   ```bash
   git clone <your-repo-url>
   cd ReviewInsightClassifier
   ```
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Configure Data Source**:
   - Update `config/data_config.yaml` to specify `local` or `s3` source and S3 bucket details.
   - Place review data (CSV) in `data/` or configure S3 access.
4. **Set Up Environment**:
   - Ensure Python 3.9 is installed.
   - Install DVC for data versioning: `pip install dvc`.

## Usage
### Training the Model
Run the training pipeline with:
```bash
python src/pipeline/train.py
```
- Configures data ingestion, preprocessing, and BERT model training.
- Saves trained model to `model/` (configurable in `config/train_config.yaml`).

### Running the API
Start the FastAPI server:
```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```
Available endpoints:
- `POST /predict`: Predict sentiment and intent for a single review.
- `POST /predict_with_confidence`: Get predictions with confidence scores.
- `POST /predict_batch`: Process multiple reviews.
- `POST /validate_input`: Validate review text.
- `GET /health`: Check API and model status.
- `GET /model_info`: Get BERT model details.
- `GET /prediction_stats`: View prediction statistics (placeholder).

### Example API Request
```bash
curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{"review_text": "Great product, fast delivery!"}'
```
**Response**:
```json
{"sentiment": "positive", "intent": "praise"}
```

## Data Format
- **Input**: CSV with columns:
  - `review_text` (string): The review content.
  - `rating` (integer): Numerical rating (e.g., 1-5).
- **Output**: JSON with:
  - `sentiment`: One of `positive`, `negative`, `neutral`.
  - `intent`: One of `complaint`, `suggestion`, `question`, `praise`.
- **Example Input**:
  ```csv
  review_text,rating
  "Great product!",5
  "Slow delivery.",2
  ```
- **Validation**: Reviews must have non-empty `review_text` and valid `rating`.

## Project Structure
```
ReviewInsightClassifier/
├── src/
│   ├── data/ingest.py          # Data loading and validation
│   ├── preprocess/text_cleaner.py  # Text preprocessing
│   ├── models/bert_classifier.py  # BERT model training/prediction
│   ├── pipeline/train.py       # Training pipeline
│   ├── api/app.py             # FastAPI application
│   ├── utils/logging.py        # Logging utilities
├── config/                    # Configuration files
├── tests/                     # Unit tests
├── logs/                      # Application logs
├── .github/workflows/ci.yml   # CI/CD pipeline
├── Dockerfile                 # Docker configuration
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
```

## MLOps Practices
- **Data Versioning**: DVC tracks review datasets and model versions.
- **CI/CD**: GitHub Actions runs tests and linting on push/pull requests.
- **Logging**: Structured logs saved to `logs/app.log` with rotation.
- **Monitoring**: Health checks and prediction statistics via API.
- **Deployment**: Dockerized for consistent environments.

## Troubleshooting
- **API Errors**: Check `logs/app.log` for detailed errors.
- **Data Issues**: Ensure CSV has `review_text` and `rating` columns.
- **Model Loading**: Verify `model/` contains trained BERT weights.
- **Dependency Issues**: Use Python 3.9 and install all requirements.
- **Performance**: Adjust batch size in `config/train_config.yaml` for memory constraints.

## Contributing
1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/your-feature`.
3. Commit changes: `git commit -m "Add your feature"`.
4. Push to the branch: `git push origin feature/your-feature`.
5. Open a pull request.

## License
This project is licensed under the MIT License.

For support, contact the project maintainers or open an issue on GitHub.
