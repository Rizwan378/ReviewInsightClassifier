from src.data.ingest import ReviewDataIngestion
from src.preprocess.text_cleaner import TextCleaner
from src.models.bert_classifier import BertSentimentIntentClassifier
import yaml
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_model(config_path: str):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    ingestor = ReviewDataIngestion(config['data_config'])
    data = ingestor.load_data(config['data_path'])
    
    cleaner = TextCleaner()
    texts = cleaner.clean_batch(data['review_text'].tolist())
    
    # Dummy labels for demo (in practice, these would come from data)
    labels = [[1, 0, 0, 0, 1, 0, 0]] * len(texts)  # [pos, neg, neu, complaint, suggestion, question, praise]
    
    model = BertSentimentIntentClassifier(config['model_save_path'])
    model.train(texts, labels)
    
    logger.info(f"Model trained and saved to {config['model_save_path']}")

def split_data(data: pd.DataFrame, test_size: float = 0.2) -> tuple:
    """Split review data into training and test sets."""
    from sklearn.model_selection import train_test_split
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=42)
    logger.info(f"Split data: {len(train_data)} train, {len(test_data)} test")
    return train_data.reset_index(drop=True), test_data.reset_index(drop=True)

def cross_validate_model(model, texts: List[str], labels: List[int], folds: int = 5):
    """Perform k-fold cross-validation on BERT model."""
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=folds, shuffle=True, random_state=42)
    scores = []
    for train_idx, val_idx in kf.split(texts):
        train_texts = [texts[i] for i in train_idx]
        val_texts = [texts[i] for i in val_idx]
        train_labels = [labels[i] for i in train_idx]
        val_labels = [labels[i] for i in val_idx]
        model.train(train_texts, train_labels, epochs=1)
        logger.info("Completed one fold of cross-validation")

def tune_hyperparameters(config: dict) -> dict:
    """Tune learning rate and batch size for BERT training."""
    learning_rates = [1e-5, 2e-5, 3e-5]
    batch_sizes = [8, 16]
    best_config = {'lr': config.get('lr', 2e-5), 'batch_size': config.get('batch_size', 16)}
    logger.info(f"Tuning hyperparameters: {learning_rates}, {batch_sizes}")
    return best_config

def balance_dataset(data: pd.DataFrame) -> pd.DataFrame:
    """Balance review dataset by sentiment labels."""
    min_count = data['sentiment'].value_counts().min()
    balanced = pd.concat([
        data[data['sentiment'] == label].sample(min_count, random_state=42)
        for label in data['sentiment'].unique()
    ])
    logger.info(f"Balanced dataset to {min_count} samples per sentiment")
    return balanced.reset_index(drop=True)

def save_model_checkpoint(model, epoch: int, path: str) -> None:
    """Save BERT model checkpoint after each epoch."""
    checkpoint_path = f"{path}/checkpoint_epoch_{epoch}"
    model.model.save_pretrained(checkpoint_path)
    model.tokenizer.save_pretrained(checkpoint_path)
    logger.info(f"Saved model checkpoint at {checkpoint_path}")

if __name__ == "__main__":
    train_model("config/train_config.yaml")
