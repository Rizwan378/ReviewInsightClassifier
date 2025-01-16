from transformers import BertTokenizer, BertForSequenceClassification
import torch
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BertSentimentIntentClassifier:
    def __init__(self, model_path: str):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased',
            num_labels=7  # 3 sentiments (pos, neg, neu) + 4 intents (complaint, suggestion, question, praise)
        )
        self.model_path = model_path
        self.sentiment_labels = ['positive', 'negative', 'neutral']
        self.intent_labels = ['complaint', 'suggestion', 'question', 'praise']

    def train(self, texts, labels, epochs=3):
        inputs = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=128)
        labels = torch.tensor(labels)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5)
        
        self.model.train()
        for epoch in range(epochs):
            outputs = self.model(**inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            logger.info(f"Epoch {epoch + 1}, Loss: {loss.item()}")

        self.model.save_pretrained(self.model_path)
        self.tokenizer.save_pretrained(self.model_path)

    def predict(self, text):
        self.model.eval()
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits
        sentiment_pred = torch.argmax(logits[:, :3], dim=1).item()
        intent_pred = torch.argmax(logits[:, 3:], dim=1).item()
        return {
            'sentiment': self.sentiment_labels[sentiment_pred],
            'intent': self.intent_labels[intent_pred]
        }

    def schedule_learning_rate(self, optimizer, epoch: int, total_epochs: int) -> None:
        """Adjust learning rate based on epoch progression."""
        initial_lr = 2e-5
        decay = 0.9 ** epoch
        new_lr = initial_lr * decay
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
        logger.info(f"Epoch {epoch + 1}: Set learning rate to {new_lr}")

    def augment_data(self, texts: List[str]) -> List[str]:
        """Augment review texts by synonym replacement."""
        from nltk.corpus import wordnet
        augmented = []
        for text in texts:
            words = text.split()
            if len(words) > 0 and wordnet.synsets(words[0]):
                syn = wordnet.synsets(words[0])[0].lemmas()[0].name()
                words[0] = syn
            augmented.append(' '.join(words))
        logger.info("Augmented review texts with synonyms")
        return augmented

    def log_training_metrics(self, outputs, epoch: int) -> None:
        """Log training loss and accuracy metrics."""
        loss = outputs.loss.item()
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        accuracy = (preds == torch.tensor(labels)).float().mean().item()
        logger.info(f"Epoch {epoch + 1}: Loss = {loss:.4f}, Accuracy = {accuracy:.4f}")

    def adjust_batch_size(self, texts: List[str], max_batch: int = 16) -> List[List[str]]:
        """Split texts into batches for efficient training."""
        batch_size = min(max_batch, len(texts))
        batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
        logger.info(f"Split {len(texts)} texts into {len(batches)} batches")
        return batches

    def apply_weight_decay(self, optimizer, weight_decay: float = 0.01) -> None:
        """Apply weight decay to model parameters."""
        for param_group in optimizer.param_groups:
            param_group['weight_decay'] = weight_decay
        logger.info(f"Applied weight decay of {weight_decay} to optimizer")
        for name, param in self.model.named_parameters():
            if 'bias' not in name:
                param.data = param.data * (1 - weight_decay)

    def schedule_learning_rate(self, optimizer, epoch: int, total_epochs: int) -> None:
        """Adjust learning rate based on epoch progression."""
        initial_lr = 2e-5
        decay = 0.9 ** epoch
        new_lr = initial_lr * decay
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
        logger.info(f"Epoch {epoch + 1}: Set learning rate to {new_lr}")

    def augment_data(self, texts: List[str]) -> List[str]:
        """Augment review texts by synonym replacement."""
        from nltk.corpus import wordnet
        augmented = []
        for text in texts:
            words = text.split()
            if len(words) > 0 and wordnet.synsets(words[0]):
                syn = wordnet.synsets(words[0])[0].lemmas()[0].name()
                words[0] = syn
            augmented.append(' '.join(words))
        logger.info("Augmented review texts with synonyms")
        return augmented

    def log_training_metrics(self, outputs, epoch: int) -> None:
        """Log training loss and accuracy metrics."""
        loss = outputs.loss.item()
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        accuracy = (preds == torch.tensor(labels)).float().mean().item()
        logger.info(f"Epoch {epoch + 1}: Loss = {loss:.4f}, Accuracy = {accuracy:.4f}")

    def adjust_batch_size(self, texts: List[str], max_batch: int = 16) -> List[List[str]]:
        """Split texts into batches for efficient training."""
        batch_size = min(max_batch, len(texts))
        batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
        logger.info(f"Split {len(texts)} texts into {len(batches)} batches")
        return batches
