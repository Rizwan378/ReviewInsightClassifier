import re
import logging
from typing import List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TextCleaner:
    def __init__(self):
        self.stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to'])

    def clean_text(self, text: str) -> str:
        """Clean review text by removing special characters, numbers, and stop words."""
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)
        words = text.split()
        words = [word for word in words if word not in self.stop_words]
        cleaned_text = ' '.join(words)
        logger.debug(f"Cleaned text: {cleaned_text}")
        return cleaned_text

    def clean_batch(self, texts: List[str]) -> List[str]:
        """Clean a batch of review texts."""
        return [self.clean_text(text) for text in texts]

    def remove_punctuation(self, text: str) -> str:
        """Remove punctuation and special characters from review text."""
        text = re.sub(r'[^\w\s]', '', text)
        text = text.replace('\n', ' ').strip()
        text = re.sub(r'\s+', ' ', text)
        logger.debug(f"Removed punctuation: {text}")
        return text

    def tokenize_text(self, text: str) -> List[str]:
        """Tokenize review text into words, preserving meaningful units."""
        words = text.lower().split()
        words = [word for word in words if len(word) > 1]
        words = [word for word in words if word not in self.stop_words]
        logger.debug(f"Tokenized text: {words}")
        return words

    def stem_words(self, text: str) -> str:
        """Apply simple stemming to review text words."""
        from nltk.stem.porter import PorterStemmer
        stemmer = PorterStemmer()
        words = self.tokenize_text(text)
        stemmed = [stemmer.stem(word) for word in words]
        result = ' '.join(stemmed)
        logger.debug(f"Stemmed text: {result}")
        return result

    def remove_numbers(self, text: str) -> str:
        """Remove numerical digits from review text."""
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        if not text:
            logger.warning("Text empty after number removal")
        logger.debug(f"Removed numbers: {text}")
        return text

    def handle_contractions(self, text: str) -> str:
        """Expand common contractions in review text."""
        contractions = {"dont": "do not", "wont": "will not", "cant": "cannot"}
        for contraction, expanded in contractions.items():
            text = text.replace(contraction, expanded)
        text = re.sub(r'\s+', ' ', text).strip()
        logger.debug(f"Expanded contractions: {text}")
        return text

    def extract_ngrams(self, text: str, n: int = 2) -> List[str]:
        """Extract n-grams from review text for feature enrichment."""
        words = self.tokenize_text(text)
        if len(words) < n:
            return []
        ngrams = [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]
        logger.debug(f"Extracted {len(ngrams)} {n}-grams: {ngrams[:5]}")
        return ngrams

    def remove_punctuation(self, text: str) -> str:
        """Remove punctuation and special characters from review text."""
        text = re.sub(r'[^\w\s]', '', text)
        text = text.replace('\n', ' ').strip()
        text = re.sub(r'\s+', ' ', text)
        logger.debug(f"Removed punctuation: {text}")
        return text
