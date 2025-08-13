import re
import string
from nltk.corpus import stopwords
import nltk

# Make sure NLTK data is downloaded
nltk.download('stopwords')
nltk.download('punkt')

class DataPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words("english"))

    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r"http\S+", "", text)
        text = text.translate(str.maketrans("", "", string.punctuation))
        tokens = nltk.word_tokenize(text)
        tokens = [w for w in tokens if w not in self.stop_words and w.isalpha()]
        return " ".join(tokens)
