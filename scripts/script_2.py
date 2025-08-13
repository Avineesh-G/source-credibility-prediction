# 1. Data Preprocessing Module
class DataPreprocessor:
    """
    Comprehensive data preprocessing for source credibility prediction
    """
    
    def __init__(self):
        self.stop_words = set(nltk.corpus.stopwords.words('english'))
    
    def clean_text(self, text):
        """Clean and normalize text data"""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove punctuation (optional, keep for some features)
        # text = text.translate(str.maketrans('', '', string.punctuation))
        
        return text.strip()
    
    def remove_stop_words(self, text):
        """Remove stop words from text"""
        words = text.split()
        return ' '.join([word for word in words if word not in self.stop_words])
    
    def preprocess_dataset(self, df, text_column='text', label_column='label'):
        """Preprocess entire dataset"""
        df = df.copy()
        
        # Clean text
        df['cleaned_text'] = df[text_column].apply(self.clean_text)
        
        # Create processed text without stop words
        df['processed_text'] = df['cleaned_text'].apply(self.remove_stop_words)
        
        # Remove empty texts
        df = df[df['cleaned_text'].str.len() > 0]
        
        return df

# 2. Feature Engineering Module
class FeatureEngineer:
    """
    Extract various features for credibility assessment
    """
    
    def __init__(self):
        pass
    
    def extract_linguistic_features(self, text):
        """Extract linguistic features from text"""
        if pd.isna(text) or len(str(text).strip()) == 0:
            return {
                'word_count': 0,
                'char_count': 0,
                'sentence_count': 0,
                'avg_word_length': 0,
                'punctuation_ratio': 0,
                'uppercase_ratio': 0,
                'question_marks': 0,
                'exclamation_marks': 0,
            }
        
        text = str(text)
        words = text.split()
        
        features = {
            'word_count': len(words),
            'char_count': len(text),
            'sentence_count': len(re.findall(r'[.!?]+', text)),
            'avg_word_length': np.mean([len(word) for word in words]) if words else 0,
            'punctuation_ratio': len(re.findall(r'[^\w\s]', text)) / len(text) if text else 0,
            'uppercase_ratio': len(re.findall(r'[A-Z]', text)) / len(text) if text else 0,
            'question_marks': text.count('?'),
            'exclamation_marks': text.count('!'),
        }
        
        return features
    
    def extract_sentiment_features(self, text):
        """Extract sentiment-based features"""
        if pd.isna(text) or len(str(text).strip()) == 0:
            return {
                'sentiment_polarity': 0,
                'sentiment_subjectivity': 0,
            }
        
        blob = TextBlob(str(text))
        
        return {
            'sentiment_polarity': blob.sentiment.polarity,
            'sentiment_subjectivity': blob.sentiment.subjectivity,
        }
    
    def extract_readability_features(self, text):
        """Extract readability metrics"""
        if pd.isna(text) or len(str(text).strip()) == 0:
            return {
                'flesch_reading_ease': 0,
                'flesch_kincaid_grade': 0,
            }
        
        try:
            # Using simple approximations since textstat might not be available
            words = len(str(text).split())
            sentences = max(1, len(re.findall(r'[.!?]+', str(text))))
            syllables = sum([max(1, len(re.findall(r'[aeiouAEIOU]', word))) for word in str(text).split()])
            
            # Simplified Flesch Reading Ease
            avg_sentence_length = words / sentences
            avg_syllables_per_word = syllables / max(1, words)
            flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
            
            # Simplified Flesch-Kincaid Grade
            fk_grade = (0.39 * avg_sentence_length) + (11.8 * avg_syllables_per_word) - 15.59
            
            return {
                'flesch_reading_ease': max(0, min(100, flesch_score)),
                'flesch_kincaid_grade': max(0, fk_grade),
            }
        except:
            return {
                'flesch_reading_ease': 0,
                'flesch_kincaid_grade': 0,
            }
    
    def create_feature_matrix(self, df, text_column='processed_text'):
        """Create comprehensive feature matrix"""
        features_list = []
        
        for idx, text in df[text_column].items():
            features = {}
            features.update(self.extract_linguistic_features(text))
            features.update(self.extract_sentiment_features(text))
            features.update(self.extract_readability_features(text))
            features_list.append(features)
        
        return pd.DataFrame(features_list, index=df.index)

# Test the preprocessing and feature engineering
print("Data Preprocessing and Feature Engineering modules created successfully!")

# Create sample data for testing
sample_data = {
    'text': [
        "BREAKING: Scientists discover cure for all diseases! Click here to learn more!!!",
        "According to a peer-reviewed study published in Nature, researchers have identified a new mechanism for cellular repair.",
        "You won't believe what this celebrity did! The media doesn't want you to know this secret.",
        "The World Health Organization released new guidelines for pandemic preparedness based on extensive research.",
        "URGENT: Government conspiracy exposed! They don't want you to see this information!"
    ],
    'label': [0, 1, 0, 1, 0]  # 0 = fake/unreliable, 1 = real/reliable
}

df_sample = pd.DataFrame(sample_data)
print(f"\nSample dataset created with {len(df_sample)} examples")
print(df_sample.head())