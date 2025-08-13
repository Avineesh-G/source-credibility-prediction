# Continue with evaluation and testing
evaluation_results = pipeline.evaluate_pipeline(X_test, y_test)

# Test predictions on new examples
test_examples = [
    "BREAKING: Miracle cure discovered! Scientists hate this one trick!",
    "According to a recent study published in Nature, researchers have identified new therapeutic targets.",
    "You won't believe what celebrities are doing! Click here for shocking secrets!",
    "The Centers for Disease Control released updated guidelines based on scientific evidence."
]

print("\nTesting Predictions on New Examples:")
print("=" * 50)

predictions, probabilities = pipeline.predict(test_examples)

for i, (text, pred, prob) in enumerate(zip(test_examples, predictions, probabilities)):
    label = "Reliable" if pred == 1 else "Unreliable"
    confidence = prob[pred]
    print(f"\nExample {i+1}:")
    print(f"Text: {text[:80]}...")
    print(f"Prediction: {label} (Confidence: {confidence:.4f})")

# 8. Utility Functions Module
class CredibilityUtils:
    """
    Utility functions for the credibility prediction system
    """
    
    @staticmethod
    def load_dataset(file_path, text_column='text', label_column='label'):
        """Load dataset from various formats"""
        if file_path.endswith('.csv'):
            return pd.read_csv(file_path)
        elif file_path.endswith('.json'):
            return pd.read_json(file_path)
        elif file_path.endswith('.xlsx'):
            return pd.read_excel(file_path)
        else:
            raise ValueError("Unsupported file format")
    
    @staticmethod
    def save_model(model, file_path):
        """Save trained model"""
        import pickle
        with open(file_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"Model saved to {file_path}")
    
    @staticmethod
    def load_model(file_path):
        """Load trained model"""
        import pickle
        with open(file_path, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded from {file_path}")
        return model
    
    @staticmethod
    def create_credibility_report(text, prediction, confidence, features=None):
        """Create detailed credibility assessment report"""
        report = {
            'text': text[:200] + "..." if len(text) > 200 else text,
            'prediction': 'Reliable' if prediction == 1 else 'Unreliable',
            'confidence': float(confidence),
            'assessment_date': pd.Timestamp.now().isoformat(),
            'features': features if features else {}
        }
        return report
    
    @staticmethod
    def batch_process(texts, pipeline, batch_size=100):
        """Process large batches of texts efficiently"""
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            predictions, probabilities = pipeline.predict(batch)
            
            for text, pred, prob in zip(batch, predictions, probabilities):
                results.append({
                    'text': text,
                    'prediction': pred,
                    'confidence': prob[pred],
                    'reliable_prob': prob[1],
                    'unreliable_prob': prob[0]
                })
        
        return pd.DataFrame(results)

# 9. Configuration and Constants
class Config:
    """Configuration settings for the credibility prediction system"""
    
    # Model parameters
    MAX_FEATURES = 10000
    MAX_SEQUENCE_LENGTH = 500
    EMBEDDING_DIM = 100
    LSTM_UNITS = 128
    
    # Training parameters
    BATCH_SIZE = 32
    EPOCHS = 10
    LEARNING_RATE = 0.001
    VALIDATION_SPLIT = 0.2
    
    # Evaluation parameters
    TEST_SIZE = 0.2
    CV_FOLDS = 5
    RANDOM_STATE = 42
    
    # Feature engineering
    TFIDF_MAX_FEATURES = 5000
    NGRAM_RANGE = (1, 2)
    
    # Thresholds
    CONFIDENCE_THRESHOLD = 0.7
    HIGH_RISK_THRESHOLD = 0.9

# Test utility functions
utils = CredibilityUtils()

# Create detailed reports for test examples
print("\nDetailed Credibility Reports:")
print("=" * 50)

for i, (text, pred, prob) in enumerate(zip(test_examples, predictions, probabilities)):
    confidence = prob[pred]
    report = utils.create_credibility_report(text, pred, confidence)
    
    print(f"\nReport {i+1}:")
    print(f"Text: {report['text']}")
    print(f"Prediction: {report['prediction']}")
    print(f"Confidence: {report['confidence']:.4f}")
    print(f"Assessment Date: {report['assessment_date']}")

# Create batch processing example
print("\nBatch Processing Example:")
print("=" * 30)

batch_texts = [
    "URGENT: Government hiding truth about vaccines!",
    "Scientific study shows promising results for new treatment.",
    "You won't believe this celebrity scandal!",
    "WHO reports on global health initiatives.",
    "BREAKING: Shocking discovery that doctors hate!"
]

batch_results = utils.batch_process(batch_texts, pipeline)
print(batch_results[['prediction', 'confidence', 'reliable_prob']].round(4))