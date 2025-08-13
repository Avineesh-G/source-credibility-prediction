# 6. Evaluation Module
class CredibilityEvaluator:
    """
    Comprehensive evaluation system for credibility assessment models
    """
    
    def __init__(self):
        self.results = {}
    
    def evaluate_model(self, y_true, y_pred, y_pred_proba=None, model_name="Model"):
        """Evaluate a single model's performance"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1_score': f1_score(y_true, y_pred, average='weighted'),
            'classification_report': classification_report(y_true, y_pred)
        }
        
        if y_pred_proba is not None:
            from sklearn.metrics import roc_auc_score
            try:
                metrics['auc_roc'] = roc_auc_score(y_true, y_pred_proba)
            except:
                metrics['auc_roc'] = None
        
        self.results[model_name] = metrics
        return metrics
    
    def compare_models(self, results_dict):
        """Compare multiple models and create a summary"""
        comparison_df = pd.DataFrame({
            model: {
                'Accuracy': results['accuracy'],
                'Precision': results['precision'],
                'Recall': results['recall'],
                'F1-Score': results['f1_score']
            }
            for model, results in results_dict.items()
        }).T
        
        comparison_df = comparison_df.round(4)
        comparison_df = comparison_df.sort_values('F1-Score', ascending=False)
        
        return comparison_df
    
    def plot_confusion_matrix(self, y_true, y_pred, model_name="Model"):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        return {
            'confusion_matrix': cm,
            'model_name': model_name,
            'accuracy': accuracy_score(y_true, y_pred)
        }

# 7. Complete Pipeline Integration
class CredibilityPredictionPipeline:
    """
    Complete pipeline for source credibility prediction
    """
    
    def __init__(self):
        self.preprocessor = DataPreprocessor()
        self.feature_engineer = FeatureEngineer()
        self.ml_models = TraditionalMLModels()
        self.evaluator = CredibilityEvaluator()
        self.is_fitted = False
        self.best_model = None
    
    def prepare_data(self, df, text_column='text', label_column='label', test_size=0.2):
        """Prepare data for training and testing"""
        # Preprocess text
        df_processed = self.preprocessor.preprocess_dataset(df, text_column, label_column)
        
        # Split data
        X = df_processed['processed_text']
        y = df_processed[label_column]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        return X_train, X_test, y_train, y_test
    
    def train_models(self, X_train, X_test, y_train, y_test):
        """Train all models and evaluate performance"""
        print("Training traditional ML models...")
        print("=" * 50)
        
        # Train and evaluate models
        self.ml_models.train_and_evaluate(X_train, X_test, y_train, y_test)
        
        # Get best model
        best_model_name, self.best_model = self.ml_models.get_best_model()
        
        print(f"\nBest model: {best_model_name}")
        print(f"Best F1 Score: {self.ml_models.results[best_model_name]['f1_score']:.4f}")
        
        self.is_fitted = True
        return self.ml_models.results
    
    def predict(self, texts):
        """Make predictions on new texts"""
        if not self.is_fitted:
            raise ValueError("Pipeline must be trained before making predictions")
        
        # Preprocess texts
        processed_texts = [self.preprocessor.clean_text(text) for text in texts]
        processed_texts = [self.preprocessor.remove_stop_words(text) for text in processed_texts]
        
        # Make predictions
        predictions = self.best_model.predict(processed_texts)
        probabilities = self.best_model.predict_proba(processed_texts)
        
        return predictions, probabilities
    
    def evaluate_pipeline(self, X_test, y_test):
        """Evaluate the complete pipeline"""
        results_comparison = self.evaluator.compare_models(self.ml_models.results)
        print("\nModel Comparison:")
        print("=" * 60)
        print(results_comparison)
        
        return results_comparison

# Test the complete pipeline
print("Testing Complete Credibility Prediction Pipeline")
print("=" * 60)

# Initialize pipeline
pipeline = CredibilityPredictionPipeline()

# Prepare data
X_train, X_test, y_train, y_test = pipeline.prepare_data(synthetic_df)

print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")
print(f"Class distribution in training: {y_train.value_counts().to_dict()}")
print(f"Class distribution in test: {y_test.value_counts().to_dict()}")

# Train models
results = pipeline.train_models(X_train, X_test, y_train, y_test)