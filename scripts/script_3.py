# 3. Traditional Machine Learning Models Module
class TraditionalMLModels:
    """
    Traditional machine learning models for source credibility prediction
    """
    
    def __init__(self):
        self.models = {
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'svm': SVC(random_state=42, probability=True),
            'random_forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'naive_bayes': MultinomialNB(),
            'decision_tree': DecisionTreeClassifier(random_state=42)
        }
        self.vectorizers = {
            'tfidf': TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2)),
            'count': CountVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
        }
        self.pipelines = {}
        self.results = {}
    
    def create_pipelines(self):
        """Create ML pipelines combining vectorizers and models"""
        for vec_name, vectorizer in self.vectorizers.items():
            for model_name, model in self.models.items():
                pipeline_name = f"{vec_name}_{model_name}"
                self.pipelines[pipeline_name] = Pipeline([
                    ('vectorizer', vectorizer),
                    ('classifier', model)
                ])
    
    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        """Train and evaluate all model pipelines"""
        self.create_pipelines()
        
        for pipeline_name, pipeline in self.pipelines.items():
            print(f"Training {pipeline_name}...")
            
            # Train the model
            pipeline.fit(X_train, y_train)
            
            # Make predictions
            y_pred = pipeline.predict(X_test)
            y_pred_proba = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, 'predict_proba') else None
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            self.results[pipeline_name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            print(f"  Accuracy: {accuracy:.4f}")
    
    def get_best_model(self):
        """Get the best performing model based on F1 score"""
        if not self.results:
            return None
        
        best_model_name = max(self.results.keys(), key=lambda k: self.results[k]['f1_score'])
        return best_model_name, self.pipelines[best_model_name]
    
    def cross_validate(self, X, y, cv=5):
        """Perform cross-validation on all models"""
        cv_results = {}
        
        for pipeline_name, pipeline in self.pipelines.items():
            scores = cross_val_score(pipeline, X, y, cv=cv, scoring='f1_weighted')
            cv_results[pipeline_name] = {
                'mean_score': scores.mean(),
                'std_score': scores.std(),
                'scores': scores
            }
        
        return cv_results

# 4. Deep Learning Models Framework (Structure for when TensorFlow is available)
class DeepLearningModels:
    """
    Deep learning models for source credibility prediction
    Note: This is a framework - requires TensorFlow/PyTorch for actual implementation
    """
    
    def __init__(self, max_features=10000, max_length=500):
        self.max_features = max_features
        self.max_length = max_length
        self.tokenizer = None
        self.models = {}
    
    def prepare_sequences(self, texts):
        """Prepare text sequences for deep learning models"""
        # Note: This would use tensorflow.keras.preprocessing in a full implementation
        print("Preparing text sequences for deep learning...")
        print("Note: Requires TensorFlow for full implementation")
        
        # Mock implementation for structure
        return {
            'sequences': f"Tokenized sequences (length: {self.max_length})",
            'vocab_size': self.max_features
        }
    
    def build_lstm_model(self, embedding_dim=100, lstm_units=128):
        """Build LSTM model architecture"""
        model_structure = f"""
        LSTM Model Architecture:
        ========================
        1. Embedding Layer: {self.max_features} -> {embedding_dim}
        2. LSTM Layer: {lstm_units} units
        3. Dropout: 0.5
        4. Dense Layer: 64 units (ReLU)
        5. Output Layer: 1 unit (Sigmoid)
        
        Total Parameters: ~{(self.max_features * embedding_dim + lstm_units * 4 * (embedding_dim + lstm_units) + 64 * lstm_units + 64 + 1):,}
        """
        return model_structure
    
    def build_cnn_lstm_model(self):
        """Build CNN-LSTM hybrid model"""
        model_structure = """
        CNN-LSTM Hybrid Model:
        ======================
        1. Embedding Layer
        2. Conv1D Layer: 128 filters, kernel_size=5
        3. MaxPooling1D: pool_size=2
        4. LSTM Layer: 128 units
        5. Dense Layer: 64 units
        6. Output Layer: 1 unit
        """
        return model_structure
    
    def build_attention_model(self):
        """Build attention-based model"""
        model_structure = """
        Attention-Based Model:
        ======================
        1. Embedding Layer
        2. Bidirectional LSTM
        3. Multi-Head Attention
        4. Global Average Pooling
        5. Dense Layers with Dropout
        6. Output Layer
        """
        return model_structure

# 5. Transformer Models Framework
class TransformerModels:
    """
    Transformer-based models for credibility assessment
    Note: Requires transformers library for full implementation
    """
    
    def __init__(self):
        self.models = [
            'bert-base-uncased',
            'roberta-base',
            'distilbert-base-uncased',
            'xlnet-base-cased'
        ]
    
    def load_pretrained_model(self, model_name):
        """Load pre-trained transformer model"""
        print(f"Loading {model_name}...")
        print("Note: Requires transformers library for full implementation")
        
        model_info = {
            'model_name': model_name,
            'max_length': 512,
            'num_labels': 2,
            'architecture': f"{model_name} + Classification Head"
        }
        return model_info
    
    def fine_tune_bert(self, train_data, val_data):
        """Fine-tune BERT for credibility classification"""
        training_config = {
            'model': 'bert-base-uncased',
            'num_epochs': 3,
            'learning_rate': 2e-5,
            'batch_size': 16,
            'max_length': 512,
            'warmup_steps': 500,
            'weight_decay': 0.01
        }
        
        print("BERT Fine-tuning Configuration:")
        for key, value in training_config.items():
            print(f"  {key}: {value}")
        
        return training_config

# Test the traditional ML models
preprocessor = DataPreprocessor()
feature_engineer = FeatureEngineer()
ml_models = TraditionalMLModels()

# Preprocess sample data
df_processed = preprocessor.preprocess_dataset(df_sample)
print(f"\nProcessed dataset shape: {df_processed.shape}")
print(df_processed[['cleaned_text', 'label']].head())