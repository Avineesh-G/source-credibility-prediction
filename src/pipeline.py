"""
Complete Pipeline Module
========================

This module integrates all components into a unified pipeline for
source credibility prediction.

Classes:
    CredibilityPredictionPipeline: Main pipeline class
"""

from data_preprocessing import DataPreprocessor


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Import custom modules
from data_preprocessing import DataPreprocessor
from feature_engineering import FeatureEngineer
from traditional_ml_models import TraditionalMLModels
from evaluation import CredibilityEvaluator

class CredibilityPredictionPipeline:
    """
    Complete pipeline for source credibility prediction
    """
    
    def __init__(self):
        """Initialize the pipeline with all components"""
        self.preprocessor = DataPreprocessor()
        self.feature_engineer = FeatureEngineer()
        self.ml_models = TraditionalMLModels()
        self.evaluator = CredibilityEvaluator()
        self.is_fitted = False
        self.best_model = None
        self.best_model_name = None
    
    def prepare_data(self, df, text_column='text', label_column='label', test_size=0.2):
        """
        Prepare data for training and testing
        
        Args:
            df (pd.DataFrame): Input dataset
            text_column (str): Name of text column
            label_column (str): Name of label column
            test_size (float): Proportion of test data
            
        Returns:
            tuple: X_train, X_test, y_train, y_test
        """
        print("Preparing data...")
        
        # Preprocess text
        df_processed = self.preprocessor.preprocess_dataset(df, text_column, label_column)
        
        # Split data
        X = df_processed['processed_text']
        y = df_processed[label_column]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"Data split: {len(X_train)} training, {len(X_test)} testing samples")
        
        return X_train, X_test, y_train, y_test
    
    def train_models(self, X_train, X_test, y_train, y_test):
        """
        Train all models and evaluate performance
        
        Args:
            X_train: Training features
            X_test: Test features
            y_train: Training labels
            y_test: Test labels
            
        Returns:
            dict: Training results
        """
        print("Training traditional ML models...")
        print("=" * 50)
        
        # Train and evaluate models
        self.ml_models.train_and_evaluate(X_train, X_test, y_train, y_test)
        
        # Get best model
        self.best_model_name, self.best_model = self.ml_models.get_best_model()
        
        if self.best_model_name:
            print(f"\\nBest model: {self.best_model_name}")
            print(f"Best F1 Score: {self.ml_models.results[self.best_model_name]['f1_score']:.4f}")
            self.is_fitted = True
        
        return self.ml_models.results
    
    def predict(self, texts):
        """
        Make predictions on new texts
        
        Args:
            texts (list): List of texts to predict
            
        Returns:
            tuple: predictions, probabilities
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be trained before making predictions")
        
        # Preprocess texts
        processed_texts = []
        for text in texts:
            cleaned = self.preprocessor.clean_text(text)
            processed = self.preprocessor.remove_stop_words(cleaned)
            processed_texts.append(processed)
        
        # Make predictions
        predictions = self.best_model.predict(processed_texts)
        probabilities = self.best_model.predict_proba(processed_texts)
        
        return predictions, probabilities
    
    def evaluate_pipeline(self, X_test, y_test):
        """
        Evaluate the complete pipeline
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            pd.DataFrame: Comparison results
        """
        results_comparison = self.evaluator.compare_models(self.ml_models.results)
        print("\\nModel Comparison:")
        print("=" * 60)
        print(results_comparison)
        
        return results_comparison
    
    def cross_validate(self, X, y, cv=5):
        """
        Perform cross-validation
        
        Args:
            X: Features
            y: Labels
            cv: Number of folds
            
        Returns:
            dict: Cross-validation results
        """
        print(f"Performing {cv}-fold cross-validation...")
        cv_results = self.ml_models.cross_validate(X, y, cv)
        
        print("\\nCross-validation Results:")
        for model_name, results in cv_results.items():
            print(f"{model_name}: {results['mean_score']:.4f} (+/- {results['std_score']*2:.4f})")
        
        return cv_results
    
    def get_feature_importance(self, X_train, y_train):
        """
        Get feature importance from the best model
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            pd.DataFrame: Feature importance scores
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be trained first")
        
        # Extract features for importance analysis
        df_temp = pd.DataFrame({'processed_text': X_train, 'label': y_train})
        feature_matrix = self.feature_engineer.create_feature_matrix(df_temp)
        
        importance_df = self.feature_engineer.get_feature_importance(feature_matrix, y_train)
        
        return importance_df
    
    def predict_with_explanation(self, text):
        """
        Predict with detailed explanation of features
        
        Args:
            text (str): Input text
            
        Returns:
            dict: Prediction with explanation
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be trained before making predictions")
        
        # Preprocess
        cleaned_text = self.preprocessor.clean_text(text)
        processed_text = self.preprocessor.remove_stop_words(cleaned_text)
        
        # Extract features
        df_temp = pd.DataFrame({'processed_text': [processed_text]})
        features = self.feature_engineer.create_feature_matrix(df_temp)
        
        # Make prediction
        prediction = self.best_model.predict([processed_text])[0]
        probability = self.best_model.predict_proba([processed_text])[0]
        
        # Compile explanation
        explanation = {
            'original_text': text,
            'cleaned_text': cleaned_text,
            'processed_text': processed_text,
            'prediction': 'Reliable' if prediction == 1 else 'Unreliable',
            'confidence': probability[prediction],
            'probability_reliable': probability[1],
            'probability_unreliable': probability[0],
            'model_used': self.best_model_name,
            'extracted_features': features.iloc[0].to_dict()
        }
        
        return explanation