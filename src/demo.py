"""
Demonstration Script for Source Credibility Prediction System
=============================================================

This script demonstrates the complete functionality of the source
credibility prediction system using all implemented components.

Run this script to see the system in action with sample data.
"""

from src.traditional_ml_models import TraditionalMLModels
from src.evaluation import CredibilityEvaluator
from src.data_preprocessing import DataPreprocessor


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add src directory to path for imports
sys.path.append('src')

# Import all modules
from data_preprocessing import DataPreprocessor
from feature_engineering import FeatureEngineer
from traditional_ml_models import TraditionalMLModels
from evaluation import CredibilityEvaluator
from pipeline import CredibilityPredictionPipeline

def create_sample_dataset():
    """Create a sample dataset for demonstration"""
    
    print("Creating sample dataset...")
    
    # Sample fake news examples (label = 0)
    fake_news = [
        "BREAKING: Scientists discover cure for all diseases! Click here to learn more!!!",
        "You won't believe what this celebrity did! The media doesn't want you to know this secret.",
        "URGENT: Government conspiracy exposed! They don't want you to see this information!",
        "SHOCKING: Scientists hate this one simple trick! Doctors are furious!",
        "Miracle weight loss pill burns fat instantly! Pharmaceutical companies hate this!",
        "EXPOSED: The truth about vaccines that doctors won't tell you!",
        "Unbelievable discovery that will change everything! Click to see what they're hiding!",
        "Secret government program revealed! This will shock you!",
        "Amazing superfood that cures cancer! Big pharma doesn't want you to know!",
        "Incredible technology breakthrough hidden from the public!"
    ]
    
    # Sample reliable news examples (label = 1)
    real_news = [
        "According to a peer-reviewed study published in Nature, researchers have identified a new mechanism.",
        "The World Health Organization released new guidelines based on extensive research.",
        "A comprehensive analysis conducted by Stanford University indicates significant progress.",
        "Researchers from Harvard Medical School have published evidence in the Journal of Medicine.",
        "The Centers for Disease Control announced updated recommendations following clinical trials.",
        "A meta-analysis of 50 studies shows consistent results across multiple populations.",
        "Scientists from MIT published findings in Science magazine after rigorous peer review.",
        "The Food and Drug Administration approved the treatment based on Phase III clinical trials.",
        "Research published in The Lancet demonstrates the efficacy of the new approach.",
        "A longitudinal study spanning 10 years reveals important health trends."
    ]
    
    # Combine data
    texts = fake_news + real_news
    labels = [0] * len(fake_news) + [1] * len(real_news)
    
    # Create DataFrame
    df = pd.DataFrame({
        'text': texts,
        'label': labels
    })
    
    # Shuffle the data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Sample dataset created with {len(df)} examples")
    print(f"Class distribution: {df['label'].value_counts().to_dict()}")
    
    return df

def demonstrate_preprocessing():
    """Demonstrate data preprocessing capabilities"""
    
    print("\\n" + "="*60)
    print("DEMONSTRATING DATA PREPROCESSING")
    print("="*60)
    
    preprocessor = DataPreprocessor()
    
    # Sample texts
    sample_texts = [
        "BREAKING: Scientists HATE this ONE simple trick!!! Click HERE to learn more: http://fake-site.com #fakenews @user123",
        "According to a peer-reviewed study published in Nature, researchers have identified new therapeutic targets.",
    ]
    
    for i, text in enumerate(sample_texts, 1):
        print(f"\\nExample {i}:")
        print(f"Original: {text}")
        print(f"Cleaned:  {preprocessor.clean_text(text)}")
        print(f"No stops: {preprocessor.remove_stop_words(preprocessor.clean_text(text))}")

def demonstrate_feature_engineering():
    """Demonstrate feature engineering capabilities"""
    
    print("\\n" + "="*60)
    print("DEMONSTRATING FEATURE ENGINEERING")
    print("="*60)
    
    feature_engineer = FeatureEngineer()
    
    sample_text = "BREAKING: Scientists discover SHOCKING truth! You won't believe what they found!!!"
    
    print(f"\\nAnalyzing text: '{sample_text}'")
    print("\\nExtracted Features:")
    
    # Extract all types of features
    linguistic = feature_engineer.extract_linguistic_features(sample_text)
    sentiment = feature_engineer.extract_sentiment_features(sample_text)
    readability = feature_engineer.extract_readability_features(sample_text)
    structural = feature_engineer.extract_structural_features(sample_text)
    credibility = feature_engineer.extract_credibility_indicators(sample_text)
    
    print("\\nLinguistic Features:")
    for key, value in linguistic.items():
        print(f"  {key}: {value}")
    
    print("\\nSentiment Features:")
    for key, value in sentiment.items():
        print(f"  {key}: {value}")
    
    print("\\nCredibility Indicators:")
    for key, value in credibility.items():
        print(f"  {key}: {value}")

def demonstrate_model_training():
    """Demonstrate model training and evaluation"""
    
    print("\\n" + "="*60)
    print("DEMONSTRATING MODEL TRAINING")
    print("="*60)
    
    # Create sample dataset
    df = create_sample_dataset()
    
    # Initialize pipeline
    pipeline = CredibilityPredictionPipeline()
    
    # Prepare data
    X_train, X_test, y_train, y_test = pipeline.prepare_data(df)
    
    # Train models
    results = pipeline.train_models(X_train, X_test, y_train, y_test)
    
    # Evaluate pipeline
    evaluation_results = pipeline.evaluate_pipeline(X_test, y_test)
    
    return pipeline, evaluation_results

def demonstrate_predictions(pipeline):
    """Demonstrate making predictions"""
    
    print("\\n" + "="*60)
    print("DEMONSTRATING PREDICTIONS")
    print("="*60)
    
    # Test examples
    test_examples = [
        "BREAKING: Miracle cure discovered! Scientists hate this one trick!",
        "According to a recent study published in Nature, researchers have identified new therapeutic targets.",
        "You won't believe what celebrities are doing! Click here for shocking secrets!",
        "The Centers for Disease Control released updated guidelines based on scientific evidence.",
        "URGENT: Government hiding truth! They don't want you to know this!"
    ]
    
    print("\\nTesting predictions on new examples:")
    
    for i, text in enumerate(test_examples, 1):
        try:
            explanation = pipeline.predict_with_explanation(text)
            
            print(f"\\nExample {i}:")
            print(f"Text: {text[:80]}...")
            print(f"Prediction: {explanation['prediction']}")
            print(f"Confidence: {explanation['confidence']:.4f}")
            print(f"Reliable Probability: {explanation['probability_reliable']:.4f}")
            print(f"Model Used: {explanation['model_used']}")
        except Exception as e:
            print(f"\\nExample {i}: Error - {str(e)}")

def demonstrate_batch_processing(pipeline):
    """Demonstrate batch processing capabilities"""
    
    print("\\n" + "="*60)
    print("DEMONSTRATING BATCH PROCESSING")
    print("="*60)
    
    batch_texts = [
        "URGENT: Government hiding truth about vaccines!",
        "Scientific study shows promising results for new treatment.",
        "You won't believe this celebrity scandal!",
        "WHO reports on global health initiatives.",
        "BREAKING: Shocking discovery that doctors hate!",
        "Peer-reviewed research published in medical journal.",
        "Click here for the secret they don't want you to know!",
        "Clinical trial results demonstrate safety and efficacy."
    ]
    
    print(f"\\nProcessing {len(batch_texts)} texts...")
    
    try:
        predictions, probabilities = pipeline.predict(batch_texts)
        
        results_df = pd.DataFrame({
            'text': [text[:50] + "..." if len(text) > 50 else text for text in batch_texts],
            'prediction': ['Reliable' if pred == 1 else 'Unreliable' for pred in predictions],
            'confidence': [prob[pred] for pred, prob in zip(predictions, probabilities)],
            'reliable_prob': [prob[1] for prob in probabilities]
        })
        
        print("\\nBatch Processing Results:")
        print(results_df.to_string(index=False))
        
    except Exception as e:
        print(f"Batch processing error: {str(e)}")

def main():
    """Main demonstration function"""
    
    print("SOURCE CREDIBILITY PREDICTION SYSTEM")
    print("COMPREHENSIVE DEMONSTRATION")
    print("="*80)
    
    try:
        # 1. Demonstrate preprocessing
        demonstrate_preprocessing()
        
        # 2. Demonstrate feature engineering
        demonstrate_feature_engineering()
        
        # 3. Demonstrate model training
        pipeline, evaluation_results = demonstrate_model_training()
        
        # 4. Demonstrate predictions
        demonstrate_predictions(pipeline)
        
        # 5. Demonstrate batch processing
        demonstrate_batch_processing(pipeline)
        
        print("\\n" + "="*80)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("="*80)
        
        print("\\nNext Steps:")
        print("1. Replace sample data with real datasets")
        print("2. Experiment with deep learning models (requires TensorFlow)")
        print("3. Add transformer models (requires transformers library)")
        print("4. Deploy as web API using the provided Flask interface")
        print("5. Integrate with real-time news feeds for live monitoring")
        
    except Exception as e:
        print(f"\\nDemonstration failed with error: {str(e)}")
        print("Please ensure all dependencies are installed:")
        print("pip install -r requirements.txt")

if __name__ == "__main__":
    main()