"""
Feature Engineering Module
==========================

This module handles feature extraction for source credibility prediction,
including linguistic, sentiment, readability, and structural features.

Classes:
    FeatureEngineer: Main class for feature engineering operations
"""

from data_preprocessing import DataPreprocessor


import pandas as pd
import numpy as np
import re
from textblob import TextBlob
from collections import Counter
import string

class FeatureEngineer:
    """
    Extract various features for credibility assessment
    """
    
    def __init__(self):
        """Initialize the feature engineer"""
        self.feature_names = []
    
    def extract_linguistic_features(self, text):
        """
        Extract linguistic features from text
        
        Args:
            text (str): Input text
            
        Returns:
            dict: Dictionary of linguistic features
        """
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
                'digit_ratio': 0,
                'unique_word_ratio': 0
            }
        
        text = str(text)
        words = text.split()
        
        # Basic counts
        word_count = len(words)
        char_count = len(text)
        sentence_count = max(1, len(re.findall(r'[.!?]+', text)))
        
        # Word-related features
        avg_word_length = np.mean([len(word) for word in words]) if words else 0
        unique_words = len(set(words))
        unique_word_ratio = unique_words / word_count if word_count > 0 else 0
        
        # Character-related features
        punctuation_count = len(re.findall(r'[^\w\s]', text))
        punctuation_ratio = punctuation_count / char_count if char_count > 0 else 0
        
        uppercase_count = len(re.findall(r'[A-Z]', text))
        uppercase_ratio = uppercase_count / char_count if char_count > 0 else 0
        
        digit_count = len(re.findall(r'\d', text))
        digit_ratio = digit_count / char_count if char_count > 0 else 0
        
        # Specific punctuation
        question_marks = text.count('?')
        exclamation_marks = text.count('!')
        
        features = {
            'word_count': word_count,
            'char_count': char_count,
            'sentence_count': sentence_count,
            'avg_word_length': avg_word_length,
            'punctuation_ratio': punctuation_ratio,
            'uppercase_ratio': uppercase_ratio,
            'question_marks': question_marks,
            'exclamation_marks': exclamation_marks,
            'digit_ratio': digit_ratio,
            'unique_word_ratio': unique_word_ratio
        }
        
        return features
    
    def extract_sentiment_features(self, text):
        """
        Extract sentiment-based features
        
        Args:
            text (str): Input text
            
        Returns:
            dict: Dictionary of sentiment features
        """
        if pd.isna(text) or len(str(text).strip()) == 0:
            return {
                'sentiment_polarity': 0,
                'sentiment_subjectivity': 0,
                'positive_words': 0,
                'negative_words': 0
            }
        
        try:
            blob = TextBlob(str(text))
            
            # Basic sentiment scores
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            
            # Word-level sentiment analysis
            words = text.lower().split()
            
            # Simple positive/negative word lists (can be expanded)
            positive_words = {
                'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
                'outstanding', 'brilliant', 'superb', 'perfect', 'incredible',
                'awesome', 'magnificent', 'remarkable', 'exceptional', 'marvelous'
            }
            
            negative_words = {
                'bad', 'terrible', 'awful', 'horrible', 'disgusting', 'pathetic',
                'disaster', 'catastrophe', 'nightmare', 'tragic', 'devastating',
                'shocking', 'outrageous', 'appalling', 'dreadful', 'abysmal'
            }
            
            pos_count = sum(1 for word in words if word in positive_words)
            neg_count = sum(1 for word in words if word in negative_words)
            
            return {
                'sentiment_polarity': polarity,
                'sentiment_subjectivity': subjectivity,
                'positive_words': pos_count,
                'negative_words': neg_count
            }
        
        except Exception:
            return {
                'sentiment_polarity': 0,
                'sentiment_subjectivity': 0,
                'positive_words': 0,
                'negative_words': 0
            }
    
    def extract_readability_features(self, text):
        """
        Extract readability metrics
        
        Args:
            text (str): Input text
            
        Returns:
            dict: Dictionary of readability features
        """
        if pd.isna(text) or len(str(text).strip()) == 0:
            return {
                'flesch_reading_ease': 0,
                'flesch_kincaid_grade': 0,
                'avg_sentence_length': 0,
                'avg_syllables_per_word': 0
            }
        
        try:
            text = str(text)
            words = text.split()
            sentences = re.findall(r'[.!?]+', text)
            
            if len(words) == 0 or len(sentences) == 0:
                return {
                    'flesch_reading_ease': 0,
                    'flesch_kincaid_grade': 0,
                    'avg_sentence_length': 0,
                    'avg_syllables_per_word': 0
                }
            
            # Count syllables (simplified method)
            def count_syllables(word):
                word = word.lower()
                vowels = 'aeiouy'
                syllable_count = 0
                previous_was_vowel = False
                
                for char in word:
                    if char in vowels:
                        if not previous_was_vowel:
                            syllable_count += 1
                        previous_was_vowel = True
                    else:
                        previous_was_vowel = False
                
                # Handle silent 'e'
                if word.endswith('e') and syllable_count > 1:
                    syllable_count -= 1
                
                return max(1, syllable_count)
            
            total_syllables = sum(count_syllables(word) for word in words)
            
            # Calculate metrics
            avg_sentence_length = len(words) / len(sentences)
            avg_syllables_per_word = total_syllables / len(words)
            
            # Flesch Reading Ease
            flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
            flesch_score = max(0, min(100, flesch_score))
            
            # Flesch-Kincaid Grade Level
            fk_grade = (0.39 * avg_sentence_length) + (11.8 * avg_syllables_per_word) - 15.59
            fk_grade = max(0, fk_grade)
            
            return {
                'flesch_reading_ease': flesch_score,
                'flesch_kincaid_grade': fk_grade,
                'avg_sentence_length': avg_sentence_length,
                'avg_syllables_per_word': avg_syllables_per_word
            }
        
        except Exception:
            return {
                'flesch_reading_ease': 0,
                'flesch_kincaid_grade': 0,
                'avg_sentence_length': 0,
                'avg_syllables_per_word': 0
            }
    
    def extract_structural_features(self, text):
        """
        Extract structural and style features
        
        Args:
            text (str): Input text
            
        Returns:
            dict: Dictionary of structural features
        """
        if pd.isna(text) or len(str(text).strip()) == 0:
            return {
                'has_urls': 0,
                'has_mentions': 0,
                'has_hashtags': 0,
                'has_email': 0,
                'capitalized_words_ratio': 0,
                'repeated_chars': 0,
                'ellipsis_count': 0
            }
        
        text = str(text)
        words = text.split()
        
        # URL detection
        has_urls = 1 if re.search(r'http\S+|www\S+', text) else 0
        
        # Social media features
        has_mentions = 1 if re.search(r'@\w+', text) else 0
        has_hashtags = 1 if re.search(r'#\w+', text) else 0
        
        # Email detection
        has_email = 1 if re.search(r'\S+@\S+', text) else 0
        
        # Capitalization analysis
        if words:
            capitalized_words = sum(1 for word in words if word.isupper() and len(word) > 1)
            capitalized_words_ratio = capitalized_words / len(words)
        else:
            capitalized_words_ratio = 0
        
        # Repeated characters (like "sooooo")
        repeated_chars = len(re.findall(r'(.)\1{2,}', text))
        
        # Ellipsis count
        ellipsis_count = text.count('...')
        
        return {
            'has_urls': has_urls,
            'has_mentions': has_mentions,
            'has_hashtags': has_hashtags,
            'has_email': has_email,
            'capitalized_words_ratio': capitalized_words_ratio,
            'repeated_chars': repeated_chars,
            'ellipsis_count': ellipsis_count
        }
    
    def extract_credibility_indicators(self, text):
        """
        Extract features specifically related to credibility signals
        
        Args:
            text (str): Input text
            
        Returns:
            dict: Dictionary of credibility indicator features
        """
        if pd.isna(text) or len(str(text).strip()) == 0:
            return {
                'clickbait_words': 0,
                'authority_words': 0,
                'uncertainty_words': 0,
                'emotion_words': 0,
                'numbers_present': 0
            }
        
        text = str(text).lower()
        words = set(text.split())
        
        # Clickbait indicators
        clickbait_terms = {
            'breaking', 'urgent', 'shocking', 'unbelievable', 'amazing', 'incredible',
            'secret', 'revealed', 'exposed', 'trick', 'hate', 'love', 'believe',
            'won\'t', 'can\'t', 'must', 'everyone', 'nobody', 'click', 'here'
        }
        
        # Authority indicators
        authority_terms = {
            'study', 'research', 'university', 'professor', 'doctor', 'expert',
            'scientist', 'journal', 'published', 'peer-reviewed', 'evidence',
            'data', 'analysis', 'report', 'official', 'government'
        }
        
        # Uncertainty indicators
        uncertainty_terms = {
            'might', 'could', 'possibly', 'perhaps', 'maybe', 'likely',
            'probably', 'seems', 'appears', 'suggests', 'indicates'
        }
        
        # Emotional language
        emotion_terms = {
            'amazing', 'terrible', 'shocking', 'disgusting', 'outrageous',
            'incredible', 'unbelievable', 'devastating', 'tragic', 'fantastic'
        }
        
        # Count occurrences
        clickbait_count = sum(1 for word in words if word in clickbait_terms)
        authority_count = sum(1 for word in words if word in authority_terms)
        uncertainty_count = sum(1 for word in words if word in uncertainty_terms)
        emotion_count = sum(1 for word in words if word in emotion_terms)
        
        # Numbers present (indicating specific data/statistics)
        numbers_present = 1 if re.search(r'\d+', text) else 0
        
        return {
            'clickbait_words': clickbait_count,
            'authority_words': authority_count,
            'uncertainty_words': uncertainty_count,
            'emotion_words': emotion_count,
            'numbers_present': numbers_present
        }
    
    def create_feature_matrix(self, df, text_column='processed_text'):
        """
        Create comprehensive feature matrix from all feature extraction methods
        
        Args:
            df (pd.DataFrame): Input dataframe
            text_column (str): Name of text column
            
        Returns:
            pd.DataFrame: Feature matrix
        """
        print("Extracting features...")
        
        features_list = []
        
        for idx, text in df[text_column].items():
            features = {}
            
            # Extract all types of features
            features.update(self.extract_linguistic_features(text))
            features.update(self.extract_sentiment_features(text))
            features.update(self.extract_readability_features(text))
            features.update(self.extract_structural_features(text))
            features.update(self.extract_credibility_indicators(text))
            
            features_list.append(features)
        
        feature_df = pd.DataFrame(features_list, index=df.index)
        
        # Store feature names for later use
        self.feature_names = feature_df.columns.tolist()
        
        print(f"Extracted {len(self.feature_names)} features")
        
        return feature_df
    
    def get_feature_importance(self, feature_matrix, labels, method='mutual_info'):
        """
        Calculate feature importance scores
        
        Args:
            feature_matrix (pd.DataFrame): Feature matrix
            labels (pd.Series): Target labels
            method (str): Method for importance calculation
            
        Returns:
            pd.DataFrame: Feature importance scores
        """
        from sklearn.feature_selection import mutual_info_classif
        from sklearn.ensemble import RandomForestClassifier
        
        if method == 'mutual_info':
            importance_scores = mutual_info_classif(feature_matrix, labels)
        elif method == 'random_forest':
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(feature_matrix, labels)
            importance_scores = rf.feature_importances_
        else:
            raise ValueError("Method must be 'mutual_info' or 'random_forest'")
        
        importance_df = pd.DataFrame({
            'feature': feature_matrix.columns,
            'importance': importance_scores
        }).sort_values('importance', ascending=False)
        
        return importance_df