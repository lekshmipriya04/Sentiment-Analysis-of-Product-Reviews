import string
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif

class LinguisticFeaturesExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.negation_words = {'not', 'no', 'never', 'none', 'neither', 'nor', 'cannot', "isn't", "aren't", "wasn't", "weren't", "hasn't", "haven't", "hadn't", "doesn't", "don't", "didn't", "won't", "wouldn't", "shan't", "shouldn't", "can't", "couldn't", "mustn't"}

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features = []
        for text in X:
            if not isinstance(text, str):
                text = ""
                
            # Sentence length (word count)
            words = text.split()
            word_count = len(words)
            
            # Average word length
            if word_count > 0:
                avg_word_length = sum(len(word) for word in words) / word_count
            else:
                avg_word_length = 0
                
            # Punctuation density
            char_count = len(text)
            if char_count > 0:
                punct_count = sum(1 for char in text if char in string.punctuation)
                punct_density = punct_count / char_count
            else:
                punct_density = 0
                
            # Negation word count
            text_lower = text.lower()
            neg_count = sum(1 for word in words if word.lower() in self.negation_words)
            
            features.append([word_count, avg_word_length, punct_density, neg_count])
            
        return np.array(features)

def build_feature_pipeline(max_tfidf_features=10000, use_feature_selection=True, selection_method='chi2', k_best=3000):
    """
    Builds the feature extraction pipeline combining TF-IDF and Linguistic features.
    Optionally applies feature selection.
    """
    # TF-IDF Vectorizer
    # Note: Preprocessing will be done beforehand, so we use a standard vectorizer here
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2), # unigrams + bigrams
        max_features=max_tfidf_features,
        min_df=5,
        max_df=0.9
    )
    
    # Linguistic Features
    ling_features = LinguisticFeaturesExtractor()
    
    # Combine features
    combined_features = FeatureUnion([
        ('tfidf', tfidf),
        ('linguistic', ling_features)
    ])
    
    return combined_features

def get_feature_selector(method='chi2', k=3000):
    if method == 'chi2':
        return SelectKBest(score_func=chi2, k=k)
    elif method == 'mutual_info':
        # mutual_info_classif can be slow, so k=3000 is reasonable
        return SelectKBest(score_func=mutual_info_classif, k=k)
    else:
        raise ValueError("Method must be 'chi2' or 'mutual_info'")
