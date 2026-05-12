import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class ModelTrainer:
    def __init__(self):
        self.models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Multinomial Naive Bayes': MultinomialNB(),
            'Linear SVM': LinearSVC(random_state=42, max_iter=2000),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42, n_estimators=100)
        }
        self.best_model_name = None
        self.best_model = None
        
    def train_and_evaluate(self, X_train, y_train, X_test, y_test):
        results = {}
        best_f1 = 0
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            # We must handle negative values for MultinomialNB because chi2 / feature union could produce them?
            # Actually, TFIDF is non-negative. Linguistic features are non-negative.
            # BUT if we standardized (which we aren't), we could get negatives.
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            macro_f1 = f1_score(y_test, y_pred, average='macro')
            report = classification_report(y_test, y_pred, output_dict=True)
            cm = confusion_matrix(y_test, y_pred)
            
            results[name] = {
                'model': model,
                'macro_f1': macro_f1,
                'report': report,
                'confusion_matrix': cm
            }
            
            if macro_f1 > best_f1:
                best_f1 = macro_f1
                self.best_model_name = name
                self.best_model = model
                
        return results

    def get_top_features(self, model, feature_names, top_n=10):
        """
        Extract top discriminating features per class.
        Assumes classes are ordered as in model.classes_
        """
        top_features = {}
        classes = model.classes_
        
        if hasattr(model, 'coef_'):
            coef = model.coef_
            # For binary classification coef_ is 1D, for multi-class it's (n_classes, n_features)
            if len(classes) == 2:
                # Logistic Regression binary
                top_positive = np.argsort(coef[0])[-top_n:]
                top_negative = np.argsort(coef[0])[:top_n]
                top_features[classes[1]] = [feature_names[i] for i in top_positive]
                top_features[classes[0]] = [feature_names[i] for i in top_negative]
            else:
                for i, class_label in enumerate(classes):
                    top_indices = np.argsort(coef[i])[-top_n:]
                    top_features[class_label] = [feature_names[idx] for idx in reversed(top_indices)]
                    
        elif hasattr(model, 'feature_log_prob_'):
            # Naive Bayes
            log_prob = model.feature_log_prob_
            for i, class_label in enumerate(classes):
                top_indices = np.argsort(log_prob[i])[-top_n:]
                top_features[class_label] = [feature_names[idx] for idx in reversed(top_indices)]
                
        elif hasattr(model, 'feature_importances_'):
            # Tree-based (Gradient Boosting) -> Global feature importance
            importances = model.feature_importances_
            top_indices = np.argsort(importances)[-top_n:]
            top_features['Global Importance'] = [feature_names[idx] for idx in reversed(top_indices)]
            
        return top_features

def save_pipeline(pipeline, filename="pipeline.pkl"):
    with open(filename, 'wb') as f:
        pickle.dump(pipeline, f)
    print(f"Pipeline saved to {filename}")

def load_pipeline(filename="pipeline.pkl"):
    with open(filename, 'rb') as f:
        pipeline = pickle.load(f)
    return pipeline
