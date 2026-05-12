import os
import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from preprocessing import preprocess_text
from features import build_feature_pipeline, get_feature_selector
from model import ModelTrainer, save_pipeline

def find_dataset_file(path):
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.csv'):
                return os.path.join(root, file)
    return None

def load_data():
    print("Downloading dataset...")
    path = kagglehub.dataset_download("jillanisofttech/amazon-product-reviews")
    csv_file = find_dataset_file(path)
    
    if not csv_file:
        raise FileNotFoundError("Could not find a CSV file in the downloaded dataset.")
        
    print(f"Loading data from {csv_file}...")
    df = pd.read_csv(csv_file)
    
    text_col = None
    rating_col = None
    
    potential_text_cols = ['reviews.text', 'text', 'review_text', 'ReviewText', 'Review']
    potential_rating_cols = ['reviews.rating', 'rating', 'review_rating', 'Score', 'Rating']
    
    for col in df.columns:
        if col.lower() in [c.lower() for c in potential_text_cols] and not text_col:
            text_col = col
        if col.lower() in [c.lower() for c in potential_rating_cols] and not rating_col:
            rating_col = col
            
    if not text_col or not rating_col:
        text_col = text_col or df.columns[0]
        rating_col = rating_col or df.columns[1]
        
    df = df.dropna(subset=[text_col, rating_col])
    
    def map_sentiment(rating):
        try:
            r = float(rating)
            if r >= 4: return 'Positive'
            elif r == 3: return 'Neutral'
            else: return 'Negative'
        except:
            return 'Neutral'
            
    df['Sentiment'] = df[rating_col].apply(map_sentiment)
    
    if len(df) > 50000:
        print("Sampling dataset to 50,000 rows for faster training...")
        df = df.sample(50000, random_state=42)
        
    # Calculate review lengths
    df['Review_Length'] = df[text_col].astype(str).apply(lambda x: len(x.split()))
    
    return df, text_col

def plot_dataset_visuals(df, text_col):
    os.makedirs('visuals', exist_ok=True)
    
    # 1. Class Distribution
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x='Sentiment', hue='Sentiment', legend=False, order=['Positive', 'Neutral', 'Negative'], palette=['#2ecc71', '#95a5a6', '#e74c3c'])
    plt.title('Class Distribution (Amazon Reviews)')
    plt.ylabel('Number of Reviews')
    plt.savefig('visuals/class_dist.png')
    plt.close()
    
    # 2. Review Length Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='Review_Length', bins=50, kde=True, color='purple')
    plt.title('Review Length Distribution (Word Count)')
    plt.xlim(0, df['Review_Length'].quantile(0.95)) # Cap at 95th percentile for better visualization
    plt.xlabel('Number of Words')
    plt.ylabel('Frequency')
    plt.savefig('visuals/review_len_dist.png')
    plt.close()

def plot_evaluation_visuals(results, top_features=None):
    os.makedirs('visuals', exist_ok=True)
    
    # 1. Model Comparison Chart
    plt.figure(figsize=(10, 6))
    models = list(results.keys())
    f1_scores = [res['macro_f1'] for res in results.values()]
    sns.barplot(x=models, y=f1_scores, palette='viridis')
    plt.title('Model Comparison (Macro F1-Score)')
    plt.ylabel('Macro F1-Score')
    plt.ylim(0, 1)
    for i, v in enumerate(f1_scores):
        plt.text(i, v + 0.02, f"{v:.3f}", ha='center', va='bottom')
    plt.savefig('visuals/model_comparison.png')
    plt.close()
    
    # 2. Confusion Matrices
    classes = ['Negative', 'Neutral', 'Positive'] # typical sorted order
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    for idx, (name, res) in enumerate(results.items()):
        cm = res['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx], xticklabels=classes, yticklabels=classes)
        axes[idx].set_title(f'Confusion Matrix: {name}')
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig('visuals/confusion_matrices.png')
    plt.close()
    
    # 3. Top TF-IDF Features
    if top_features and 'Positive' in top_features and 'Negative' in top_features:
        plt.figure(figsize=(14, 6))
        
        plt.subplot(1, 2, 1)
        pos_feats = top_features['Positive'][:20]
        sns.barplot(x=[1]*20, y=pos_feats, color='#2ecc71')
        plt.title('Top 20 Features (Positive)')
        
        plt.subplot(1, 2, 2)
        neg_feats = top_features['Negative'][:20]
        sns.barplot(x=[1]*20, y=neg_feats, color='#e74c3c')
        plt.title('Top 20 Features (Negative)')
        
        plt.tight_layout()
        plt.savefig('visuals/top_features.png')
        plt.close()

def main():
    df, text_col = load_data()
    
    print("Generating dataset visualisations...")
    plot_dataset_visuals(df, text_col)
    
    X_raw = df[text_col].tolist()
    y = df['Sentiment'].tolist()
    
    print("Preprocessing text...")
    X_clean = [preprocess_text(str(text)) for text in X_raw]
    
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X_clean, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Building Feature Pipeline...")
    feature_union = build_feature_pipeline()
    feature_selector = get_feature_selector(method='chi2', k=3000)
    
    feature_pipeline = Pipeline([
        ('features', feature_union),
        ('selector', feature_selector)
    ])
    
    print("Extracting and selecting features...")
    X_train_features = feature_pipeline.fit_transform(X_train, y_train)
    X_test_features = feature_pipeline.transform(X_test)
    
    print(f"Feature matrix shape after selection: {X_train_features.shape}")
    
    trainer = ModelTrainer()
    results = trainer.train_and_evaluate(X_train_features, y_train, X_test_features, y_test)
    
    # Top features extraction
    top_features_dict = None
    try:
        tfidf = feature_pipeline.named_steps['features'].transformer_list[0][1]
        tfidf_names = tfidf.get_feature_names_out()
        ling_names = ['word_count', 'avg_word_length', 'punct_density', 'neg_count']
        all_features = list(tfidf_names) + ling_names
        
        selector = feature_pipeline.named_steps['selector']
        selected_mask = selector.get_support()
        selected_features = [all_features[i] for i in range(len(all_features)) if selected_mask[i]]
        
        top_features_dict = trainer.get_top_features(trainer.best_model, selected_features, top_n=20)
    except Exception as e:
        print("Could not extract feature names for analysis:", e)
    
    print("Generating evaluation visualisations...")
    plot_evaluation_visuals(results, top_features_dict)
    
    # Save the full pipeline
    final_pipeline = Pipeline([
        ('features', feature_union),
        ('selector', feature_selector),
        ('classifier', trainer.best_model)
    ])
    
    save_pipeline(final_pipeline, 'pipeline.pkl')
    print("Training complete! Visuals saved to /visuals directory.")

if __name__ == "__main__":
    main()
