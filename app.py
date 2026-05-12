import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
from PIL import Image

from youtube_api import extract_video_id, get_youtube_comments
from predict import SentimentPredictor

st.set_page_config(page_title="YouTube Sentiment Analyzer", page_icon="📊", layout="wide")

# Custom CSS for better aesthetics
st.markdown("""
<style>
    .reportview-container { background: #f0f2f6; }
    .metric-card { background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); text-align: center; }
    .metric-value { font-size: 36px; font-weight: bold; color: #1f77b4; }
    .metric-label { font-size: 16px; color: #555; }
    .insight-text { font-style: italic; color: #666; font-size: 14px; background-color: #f8f9fa; padding: 10px; border-left: 4px solid #1f77b4; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    try:
        return SentimentPredictor("pipeline.pkl")
    except Exception as e:
        return None

def extract_top_keywords(comments, n=10):
    text = " ".join(comments).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    stopwords = {'the', 'and', 'is', 'to', 'in', 'it', 'of', 'for', 'a', 'this', 'that', 'i', 'my', 'you', 'on', 'with', 'was', 'as', 'are', 'not', 'have'}
    words = [w for w in words if len(w) > 3 and w not in stopwords]
    most_common = Counter(words).most_common(n)
    return pd.DataFrame(most_common, columns=['Word', 'Frequency'])

def main():
    st.title("🎥 YouTube Comment Sentiment Analyzer")
    st.markdown("Analyze public opinion on YouTube videos using Machine Learning.")
    
    predictor = load_model()
    
    # Sidebar for inputs
    with st.sidebar:
        st.header("Configuration")
        api_key = st.text_input("YouTube API Key", type="password", help="Get this from Google Cloud Console.")
        if not api_key:
            st.info("API Key is required to fetch comments.")
            
        st.markdown("---")
        if predictor:
            st.write("**Model Loaded Successfully!** ✅")
            st.write("Using Pipeline: TF-IDF + Linguistic Features")
        else:
            st.warning("⚠️ Model not found! Please run `train.py` first.")

    tab_app, tab_training = st.tabs(["Dashboard", "Training Metrics"])
    
    with tab_app:
        video_url = st.text_input("Enter YouTube Video URL:", placeholder="https://www.youtube.com/watch?v=...")
        
        if st.button("Analyze Sentiments", type="primary"):
            if not predictor:
                st.error("Model is not trained yet.")
                return
            if not api_key:
                st.error("Please provide a valid YouTube API key in the sidebar.")
                return
                
            video_id = extract_video_id(video_url)
            if not video_id:
                st.error("Invalid YouTube URL. Please check and try again.")
                return
                
            with st.spinner("Fetching comments and analyzing sentiments..."):
                try:
                    raw_comments = get_youtube_comments(video_id, api_key, max_comments=200)
                    if not raw_comments:
                        st.warning("No comments found for this video.")
                        return
                    
                    valid_comments = [c for c in raw_comments if len(c.strip()) > 5]
                    if not valid_comments:
                        st.warning("Only extremely short/spam comments were found.")
                        return
                        
                    predictions = predictor.predict(valid_comments)
                    df = pd.DataFrame({
                        'Comment': valid_comments,
                        'Sentiment': predictions,
                        'Length': [len(c.split()) for c in valid_comments]
                    })
                    
                    st.success(f"Successfully analyzed {len(df)} comments!")
                    
                    dist = df['Sentiment'].value_counts(normalize=True) * 100
                    pos_pct = dist.get('Positive', 0)
                    neu_pct = dist.get('Neutral', 0)
                    neg_pct = dist.get('Negative', 0)
                    
                    st.markdown("### 📊 Overall Public Opinion")
                    if pos_pct > 70: summary = "Highly Positive Response 🌟"
                    elif neg_pct > 50: summary = "Negative Feedback Trend 📉"
                    elif pos_pct > 40 and neg_pct > 30: summary = "Mixed Reactions ⚖️"
                    else: summary = "Neutral/Mixed Reactions 😐"
                    st.markdown(f"<h2 style='text-align: center;'>{summary}</h2>", unsafe_allow_html=True)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1: st.markdown(f"<div class='metric-card'><div class='metric-value'>{pos_pct:.1f}%</div><div class='metric-label'>Positive 😊</div></div>", unsafe_allow_html=True)
                    with col2: st.markdown(f"<div class='metric-card'><div class='metric-value'>{neu_pct:.1f}%</div><div class='metric-label'>Neutral 😐</div></div>", unsafe_allow_html=True)
                    with col3: st.markdown(f"<div class='metric-card'><div class='metric-value'>{neg_pct:.1f}%</div><div class='metric-label'>Negative 😡</div></div>", unsafe_allow_html=True)
                        
                    st.markdown("---")
                    
                    # 1. Sentiment Distribution (Pie Chart)
                    st.markdown("### 1. Sentiment Distribution")
                    fig, ax = plt.subplots(figsize=(6, 4))
                    colors = {'Positive': '#2ecc71', 'Neutral': '#95a5a6', 'Negative': '#e74c3c'}
                    counts = df['Sentiment'].value_counts()
                    ax.pie(counts, labels=counts.index, autopct='%1.1f%%', colors=[colors[x] for x in counts.index], startangle=90)
                    ax.axis('equal')
                    st.pyplot(fig)
                    st.markdown("<div class='insight-text'><b>Insight:</b> This chart shows the overall breakdown of viewer sentiment. It helps instantly quantify whether the video's reception is leaning towards a positive or negative consensus.</div>", unsafe_allow_html=True)
                    
                    st.markdown("---")
                    
                    # 2. Comment Length vs Sentiment (Boxplot)
                    st.markdown("### 2. Comment Length vs Sentiment")
                    fig, ax = plt.subplots(figsize=(8, 5))
                    sns.boxplot(data=df, x='Sentiment', y='Length', hue='Sentiment', legend=False, palette=colors, ax=ax, order=['Positive', 'Neutral', 'Negative'])
                    ax.set_title('Word Count Distribution across Sentiments')
                    ax.set_ylabel('Number of Words')
                    st.pyplot(fig)
                    st.markdown("<div class='insight-text'><b>Insight:</b> This boxplot analyzes behavior patterns. It reveals whether angry viewers write longer rants compared to happy viewers, or if neutral comments tend to be shorter.</div>", unsafe_allow_html=True)
                    
                    st.markdown("---")
                    
                    # 3. Word Frequency Visualization (Bar Charts)
                    st.markdown("### 3. Word Frequency Visualization")
                    col_pos, col_neg = st.columns(2)
                    
                    with col_pos:
                        pos_df = extract_top_keywords(df[df['Sentiment'] == 'Positive']['Comment'].tolist())
                        if not pos_df.empty:
                            fig, ax = plt.subplots(figsize=(6, 4))
                            sns.barplot(data=pos_df, x='Frequency', y='Word', color='#2ecc71', ax=ax)
                            ax.set_title('Top 10 Positive Words')
                            st.pyplot(fig)
                        else:
                            st.write("Not enough positive words.")
                            
                    with col_neg:
                        neg_df = extract_top_keywords(df[df['Sentiment'] == 'Negative']['Comment'].tolist())
                        if not neg_df.empty:
                            fig, ax = plt.subplots(figsize=(6, 4))
                            sns.barplot(data=neg_df, x='Frequency', y='Word', color='#e74c3c', ax=ax)
                            ax.set_title('Top 10 Negative Words')
                            st.pyplot(fig)
                        else:
                            st.write("Not enough negative words.")
                            
                    st.markdown("<div class='insight-text'><b>Insight:</b> These bar charts highlight the most frequent words in positive vs. negative comments. This helps identify exact features or topics driving the audience's emotional response.</div>", unsafe_allow_html=True)
                    
                    st.markdown("---")
                    
                    # Sample Comments
                    st.markdown("### 💬 Sample Comments")
                    tab1, tab2, tab3 = st.tabs(["Positive", "Neutral", "Negative"])
                    with tab1:
                        for _, row in df[df['Sentiment'] == 'Positive'].head(5).iterrows():
                            st.success(row['Comment'])
                    with tab2:
                        for _, row in df[df['Sentiment'] == 'Neutral'].head(5).iterrows():
                            st.info(row['Comment'])
                    with tab3:
                        for _, row in df[df['Sentiment'] == 'Negative'].head(5).iterrows():
                            st.error(row['Comment'])

                except Exception as e:
                    st.error(f"Error: {e}")

    with tab_training:
        st.header("Dataset & Model Evaluation Visualizations")
        st.write("These charts were generated during the training phase using the Amazon Product Reviews dataset.")
        
        vis_dir = 'visuals'
        if not os.path.exists(vis_dir):
            st.warning("Visualizations directory not found. Please run `train.py` to generate them.")
        else:
            st.markdown("### A. Dataset-Level Visualizations")
            
            # 1. Class Distribution
            img_path = os.path.join(vis_dir, 'class_dist.png')
            if os.path.exists(img_path):
                st.image(Image.open(img_path), use_column_width=True)
                st.markdown("<div class='insight-text'><b>Insight:</b> Shows the distribution of training classes. It matters because heavy class imbalance can cause the model to be biased toward the majority class (e.g., predicting 'Positive' too often).</div>", unsafe_allow_html=True)
                
            # 2. Review Length Distribution
            img_path = os.path.join(vis_dir, 'review_len_dist.png')
            if os.path.exists(img_path):
                st.image(Image.open(img_path), use_column_width=True)
                st.markdown("<div class='insight-text'><b>Insight:</b> Displays the distribution of review lengths. This helps us understand text variability and determine if our feature extractor needs to cap length to remove outliers.</div>", unsafe_allow_html=True)
                
            # 3. Top Features
            img_path = os.path.join(vis_dir, 'top_features.png')
            if os.path.exists(img_path):
                st.image(Image.open(img_path), use_column_width=True)
                st.markdown("<div class='insight-text'><b>Insight:</b> Extracts the top 20 TF-IDF features. It visualizes the raw keywords that the model relies heavily upon, proving that it learns semantic meaning rather than noise.</div>", unsafe_allow_html=True)
                
            st.markdown("---")
            st.markdown("### B. Model Evaluation Visualizations")
            
            # 4. Model Comparison
            img_path = os.path.join(vis_dir, 'model_comparison.png')
            if os.path.exists(img_path):
                st.image(Image.open(img_path), use_column_width=True)
                st.markdown("<div class='insight-text'><b>Insight:</b> Compares Logistic Regression, Naive Bayes, SVM, and Gradient Boosting by Macro F1-Score. This dictates which model is selected for production based on predictive balance.</div>", unsafe_allow_html=True)
                
            # 5. Confusion Matrices
            img_path = os.path.join(vis_dir, 'confusion_matrices.png')
            if os.path.exists(img_path):
                st.image(Image.open(img_path), use_column_width=True)
                st.markdown("<div class='insight-text'><b>Insight:</b> Heatmaps showing True Positives vs False Positives for all models. It highlights exactly where each model struggles (e.g., misclassifying 'Neutral' as 'Negative').</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
