import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

# Initialize components
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
# Keep negation words
negation_words = {'not', 'no', 'never', 'none', 'neither', 'nor', 'cannot', "isn't", "aren't", "wasn't", "weren't", "hasn't", "haven't", "hadn't", "doesn't", "don't", "didn't", "won't", "wouldn't", "shan't", "shouldn't", "can't", "couldn't", "mustn't"}
stop_words = stop_words - negation_words

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    
    # 1. Lowercase
    text = text.lower()
    
    # 2. Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # 3. Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)
    
    # 4. Remove special characters (retain punctuation for features: . , ! ? ' ")
    # Replace anything that isn't a letter, number, whitespace, or basic punctuation with a space
    text = re.sub(r'[^a-z0-9\s.,!?\'"]', ' ', text)
    
    # 5. Tokenization
    tokens = word_tokenize(text)
    
    # 6. Remove stopwords & 7. Lemmatization
    cleaned_tokens = []
    for token in tokens:
        # We don't remove punctuation tokens here if they are separate, because the prompt says 
        # "retain punctuation for features". If TFIDF vectorizer drops them, that's fine,
        # but our custom feature extractor might use the raw text anyway. 
        # Wait, if we keep punctuation as tokens, the TFIDF vectorizer (if we pass preprocessed text) 
        # might use them if its regex allows. Sklearn's default token pattern ignores punctuation.
        if token not in stop_words:
            cleaned_tokens.append(lemmatizer.lemmatize(token))
            
    # Join back to string for TfidfVectorizer (which expects strings)
    return " ".join(cleaned_tokens)

if __name__ == "__main__":
    sample = "I LOVED this product! But I did NOT like the packaging. <b>Check it out</b> at http://example.com #awesome"
    print("Original:", sample)
    print("Cleaned: ", preprocess_text(sample))
