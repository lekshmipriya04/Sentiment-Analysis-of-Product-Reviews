from model import load_pipeline
from preprocessing import preprocess_text

class SentimentPredictor:
    def __init__(self, model_path="pipeline.pkl"):
        try:
            self.pipeline = load_pipeline(model_path)
        except FileNotFoundError:
            raise Exception(f"Model file '{model_path}' not found. Please train the model first.")
            
    def predict(self, texts):
        """
        Predicts the sentiment for a list of strings.
        Returns a list of predictions ('Positive', 'Neutral', 'Negative').
        """
        if isinstance(texts, str):
            texts = [texts]
            
        # 1. Preprocess the raw texts
        clean_texts = [preprocess_text(text) for text in texts]
        
        # 2. Predict using the full pipeline (feature extraction, selection, classification)
        predictions = self.pipeline.predict(clean_texts)
        
        return predictions.tolist()

if __name__ == "__main__":
    predictor = SentimentPredictor()
    test_comments = [
        "This product is absolutely amazing! I love it so much.",
        "It's okay, nothing special but it works.",
        "Terrible experience, broke after one day. Do not buy!"
    ]
    
    preds = predictor.predict(test_comments)
    for comment, pred in zip(test_comments, preds):
        print(f"[{pred}] {comment}")
