import re
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

def extract_video_id(url):
    """
    Extracts the YouTube video ID from a given URL.
    Handles various formats like:
    - https://www.youtube.com/watch?v=abc12345678
    - https://youtu.be/abc12345678
    - https://www.youtube.com/embed/abc12345678
    """
    pattern = r'(?:v=|\/)([0-9A-Za-z_-]{11}).*'
    match = re.search(pattern, url)
    if match:
        return match.group(1)
    return None

def get_youtube_comments(video_id, api_key, max_comments=200):
    """
    Fetches comments from a YouTube video using the Data API v3.
    """
    if not api_key:
        raise ValueError("YouTube API key is required.")
        
    try:
        youtube = build('youtube', 'v3', developerKey=api_key)
        
        comments = []
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100, # Max allowed per request is 100
            textFormat="plainText"
        )
        
        while request and len(comments) < max_comments:
            response = request.execute()
            
            for item in response['items']:
                comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
                comments.append(comment)
                
            request = youtube.commentThreads().list_next(request, response)
            
        return comments[:max_comments]
        
    except HttpError as e:
        if e.resp.status == 403:
            raise Exception("API Quota Exceeded or Invalid API Key.")
        elif e.resp.status == 404:
            raise Exception("Video not found. Please check the URL.")
        else:
            raise Exception(f"YouTube API Error: {e}")
    except Exception as e:
        raise Exception(f"An unexpected error occurred: {e}")
