import whisper
import os
from .logger import get_logger
logger = get_logger("video_to_text")
# Load model once at module level
model = whisper.load_model("tiny")  # base / small / medium / large

def transcribe_video(video_path):
    """
    Transcribe video using Whisper
    
    Args:
        video_path: Path to the video file
        
    Returns:
        str: Transcribed text
    """
    try:
        result = model.transcribe(video_path)
        return result["text"]
    except Exception as e:
        logger.error(f"Error transcribing video {video_path}: {str(e)}")
        raise

def transcribe_and_save(video_path, output_file="transcript.txt"):
    """
    Transcribe video and save to text file
    
    Args:
        video_path: Path to the video file
        output_file: Output text file path
        
    Returns:
        str: Path to the saved text file
    """
    transcript = transcribe_video(video_path)
    
    print(f"Transcription saved to {output_file}")
    return output_file

if __name__ == "__main__":
    VIDEO_PATH = r"C:\Users\PRATYUSH\OneDrive\Desktop\Voice-to-text\Videos\Aerosol.mp4"
    OUTPUT_TEXT_FILE = "transcript.txt"
    transcribe_and_save(VIDEO_PATH, OUTPUT_TEXT_FILE)
