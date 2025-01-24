from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import textwrap
from youtube_transcript_api import YouTubeTranscriptApi
import re


def extract_video_id(url):
    """Extract the video ID from various forms of YouTube URLs."""
    # Regular expressions for different YouTube URL formats
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',  # Standard and shared URLs
        r'(?:embed\/)([0-9A-Za-z_-]{11})',   # Embed URLs
        r'(?:youtu\.be\/)([0-9A-Za-z_-]{11})'  # Shortened URLs
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def clean_and_combine_transcript(transcript_list):
    """
    Clean and combine transcript segments into a single coherent string.
    """
    # Extract just the text from each segment and clean it
    cleaned_texts = []
    for entry in transcript_list:
        # Get the text and clean it
        text = entry['text']
        # Remove special characters and extra whitespace
        text = re.sub(r'\[\s*__\s*\]', '', text)  # Remove [__] patterns
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        text = text.strip()
        if text:  # Only add non-empty strings
            cleaned_texts.append(text)
    
    # Join all texts with proper spacing
    full_transcript = ' '.join(cleaned_texts)
    return full_transcript


# Initialize the Ollama model
model = OllamaLLM(model="llama3.1:70b")

# Get the YouTube transcript from user input
url = input("Please enter the Youtube URL: ")


# Replace 'YOUR_VIDEO_ID' with the actual video ID
video_id = extract_video_id(url)
transcript_list = YouTubeTranscriptApi.get_transcript(video_id)

transcript = clean_and_combine_transcript(transcript_list)
print(transcript)


# Define the prompt template for quiz generation
template = textwrap.dedent("""
    You are an AI assistant tasked with creating a quiz based on the following YouTube video transcript:

    {transcript}

    Please generate a quiz with 10 multiple-choice questions based on the content of this transcript. Each question should have 4 options (A, B, C, D) with only one correct answer. After the questions, provide the correct answers.

    Format the quiz as follows:
    1. [Question 1]
       A. [Option A]
       B. [Option B]
       C. [Option C]
       D. [Option D]

    2. [Question 2]
       A. [Option A]
       B. [Option B]
       C. [Option C]
       D. [Option D]

    ... (continue for all 10 questions)

    Answers: PLEASE DO NOT FORGOT TO INCLUDE THIS :)
    1. [Correct Answer with an explanation as why it was correct]
    2. [Correct Answer with an explanation as why it was correct]
    ... (continue for all 10 questions)
""")

# Create the prompt
prompt = ChatPromptTemplate.from_template(template)

# Create the chain
chain = prompt | model

# Generate the quiz
response = chain.invoke({"transcript": transcript})

# Print the generated quiz
print("\nGenerated Quiz:\n")
print(response)