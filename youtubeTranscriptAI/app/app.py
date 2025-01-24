from flask import Flask, render_template, request, jsonify
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from youtube_transcript_api import YouTubeTranscriptApi
import textwrap
import re
import os

app = Flask(__name__)

def extract_video_id(url):
    """Extract the video ID from various forms of YouTube URLs."""
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
        r'(?:embed\/)([0-9A-Za-z_-]{11})',
        r'(?:youtu\.be\/)([0-9A-Za-z_-]{11})'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def clean_and_combine_transcript(transcript_list):
    """Clean and combine transcript segments into a single coherent string."""
    cleaned_texts = []
    for entry in transcript_list:
        text = entry['text']
        text = re.sub(r'\[\s*__\s*\]', '', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        if text:
            cleaned_texts.append(text)
    
    return ' '.join(cleaned_texts)

def generate_quiz(transcript):
    """Generate quiz using Ollama model."""
    try:
        model = OllamaLLM(model="llama3.1:latest")  # Change model as needed
        
        template = textwrap.dedent("""
            You are an AI assistant tasked with creating a quiz based on the following YouTube video transcript:

            {transcript}

            Please generate a quiz with 10 multiple-choice questions based on the content of this transcript. 
            Each question should have 4 options (A, B, C, D) with only one correct answer. 
            After the questions, provide the correct answers.

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

            Answers: 
            1. [Correct Answer with an explanation as why it was correct]
            2. [Correct Answer with an explanation as why it was correct]
            ... (continue for all 10 questions)
                                   
            PLEASE MAKE SURE IT FOLLOWS THIS FORMAT COMPLETLEY
        """)
        
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | model
        return chain.invoke({"transcript": transcript})
    except Exception as e:
        return str(e)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    try:
        url = request.form['url']
        video_id = extract_video_id(url)
        
        if not video_id:
            return jsonify({'error': 'Invalid YouTube URL'}), 400
        
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        transcript = clean_and_combine_transcript(transcript_list)
        quiz = generate_quiz(transcript)
        
        return jsonify({
            'success': True,
            'quiz': quiz
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)