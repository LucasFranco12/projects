from flask import Flask, render_template, request, redirect, url_for
import os
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from youtube_transcript_api import YouTubeTranscriptApi
import textwrap
import re


app = Flask(__name__)

# Ensure the templates directory exists
if not os.path.exists('templates'):
    os.makedirs('templates')

# Create the templates/base.html file
with open('templates/base.html', 'w') as f:
    f.write('''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}{% endblock %} - AI YouTube Quiz</title>
    <style>
        body {
            background-color: red;
            color: white;
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
        }
        .container {
            text-align: center;
            background-color: rgba(0, 0, 0, 0.5);
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.3);
        }
        input[type="text"] {
            padding: 10px;
            width: 300px;
            margin: 10px 0;
            border: none;
            border-radius: 5px;
        }
        button {
            background-color: white;
            color: red;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            font-weight: bold;
        }
        button:hover {
            background-color: #f0f0f0;
        }
    </style>
</head>
<body>
    {% block content %}{% endblock %}
</body>
</html>
''')

# Create the templates/home.html file
with open('templates/home.html', 'w') as f:
    f.write('''
{% extends "base.html" %}

{% block title %}Home{% endblock %}

{% block content %}
<div class="container">
    <h1>AI YouTube Quiz</h1>
    <form action="{{ url_for('process_url') }}" method="POST">
        <input type="text" name="youtube_url" placeholder="Enter YouTube URL" required>
        <br>
        <button type="submit">Generate Quiz</button>
    </form>
</div>
{% endblock %}
''')



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
        """)
        
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | model
        return chain.invoke({"transcript": transcript})
    except Exception as e:
        return str(e)


def parse_quiz_string(quiz_text):
    """
    Parse a quiz string into a structured dictionary containing questions, options, and answers.
    
    Args:
        quiz_text (str): The input quiz text string
        
    Returns:
        dict: A dictionary with 'questions', 'options', and 'answers' lists
    """
    # Initialize the result dictionary
    quiz_dict = {
        'questions': [],
        'options': [],
        'answers': []
    }
    
    # Split into questions and answers sections
    sections = quiz_text.split("Answers:")
    questions_section = sections[0].strip()
    answers_section = sections[1].strip()
    
    # Process questions section
    current_options = []
    
    # Split into individual questions
    questions_list = questions_section.split('\n\n')
    
    for question_block in questions_list:
        lines = question_block.strip().split('\n')
        
        # Extract question (remove question number)
        question = lines[0].split('. ', 1)[1].strip()
        quiz_dict['questions'].append(question)
        
        # Extract options
        options = []
        for option_line in lines[1:]:
            option_line = option_line.strip()
            if option_line.startswith('   '):  # Handle indentation
                option = option_line.strip()
                if option.startswith(('A.', 'B.', 'C.', 'D.')):
                    option_text = option[3:].strip()  # Remove A., B., etc.
                    options.append(option_text)
        
        quiz_dict['options'].append(options)
    
    # Process answers section
    for line in answers_section.split('\n'):
        line = line.strip()
        if line and line[0].isdigit():
            # Extract the answer letter (e.g., "A", "B", "C", "D")
            answer = line.split('. ')[1].split('.')[0]
            quiz_dict['answers'].append(answer)
    
    return quiz_dict


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/process', methods=['POST'])
def process_url():
    try:
        print("Starting process_url")  # First log
        url = request.form['youtube_url']
        print(f"Received URL: {url}")  # Log the URL
        
        video_id = extract_video_id(url)
        print(f"Extracted video ID: {video_id}")  # Log the video ID

        if not video_id:
            print("Invalid YouTube URL")  # Log invalid URL
            return render_template('home.html', error='Invalid YouTube URL')

        print("Getting transcript")  # Log before transcript
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        print("Got transcript")  # Log after transcript
        
        print("Cleaning transcript")  # Log before cleaning
        transcript = clean_and_combine_transcript(transcript_list)
        print("Cleaned transcript")  # Log after cleaning
        
        print("Generating quiz")  # Log before quiz generation
        quizN = generate_quiz(transcript)
        print("Generated quiz:", quizN)  # Log the raw quiz
        
        print("Parsing quiz")  # Log before parsing
        quiz = parse_quiz_string(quizN)
        print("Parsed quiz:", quiz)  # Log the parsed quiz

        return render_template('quiz.html', quiz=quiz)
    except Exception as e:
        print(f"Error occurred: {str(e)}")  # Log any errors
        print(f"Error type: {type(e)}")  # Log the type of error
        import traceback
        print(traceback.format_exc())  # Print the full error traceback
        return render_template('home.html', error=str(e))


if __name__ == '__main__':
    app.run(debug=True)