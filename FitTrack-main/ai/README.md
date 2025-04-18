# AI Trainer Module

This module contains scripts for training and running a personalized fitness AI trainer using GPT4All and research papers.

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Download the GPT4All model:
The script will automatically download the model on first run, or you can manually download it from the GPT4All website.

3. Prepare your research papers:
- Place your research papers in the `research_papers` directory
- Files should be in .txt format
- Each paper should have clear section headers (Abstract, Introduction, Methods, etc.)

## Usage

1. Process research papers:
```bash
python process_papers.py
```
This will clean and structure your research papers, saving the processed data to `research_papers/processed_papers.json`.

2. Train the model:
```bash
python train_model.py
```
This will:
- Load the processed research papers
- Create training prompts
- Save embeddings to the vector database
- Fine-tune the model (when implemented)

## Integration with the App

The `FitnessTrainer` class in `train_model.py` provides methods for:
- Loading and processing research papers
- Training the model
- Generating personalized responses based on user data

Example usage in your app:
```python
from train_model import FitnessTrainer

trainer = FitnessTrainer()
trainer.load_research_papers()

# Get personalized response
user_data = {
    "goals": "Weight loss",
    "weight": 80,
    "targetWeight": 70,
    "fitnessLevel": "Intermediate",
    "dietaryRestrictions": ["Vegetarian"]
}

response = trainer.generate_response(
    "What should my daily protein intake be?",
    user_data
)
```

## Notes

- The current implementation uses GPT4All for inference
- Training functionality is a placeholder and needs to be implemented based on your specific requirements
- The vector database implementation is basic and can be replaced with FAISS or similar for better performance
- Make sure to handle API rate limits and errors appropriately in production 