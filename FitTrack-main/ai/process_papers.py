import os
import re
from typing import List, Dict
import json

class PaperProcessor:
    def __init__(self, input_dir: str = "research_papers"):
        self.input_dir = input_dir
        
    def clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove citations [1], [2,3], etc.
        text = re.sub(r'\[\d+(?:,\s*\d+)*\]', '', text)
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        return text.strip()
        
    def extract_sections(self, content: str) -> Dict[str, str]:
        """Extract different sections from the paper."""
        sections = {
            "abstract": "",
            "introduction": "",
            "methods": "",
            "results": "",
            "discussion": "",
            "conclusion": ""
        }
        
        # Simple section detection (can be improved)
        current_section = None
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip().lower()
            if any(section in line for section in sections.keys()):
                for section in sections.keys():
                    if section in line:
                        current_section = section
                        break
            elif current_section and line:
                sections[current_section] += line + ' '
                
        return {k: self.clean_text(v) for k, v in sections.items()}
        
    def process_papers(self) -> List[Dict]:
        """Process all papers in the input directory."""
        processed_papers = []
        
        for filename in os.listdir(self.input_dir):
            if filename.endswith('.txt'):
                file_path = os.path.join(self.input_dir, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                    sections = self.extract_sections(content)
                    processed_papers.append({
                        "title": filename.replace('.txt', ''),
                        "sections": sections,
                        "full_text": self.clean_text(content)
                    })
                    print(f"Successfully processed {filename}")
                except Exception as e:
                    print(f"Error processing {filename}: {str(e)}")
                    
        return processed_papers
        
    def save_processed_papers(self, output_file: str = "processed_papers.json"):
        """Save processed papers to a JSON file."""
        processed_papers = self.process_papers()
        output_path = os.path.join(self.input_dir, output_file)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(processed_papers, f, indent=2)
            
        print(f"Saved processed papers to {output_path}")

if __name__ == "__main__":
    processor = PaperProcessor()
    processor.save_processed_papers() 