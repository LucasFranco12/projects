import os
import time
import json
import requests
from typing import List, Dict
from urllib.parse import quote_plus
from semanticscholar import SemanticScholar
import PyPDF2
import io

class PaperCollector:
    def __init__(self):
        self.output_dir = "research_papers"
        self.sch = SemanticScholar()
        self.seen_papers = set()
        self.min_citations = 10  # Only collect papers with at least 10 citations
        
        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
    def sanitize_filename(self, title: str) -> str:
        """Convert title to valid filename."""
        return "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip()
        
    def save_paper_text(self, title: str, text: str):
        """Save paper text to file."""
        filename = self.sanitize_filename(title)[:100] + '.txt'
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8', errors='ignore') as f:
            f.write(text)
            
        print(f"Saved: {filename}")
        
    def search_semantic_scholar(self, query: str, limit: int = 50) -> List[Dict]:
        """Search papers on Semantic Scholar."""
        try:
            # Use the raw API endpoint for more control
            base_url = "http://api.semanticscholar.org/graph/v1/paper/search"
            params = {
                "query": query,
                "limit": limit,
                "fields": "title,abstract,url,year,authors,citationCount,openAccessPdf"
            }
            
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            papers = data.get('data', [])
            
            # Filter for open access papers with sufficient citations
            return [
                paper for paper in papers 
                if paper.get('openAccessPdf') and 
                paper.get('citationCount', 0) >= self.min_citations
            ]
            
        except Exception as e:
            print(f"Error searching Semantic Scholar: {str(e)}")
            return []
            
    def extract_text_from_pdf_url(self, pdf_url: str) -> str:
        """Download and extract text from PDF URL."""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(pdf_url, headers=headers, timeout=30)
            response.raise_for_status()
            
            pdf_file = io.BytesIO(response.content)
            
            try:
                reader = PyPDF2.PdfReader(pdf_file)
                text = ""
                for page in reader.pages:
                    try:
                        text += page.extract_text() + "\n"
                    except Exception as e:
                        print(f"Error extracting page text: {str(e)}")
                        continue
                return text
            except Exception as e:
                print(f"Error reading PDF: {str(e)}")
                return ""
                
        except Exception as e:
            print(f"Error downloading PDF: {str(e)}")
            return ""
            
    def collect_papers(self):
        """Collect papers from various sources."""
        search_topics = [
            # Original topics remain valuable
            "exercise physiology methodology",
            "sports nutrition research evidence",
            "resistance training science",
            "muscle hypertrophy mechanisms",
            "weight loss metabolism studies",
            
            # More specific training topics
            "eccentric training adaptation",
            "plyometric training effects",
            "blood flow restriction training",
            "deload training benefits",
            "tempo training muscle growth",
            "cluster set training research",
            "drop set training effectiveness",
            "rest pause training study",
            
            # Detailed nutrition topics
            "meal frequency metabolism",
            "fasted training adaptation",
            "ketogenic diet athletes",
            "intermittent fasting exercise",
            "branched chain amino acids",
            "creatine supplementation effects",
            "beta alanine performance",
            "caffeine exercise performance",
            
            # Recovery and adaptation
            "muscle protein synthesis window",
            "training frequency recovery",
            "deload programming effects",
            "supercompensation training",
            "active recovery benefits",
            "foam rolling effectiveness",
            "compression garment recovery",
            
            # Special populations
            "female athlete nutrition",
            "masters athletes training",
            "adolescent strength training",
            "pregnancy exercise guidelines",
            "elderly resistance training",
            
            # Mental aspects
            "exercise adherence psychology",
            "motivation training success",
            "mindfulness athletic performance",
            "cognitive performance exercise",
            
            # Advanced concepts
            "DNA training response",
            "epigenetic exercise adaptation",
            "muscle fiber type training",
            "mitochondrial adaptation exercise",
            "anabolic signaling pathways",
            "satellite cell activation",
            
            # Measurement and testing
            "force velocity profiling",
            "rate force development",
            "vo2max testing protocols",
            "biomarker exercise response",
            "heart rate variability training"
        ]
        
        total_papers = 0
        for topic in search_topics:
            print(f"\nSearching for papers on: {topic}")
            papers = self.search_semantic_scholar(topic)
            print(f"Found {len(papers)} relevant papers")
            
            for paper in papers:
                paper_id = paper.get('paperId')
                if paper_id in self.seen_papers:
                    continue
                    
                self.seen_papers.add(paper_id)
                title = paper.get('title', '')
                pdf_url = paper.get('openAccessPdf', {}).get('url')
                
                if pdf_url and title:
                    print(f"\nProcessing: {title}")
                    print(f"Citations: {paper.get('citationCount', 0)}")
                    text = self.extract_text_from_pdf_url(pdf_url)
                    
                    if text:
                        # Add metadata at the top of the file
                        metadata = f"""Title: {title}
Authors: {', '.join(author.get('name', '') for author in paper.get('authors', []))}
Year: {paper.get('year', 'N/A')}
Citations: {paper.get('citationCount', 0)}
Source: Semantic Scholar
URL: {pdf_url}

Abstract:
{paper.get('abstract', 'No abstract available.')}

Full Text:
"""
                        full_content = metadata + text
                        self.save_paper_text(title, full_content)
                        total_papers += 1
                        
                # Be nice to the API
                time.sleep(3)
                
        print(f"\nCollection complete! Downloaded {total_papers} papers.")

if __name__ == "__main__":
    collector = PaperCollector()
    collector.collect_papers() 