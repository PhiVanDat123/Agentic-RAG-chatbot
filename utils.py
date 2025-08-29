from io import BytesIO
import PyPDF2
import docx
from typing import List

def extract_text_from_file(content: bytes, filename: str) -> str:
    """Extract text from uploaded files"""
    if filename.endswith('.txt'):
        return content.decode('utf-8')
    elif filename.endswith('.pdf'):
        pdf_reader = PyPDF2.PdfReader(BytesIO(content))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    elif filename.endswith('.docx'):
        doc = docx.Document(BytesIO(content))
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    else:
        raise ValueError(f"Unsupported file type: {filename}")

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks for better retrieval"""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
        
        if i + chunk_size >= len(words):
            break
            
    return chunks

def clean_text(text: str) -> str:
    """Clean and normalize text"""
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Remove special characters that might interfere with embeddings
    text = text.replace('\x00', '')  # Remove null bytes
    
    return text.strip()

def validate_file_type(filename: str) -> bool:
    """Validate if file type is supported"""
    supported_extensions = ['.txt', '.pdf', '.docx']
    return any(filename.lower().endswith(ext) for ext in supported_extensions)

def format_sources(sources: List[str]) -> str:
    """Format source list for display"""
    if not sources:
        return "No sources found."
    
    formatted = "Sources:\n"
    for i, source in enumerate(sources, 1):
        formatted += f"{i}. {source}\n"
    
    return formatted