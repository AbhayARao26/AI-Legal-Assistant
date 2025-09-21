"""
Utility functions for the Legal AI system
"""
import os
import hashlib
import mimetypes
from typing import List, Dict, Any, Optional
from datetime import datetime
import re


def validate_file_type(filename: str) -> bool:
    """Validate if file type is supported"""
    allowed_extensions = {'.pdf', '.docx', '.txt'}
    _, ext = os.path.splitext(filename.lower())
    return ext in allowed_extensions


def get_file_hash(file_path: str) -> str:
    """Generate MD5 hash of file for deduplication"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def clean_text(text: str) -> str:
    """Clean and normalize text content"""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove non-printable characters
    text = re.sub(r'[^\x20-\x7E\n]', '', text)
    # Normalize line breaks
    text = re.sub(r'\n+', '\n', text)
    return text.strip()


def extract_legal_entities(text: str) -> List[str]:
    """Extract legal entities from text using regex patterns"""
    entities = []
    
    # Common legal entity patterns
    patterns = [
        r'\b[A-Z][a-z]+ [A-Z][a-z]+ (?:LLC|Inc\.|Corp\.|Corporation|Company|Co\.)\b',
        r'\b[A-Z][a-z]+ (?:LLC|Inc\.|Corp\.|Corporation|Company|Co\.)\b',
        r'\b(?:The )?[A-Z][a-z]+ (?:of [A-Z][a-z]+)?\b',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text)
        entities.extend(matches)
    
    return list(set(entities))  # Remove duplicates


def format_date(date_obj: datetime) -> str:
    """Format datetime object to readable string"""
    return date_obj.strftime("%Y-%m-%d %H:%M:%S")


def calculate_reading_time(word_count: int) -> int:
    """Calculate estimated reading time in minutes"""
    words_per_minute = 200  # Average reading speed
    return max(1, word_count // words_per_minute)


def validate_api_key(api_key: str, service: str) -> bool:
    """Validate API key format"""
    if not api_key or len(api_key.strip()) < 10:
        return False
    
    # Service-specific validation
    if service == 'google':
        return api_key.startswith('AIza') and len(api_key) == 39
    elif service == 'pinecone':
        return len(api_key) > 20
    
    return True


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate text to specified length"""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def extract_key_phrases(text: str, max_phrases: int = 10) -> List[str]:
    """Extract key phrases from text using simple heuristics"""
    # Simple implementation - in production, use NLP libraries
    sentences = text.split('.')
    phrases = []
    
    # Look for phrases with legal keywords
    legal_keywords = [
        'shall', 'must', 'agreement', 'contract', 'party', 'obligation',
        'liability', 'indemnification', 'termination', 'breach'
    ]
    
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) > 20 and len(sentence) < 200:
            for keyword in legal_keywords:
                if keyword.lower() in sentence.lower():
                    phrases.append(sentence)
                    break
    
    return phrases[:max_phrases]


def create_search_index(documents: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """Create a simple search index for documents"""
    index = {}
    
    for doc in documents:
        doc_id = doc.get('id', '')
        text = doc.get('content', '')
        
        # Simple word-based indexing
        words = re.findall(r'\b\w+\b', text.lower())
        
        for word in words:
            if len(word) > 3:  # Skip short words
                if word not in index:
                    index[word] = []
                if doc_id not in index[word]:
                    index[word].append(doc_id)
    
    return index


def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe storage"""
    # Remove or replace unsafe characters
    filename = re.sub(r'[^\w\-_\.]', '_', filename)
    # Limit length
    name, ext = os.path.splitext(filename)
    if len(name) > 100:
        name = name[:100]
    return name + ext


def parse_confidence_level(score: float) -> str:
    """Convert confidence score to human readable level"""
    if score >= 0.9:
        return "Very High"
    elif score >= 0.7:
        return "High"
    elif score >= 0.5:
        return "Medium"
    elif score >= 0.3:
        return "Low"
    else:
        return "Very Low"


def estimate_complexity(text: str) -> Dict[str, Any]:
    """Estimate document complexity based on various metrics"""
    words = text.split()
    sentences = text.split('.')
    
    # Calculate metrics
    avg_word_length = sum(len(word) for word in words) / max(len(words), 1)
    avg_sentence_length = len(words) / max(len(sentences), 1)
    
    # Count legal jargon
    legal_terms = [
        'whereas', 'heretofore', 'hereby', 'thereof', 'wherein',
        'indemnification', 'notwithstanding', 'pursuant'
    ]
    jargon_count = sum(1 for term in legal_terms if term in text.lower())
    
    # Calculate complexity score
    complexity_score = (
        (avg_word_length - 4) * 0.3 +
        (avg_sentence_length - 15) * 0.2 +
        (jargon_count / len(words) * 1000) * 0.5
    )
    
    complexity_level = "Low"
    if complexity_score > 5:
        complexity_level = "High"
    elif complexity_score > 2:
        complexity_level = "Medium"
    
    return {
        'score': complexity_score,
        'level': complexity_level,
        'avg_word_length': avg_word_length,
        'avg_sentence_length': avg_sentence_length,
        'jargon_density': jargon_count / len(words) * 100,
        'metrics': {
            'total_words': len(words),
            'total_sentences': len(sentences),
            'legal_terms_found': jargon_count
        }
    }