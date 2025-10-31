"""
Document preprocessing and text cleaning utilities.
"""

import re
import string
from typing import List, Dict, Any
import logging

from src.interfaces.base import BaseProcessor

logger = logging.getLogger(__name__)


class TextProcessor(BaseProcessor):
    """Text preprocessing and cleaning utilities."""
    
    def __init__(self, 
                 lowercase: bool = True,
                 remove_punctuation: bool = False,
                 remove_extra_whitespace: bool = True,
                 max_length: int = 512):
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_extra_whitespace = remove_extra_whitespace
        self.max_length = max_length
    
    def process_document(self, text: str) -> str:
        """Process and clean document text."""
        if not text:
            return ""
        
        # Basic cleaning
        processed = text.strip()
        
        # Remove extra whitespace
        if self.remove_extra_whitespace:
            processed = re.sub(r'\s+', ' ', processed)
        
        # Convert to lowercase
        if self.lowercase:
            processed = processed.lower()
        
        # Remove punctuation if specified
        if self.remove_punctuation:
            processed = processed.translate(str.maketrans('', '', string.punctuation))
        
        # Truncate if too long
        if len(processed) > self.max_length:
            processed = processed[:self.max_length].rsplit(' ', 1)[0]
        
        return processed
    
    def process_query(self, text: str) -> str:
        """Process and clean query text."""
        if not text:
            return ""
        
        # Basic cleaning
        processed = text.strip()
        
        # Remove extra whitespace
        if self.remove_extra_whitespace:
            processed = re.sub(r'\s+', ' ', processed)
        
        # Convert to lowercase for consistency
        if self.lowercase:
            processed = processed.lower()
        
        return processed
    
    def chunk_document(self, text: str, chunk_size: int = 256, overlap: int = 50) -> List[str]:
        """Split document into overlapping chunks."""
        if not text:
            return []
        
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            chunk_text = ' '.join(chunk_words)
            chunks.append(chunk_text)
            
            # Break if we've covered all words
            if i + chunk_size >= len(words):
                break
        
        return chunks
    
    def extract_keywords(self, text: str, top_k: int = 10) -> List[str]:
        """Extract top keywords from text (simple frequency-based)."""
        if not text:
            return []
        
        # Simple keyword extraction based on word frequency
        words = text.lower().split()
        
        # Remove common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'
        }
        
        # Count word frequencies
        word_freq = {}
        for word in words:
            word = word.strip(string.punctuation)
            if word and word not in stop_words and len(word) > 2:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Sort by frequency and return top-k
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:top_k]]


class DocumentChunker:
    """Utility for chunking long documents."""
    
    def __init__(self, chunk_size: int = 256, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_by_sentences(self, text: str) -> List[str]:
        """Chunk document by sentences while respecting size limits."""
        if not text:
            return []
        
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence.split())
            
            if current_length + sentence_length > self.chunk_size and current_chunk:
                # Save current chunk
                chunks.append(' '.join(current_chunk))
                
                # Start new chunk with overlap
                overlap_sentences = current_chunk[-self.overlap//10:] if self.overlap > 0 else []
                current_chunk = overlap_sentences + [sentence]
                current_length = sum(len(s.split()) for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        # Add final chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def chunk_by_paragraphs(self, text: str) -> List[str]:
        """Chunk document by paragraphs."""
        if not text:
            return []
        
        paragraphs = text.split('\n\n')
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for paragraph in paragraphs:
            para_length = len(paragraph.split())
            
            if current_length + para_length > self.chunk_size and current_chunk:
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = [paragraph]
                current_length = para_length
            else:
                current_chunk.append(paragraph)
                current_length += para_length
        
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
        
        return chunks