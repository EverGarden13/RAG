"""
Dataset loading utilities for HQ-small dataset from HuggingFace.
"""

import json
import os
from typing import List, Dict, Any, Tuple
from datasets import load_dataset
import logging

from src.models.data_models import Query, Document
from src.models.config import DataConfig

logger = logging.getLogger(__name__)


class HQSmallLoader:
    """Loader for HotpotQA small dataset (HQ-small)."""
    
    def __init__(self, cache_dir: str = "./data/cache"):
        """Initialize HQ-small loader.
        
        Args:
            cache_dir: Directory to cache downloaded dataset
        """
        self.cache_dir = cache_dir
        self.dataset = None
        self.documents = {}
        self.collection = None
        
    def load_dataset(self) -> Dict[str, Any]:
        """Load HQ-small train, validation, and test splits from HuggingFace.
        
        Dataset: izhx/COMP5423-25Fall-HQ-small
        Splits: train, validation, test, collection
        Note: Collection has different schema, loaded separately
        """
        try:
            logger.info("Loading HQ-small dataset from HuggingFace...")
            
            # Load query splits (train, validation, test) - they have same schema
            self.dataset = {}
            
            # Load train split
            logger.info("Loading train split...")
            train_dataset = load_dataset(
                "json",
                data_files={"train": "hf://datasets/izhx/COMP5423-25Fall-HQ-small/train.jsonl"},
                split="train",
                cache_dir=self.cache_dir
            )
            self.dataset['train'] = train_dataset
            
            # Load validation split
            logger.info("Loading validation split...")
            val_dataset = load_dataset(
                "json",
                data_files={"validation": "hf://datasets/izhx/COMP5423-25Fall-HQ-small/validation.jsonl"},
                split="validation",
                cache_dir=self.cache_dir
            )
            self.dataset['validation'] = val_dataset
            
            # Load test split (may not be available yet - released 15 days before deadline)
            logger.info("Loading test split...")
            try:
                test_dataset = load_dataset(
                    "json",
                    data_files={"test": "hf://datasets/izhx/COMP5423-25Fall-HQ-small/test.jsonl"},
                    split="test",
                    cache_dir=self.cache_dir
                )
                self.dataset['test'] = test_dataset
            except FileNotFoundError:
                logger.warning("Test split not available yet (will be released 15 days before deadline)")
                self.dataset['test'] = None
            
            # Load collection split (different schema)
            logger.info("Loading collection split...")
            collection_dataset = load_dataset(
                "json",
                data_files={"collection": "hf://datasets/izhx/COMP5423-25Fall-HQ-small/collection.jsonl"},
                split="collection",
                cache_dir=self.cache_dir
            )
            self.dataset['collection'] = collection_dataset
            
            logger.info(f"Dataset loaded successfully:")
            logger.info(f"  Train: {len(self.dataset['train'])} samples")
            logger.info(f"  Validation: {len(self.dataset['validation'])} samples")
            if self.dataset['test'] is not None:
                logger.info(f"  Test: {len(self.dataset['test'])} samples")
            else:
                logger.info(f"  Test: Not available yet")
            logger.info(f"  Collection: {len(self.dataset['collection'])} documents")
            
            return self.dataset
            
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise
    
    def load_collection(self) -> List[Document]:
        """Load document collection from HQ-small dataset.
        
        Returns:
            List of Document objects from the collection split
        """
        if self.dataset is None:
            self.load_dataset()
        
        logger.info("Loading document collection...")
        documents = []
        
        # Load from collection split
        collection_data = self.dataset['collection']
        
        for item in collection_data:
            doc = Document(
                id=item['id'],
                text=item['text']
            )
            documents.append(doc)
            self.documents[doc.id] = doc
        
        logger.info(f"Loaded {len(documents)} documents from collection")
        return documents
    
    def load_train(self) -> List[Dict[str, Any]]:
        """Load training split with queries, answers, and supporting_ids."""
        if self.dataset is None:
            self.load_dataset()
        
        train_data = []
        for item in self.dataset['train']:
            train_data.append({
                'id': item['id'],
                'text': item['text'],
                'answer': item['answer'],
                'supporting_ids': item['supporting_ids']
            })
        
        logger.info(f"Loaded {len(train_data)} training samples")
        return train_data
    
    def load_validation(self) -> List[Dict[str, Any]]:
        """Load validation split with queries, answers, and supporting_ids."""
        if self.dataset is None:
            self.load_dataset()
        
        val_data = []
        for item in self.dataset['validation']:
            val_data.append({
                'id': item['id'],
                'text': item['text'],
                'answer': item['answer'],
                'supporting_ids': item['supporting_ids']
            })
        
        logger.info(f"Loaded {len(val_data)} validation samples")
        return val_data
    
    def load_test(self) -> List[Dict[str, Any]]:
        """Load test split with queries only (no answers).
        
        Note: Test split will be released 15 days before submission deadline.
        """
        if self.dataset is None:
            self.load_dataset()
        
        if self.dataset['test'] is None:
            logger.warning("Test split not available yet")
            return []
        
        test_data = []
        for item in self.dataset['test']:
            test_data.append({
                'id': item['id'],
                'text': item['text']
            })
        
        logger.info(f"Loaded {len(test_data)} test samples")
        return test_data
    
    def get_statistics(self) -> Dict[str, int]:
        """Get dataset statistics."""
        if self.dataset is None:
            self.load_dataset()
        
        stats = {
            'train': len(self.dataset['train']),
            'validation': len(self.dataset['validation']),
            'test': len(self.dataset['test']) if self.dataset['test'] is not None else 0,
            'collection': len(self.dataset['collection'])
        }
        
        return stats
    
    def validate_dataset_structure(self) -> bool:
        """Validate that the loaded dataset has the expected structure.
        
        Returns:
            True if valid, raises exception otherwise
        """
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
        
        required_splits = ['train', 'validation', 'test', 'collection']
        for split in required_splits:
            if split not in self.dataset:
                raise ValueError(f"Missing required split: {split}")
        
        # Validate train/validation structure
        for split in ['train', 'validation']:
            sample = self.dataset[split][0]
            required_fields = ['id', 'text', 'answer', 'supporting_ids']
            for field in required_fields:
                if field not in sample:
                    raise ValueError(f"Missing field '{field}' in {split} split")
        
        # Validate test structure (if available)
        if self.dataset['test'] is not None:
            test_sample = self.dataset['test'][0]
            required_test_fields = ['id', 'text']
            for field in required_test_fields:
                if field not in test_sample:
                    raise ValueError(f"Missing field '{field}' in test split")
        
        # Validate collection structure
        collection_sample = self.dataset['collection'][0]
        required_collection_fields = ['id', 'text']
        for field in required_collection_fields:
            if field not in collection_sample:
                raise ValueError(f"Missing field '{field}' in collection split")
        
        logger.info("Dataset structure validation passed")
        return True