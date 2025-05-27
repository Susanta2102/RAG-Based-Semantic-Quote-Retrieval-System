#!/usr/bin/env python3
"""
Model Fine-Tuning for Semantic Quote Retrieval
Fine-tunes a sentence transformer model on the quotes dataset
"""

import json
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader
import torch
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class QuoteEmbeddingTrainer:
    def __init__(self, base_model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the trainer with a base sentence transformer model
        
        Args:
            base_model_name: Name of the base model to fine-tune
        """
        self.base_model_name = base_model_name
        self.model = None
        self.data = None
        self.train_examples = []
        self.eval_examples = []
        
    def load_processed_data(self, filename: str = 'processed_quotes.jsonl'):
        """Load the processed quotes data"""
        print(f"Loading processed data from {filename}...")
        
        data = []
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                for line in f:
                    data.append(json.loads(line.strip()))
            
            self.data = pd.DataFrame(data)
            print(f"Loaded {len(self.data)} quotes successfully!")
            return True
            
        except FileNotFoundError:
            print(f"File {filename} not found. Please run data preprocessing first.")
            return False
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def create_training_pairs(self, num_positive_pairs: int = 5000, num_negative_pairs: int = 5000):
        """
        Create positive and negative pairs for training
        
        Positive pairs: quotes from same author or with overlapping tags
        Negative pairs: random quotes with different authors and no overlapping tags
        """
        print("Creating training pairs...")
        
        positive_pairs = []
        negative_pairs = []
        
        # Create positive pairs
        print("Creating positive pairs...")
        
        # Same author pairs
        author_groups = self.data.groupby('author')
        for author, group in tqdm(author_groups):
            if len(group) >= 2:
                quotes = group['combined_text'].tolist()
                for i in range(min(10, len(quotes))):  # Limit pairs per author
                    for j in range(i+1, min(i+6, len(quotes))):  # Max 5 pairs per quote
                        if len(positive_pairs) < num_positive_pairs // 2:
                            positive_pairs.append((quotes[i], quotes[j], 1.0))
        
        # Similar tags pairs
        print("Creating tag-based positive pairs...")
        for idx in tqdm(range(min(1000, len(self.data)))):  # Limit for efficiency
            row1 = self.data.iloc[idx]
            tags1 = set(row1['tags'])
            
            if len(tags1) > 0:
                # Find quotes with overlapping tags
                candidates = []
                for idx2 in range(idx+1, min(idx+100, len(self.data))):
                    row2 = self.data.iloc[idx2]
                    tags2 = set(row2['tags'])
                    
                    overlap = len(tags1.intersection(tags2))
                    if overlap >= 2:  # At least 2 common tags
                        similarity_score = min(1.0, overlap / len(tags1.union(tags2)))
                        candidates.append((row1['combined_text'], row2['combined_text'], similarity_score))
                
                # Add best candidates
                candidates.sort(key=lambda x: x[2], reverse=True)
                for candidate in candidates[:3]:  # Top 3 similar
                    if len(positive_pairs) < num_positive_pairs:
                        positive_pairs.append(candidate)
        
        # Create negative pairs
        print("Creating negative pairs...")
        for _ in tqdm(range(num_negative_pairs)):
            idx1, idx2 = random.sample(range(len(self.data)), 2)
            row1, row2 = self.data.iloc[idx1], self.data.iloc[idx2]
            
            # Ensure different authors and no common tags
            if (row1['author'] != row2['author'] and 
                len(set(row1['tags']).intersection(set(row2['tags']))) == 0):
                negative_pairs.append((row1['combined_text'], row2['combined_text'], 0.0))
        
        print(f"Created {len(positive_pairs)} positive pairs and {len(negative_pairs)} negative pairs")
        
        # Combine and shuffle
        all_pairs = positive_pairs + negative_pairs
        random.shuffle(all_pairs)
        
        return all_pairs
    
    def prepare_training_data(self, pairs: List[Tuple], test_size: float = 0.2):
        """Convert pairs to SentenceTransformer InputExamples"""
        print("Preparing training data...")
        
        # Convert to InputExamples
        examples = []
        for text1, text2, score in pairs:
            examples.append(InputExample(texts=[text1, text2], label=float(score)))
        
        # Split train/validation
        train_examples, eval_examples = train_test_split(
            examples, test_size=test_size, random_state=42
        )
        
        self.train_examples = train_examples
        self.eval_examples = eval_examples
        
        print(f"Training examples: {len(train_examples)}")
        print(f"Evaluation examples: {len(eval_examples)}")
        
        return train_examples, eval_examples
    
    def initialize_model(self):
        """Initialize the sentence transformer model"""
        print(f"Initializing model: {self.base_model_name}")
        
        try:
            self.model = SentenceTransformer(self.base_model_name)
            print("Model initialized successfully!")
            
            # Print model info
            print(f"Model max sequence length: {self.model.max_seq_length}")
            print(f"Model dimension: {self.model.get_sentence_embedding_dimension()}")
            
        except Exception as e:
            print(f"Error initializing model: {e}")
            return False
        
        return True
    
    def fine_tune_model(self, epochs: int = 3, batch_size: int = 16, warmup_steps: int = 100):
        """Fine-tune the model on the quote data"""
        print("Starting model fine-tuning...")
        
        if not self.model or not self.train_examples:
            print("Model or training data not prepared!")
            return False
        
        # Create data loader
        train_dataloader = DataLoader(self.train_examples, shuffle=True, batch_size=batch_size)
        
        # Define loss function
        train_loss = losses.CosineSimilarityLoss(self.model)
        
        # Create evaluator
        evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
            self.eval_examples, name='quotes-eval'
        )
        
        # Training
        print(f"Training for {epochs} epochs with batch size {batch_size}")
        
        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            evaluator=evaluator,
            epochs=epochs,
            warmup_steps=warmup_steps,
            output_path='./fine_tuned_quote_model',
            evaluation_steps=500,
            save_best_model=True,
            show_progress_bar=True
        )
        
        print("Fine-tuning completed!")
        return True
    
    def evaluate_model(self, sample_size: int = 100):
        """Evaluate the fine-tuned model"""
        print("Evaluating model performance...")
        
        if not self.model:
            print("Model not loaded!")
            return None
        
        # Sample evaluation data
        eval_sample = random.sample(self.eval_examples, min(sample_size, len(self.eval_examples)))
        
        # Get embeddings
        texts1 = [ex.texts[0] for ex in eval_sample]
        texts2 = [ex.texts[1] for ex in eval_sample]
        labels = [ex.label for ex in eval_sample]
        
        embeddings1 = self.model.encode(texts1)
        embeddings2 = self.model.encode(texts2)
        
        # Calculate similarities
        similarities = []
        for emb1, emb2 in zip(embeddings1, embeddings2):
            sim = cosine_similarity([emb1], [emb2])[0][0]
            similarities.append(sim)
        
        # Correlation with labels
        correlation = np.corrcoef(similarities, labels)[0, 1]
        
        print(f"Evaluation Results:")
        print(f"  Correlation with labels: {correlation:.4f}")
        print(f"  Mean similarity: {np.mean(similarities):.4f}")
        print(f"  Std similarity: {np.std(similarities):.4f}")
        
        # Plot results
        plt.figure(figsize=(10, 6))
        
        plt.subplot(1, 2, 1)
        plt.scatter(labels, similarities, alpha=0.6)
        plt.xlabel('True Labels')
        plt.ylabel('Predicted Similarities')
        plt.title(f'Model Predictions vs True Labels\nCorrelation: {correlation:.4f}')
        plt.plot([0, 1], [0, 1], 'r--', alpha=0.8)
        
        plt.subplot(1, 2, 2)
        plt.hist(similarities, bins=20, alpha=0.7, color='skyblue')
        plt.xlabel('Predicted Similarities')
        plt.ylabel('Frequency')
        plt.title('Distribution of Predicted Similarities')
        
        plt.tight_layout()
        plt.savefig('model_evaluation.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return {
            'correlation': correlation,
            'mean_similarity': np.mean(similarities),
            'std_similarity': np.std(similarities)
        }
    
    def test_model_inference(self):
        """Test the model with some example queries"""
        print("\nTesting model inference...")
        
        # Test queries
        test_queries = [
            "quotes about hope and inspiration",
            "Oscar Wilde funny quotes",
            "motivational quotes about success",
            "philosophical quotes about life",
            "quotes about love and relationships"
        ]
        
        # Sample quotes for comparison
        sample_quotes = self.data.sample(20)['combined_text'].tolist()
        
        print("\nRunning inference tests:")
        for query in test_queries:
            print(f"\nQuery: '{query}'")
            
            # Get query embedding
            query_embedding = self.model.encode([query])
            
            # Get quote embeddings
            quote_embeddings = self.model.encode(sample_quotes)
            
            # Calculate similarities
            similarities = cosine_similarity(query_embedding, quote_embeddings)[0]
            
            # Get top 3 matches
            top_indices = np.argsort(similarities)[-3:][::-1]
            
            print("Top 3 matches:")
            for i, idx in enumerate(top_indices):
                score = similarities[idx]
                quote_text = sample_quotes[idx].split('[AUTHOR]')[0].strip()
                author = sample_quotes[idx].split('[AUTHOR]')[1].split('[TAGS]')[0].strip()
                print(f"  {i+1}. Score: {score:.4f}")
                print(f"     Quote: {quote_text[:100]}...")
                print(f"     Author: {author}")
    
    def save_model(self, path: str = './fine_tuned_quote_model'):
        """Save the fine-tuned model"""
        if self.model:
            self.model.save(path)
            print(f"Model saved to {path}")
            
            # Save training metadata
            metadata = {
                'base_model': self.base_model_name,
                'training_examples': len(self.train_examples),
                'eval_examples': len(self.eval_examples),
                'model_dimension': self.model.get_sentence_embedding_dimension()
            }
            
            with open(f"{path}/training_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            return True
        return False
    
    def load_model(self, path: str = './fine_tuned_quote_model'):
        """Load a pre-trained model"""
        try:
            self.model = SentenceTransformer(path)
            print(f"Model loaded from {path}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

def main():
    """Main training pipeline"""
    print("Starting Quote Embedding Model Training...")
    
    # Initialize trainer
    trainer = QuoteEmbeddingTrainer(base_model_name="all-MiniLM-L6-v2")
    
    # Load processed data
    if not trainer.load_processed_data():
        print("Failed to load processed data. Please run data preprocessing first.")
        return
    
    # Initialize model
    if not trainer.initialize_model():
        print("Failed to initialize model.")
        return
    
    # Create training pairs
    pairs = trainer.create_training_pairs(num_positive_pairs=3000, num_negative_pairs=3000)
    
    # Prepare training data
    train_examples, eval_examples = trainer.prepare_training_data(pairs)
    
    # Fine-tune model
    success = trainer.fine_tune_model(epochs=2, batch_size=16, warmup_steps=100)
    
    if success:
        # Evaluate model
        eval_results = trainer.evaluate_model()
        
        # Test inference
        trainer.test_model_inference()
        
        # Save model
        trainer.save_model()
        
        print("\n" + "="*50)
        print("TRAINING COMPLETE")
        print("="*50)
        print("Model fine-tuning completed successfully!")
        print("Model saved to ./fine_tuned_quote_model")
        print("Ready for RAG pipeline integration!")
        
    else:
        print("Training failed!")

if __name__ == "__main__":
    main()