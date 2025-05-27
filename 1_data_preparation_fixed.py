#!/usr/bin/env python3
"""
FIXED Data Preparation for RAG-Based Semantic Quote Retrieval System
Loads, explores, and preprocesses the Abirate/english_quotes dataset
Fixed the tags processing error for Windows
"""

import pandas as pd
import numpy as np
import json
import re
from datasets import load_dataset
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# Try to import textstat, use fallback if not available
try:
    from textstat import flesch_reading_ease
    USE_TEXTSTAT = True
except ImportError:
    print("⚠️ textstat not available, using basic readability estimation")
    USE_TEXTSTAT = False
    
    def flesch_reading_ease(text):
        """Simple readability fallback"""
        if not text:
            return 0
        words = len(text.split())
        sentences = len([s for s in text.split('.') if s.strip()])
        if sentences == 0:
            return 50  # neutral score
        avg_sentence_length = words / sentences
        return max(0, min(100, 120 - avg_sentence_length))

class QuoteDataProcessor:
    def __init__(self):
        self.dataset = None
        self.df = None
        self.processed_df = None
        
    def load_dataset(self):
        """Load the Abirate/english_quotes dataset from HuggingFace"""
        print("Loading Abirate/english_quotes dataset...")
        try:
            self.dataset = load_dataset("Abirate/english_quotes")
            self.df = pd.DataFrame(self.dataset['train'])
            print(f"Dataset loaded successfully!")
            print(f"Shape: {self.df.shape}")
            return True
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return False
    
    def explore_dataset(self):
        """Perform comprehensive exploratory data analysis"""
        print("\n" + "="*50)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*50)
        
        # Basic info
        print(f"\nDataset Shape: {self.df.shape}")
        print(f"Columns: {list(self.df.columns)}")
        
        # Sample data
        print(f"\nSample Data:")
        for i in range(3):
            sample = self.df.iloc[i]
            print(f"\nSample {i+1}:")
            print(f"  Author: {sample['author']}")
            print(f"  Quote: {sample['quote'][:100]}...")
            print(f"  Tags: {sample['tags']}")
        
        # Missing values
        print(f"\nMissing Values:")
        print(self.df.isnull().sum())
        
        # Unique authors
        unique_authors = self.df['author'].nunique()
        print(f"\nUnique Authors: {unique_authors}")
        
        # Author distribution (top 10)
        print(f"\nTop 10 Authors by Quote Count:")
        top_authors = self.df['author'].value_counts().head(10)
        print(top_authors)
        
        # Tag analysis
        all_tags = []
        for tags in self.df['tags'].dropna():
            if isinstance(tags, list):
                all_tags.extend(tags)
        
        if all_tags:
            print(f"\nTotal Unique Tags: {len(set(all_tags))}")
            print(f"Most Common Tags:")
            tag_counts = Counter(all_tags)
            for tag, count in tag_counts.most_common(15):
                print(f"  {tag}: {count}")
        
        # Quote length analysis
        quote_lengths = self.df['quote'].str.len()
        print(f"\nQuote Length Statistics:")
        print(f"  Mean: {quote_lengths.mean():.2f}")
        print(f"  Median: {quote_lengths.median():.2f}")
        print(f"  Min: {quote_lengths.min()}")
        print(f"  Max: {quote_lengths.max()}")
        
        return {
            'total_quotes': len(self.df),
            'unique_authors': unique_authors,
            'total_tags': len(set(all_tags)) if all_tags else 0,
            'avg_quote_length': quote_lengths.mean()
        }
    
    def clean_quote_text(self, text: str) -> str:
        """Clean and normalize quote text"""
        if pd.isna(text) or not text:
            return ""
        
        # Convert to string if not already
        text = str(text)
        
        # Remove extra quotes if present
        text = text.strip()
        if text.startswith('"') and text.endswith('"'):
            text = text[1:-1]
        
        # Clean whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove any remaining problematic characters (but keep basic punctuation)
        text = re.sub(r'[^\w\s\.,!?;:\'-]', '', text)
        
        return text
    
    def clean_author_name(self, name: str) -> str:
        """Clean and normalize author names"""
        if pd.isna(name) or not name:
            return "Unknown"
        
        # Convert to string and clean
        name = str(name)
        name = re.sub(r'\s+', ' ', name).strip()
        
        # Handle common author name patterns
        name = re.sub(r'\s*,\s*', ', ', name)
        
        return name
    
    def process_tags(self, tags: Any) -> List[str]:
        """FIXED: Process and clean tags - handles all tag formats"""
        # Handle None/NaN values
        if tags is None or pd.isna(tags):
            return []
        
        # If it's already a list, process it
        if isinstance(tags, list):
            cleaned_tags = []
            for tag in tags:
                if tag and isinstance(tag, str):
                    # Clean and normalize tag
                    clean_tag = re.sub(r'[^\w\-]', '', str(tag).lower().strip())
                    if clean_tag and len(clean_tag) > 1:
                        cleaned_tags.append(clean_tag)
            return list(set(cleaned_tags))  # Remove duplicates
        
        # If it's a string representation of a list
        if isinstance(tags, str):
            try:
                # Try to evaluate string representation of list
                evaluated_tags = eval(tags)
                if isinstance(evaluated_tags, list):
                    return self.process_tags(evaluated_tags)  # Recursive call with list
            except:
                # If evaluation fails, treat as single tag
                clean_tag = re.sub(r'[^\w\-]', '', tags.lower().strip())
                if clean_tag and len(clean_tag) > 1:
                    return [clean_tag]
        
        # Default: return empty list
        return []
    
    def preprocess_data(self):
        """Preprocess and clean the entire dataset"""
        print("\n" + "="*50)
        print("DATA PREPROCESSING")
        print("="*50)
        
        self.processed_df = self.df.copy()
        
        # Clean quotes
        print("Cleaning quote texts...")
        self.processed_df['quote_clean'] = self.processed_df['quote'].apply(self.clean_quote_text)
        
        # Clean authors
        print("Cleaning author names...")
        self.processed_df['author_clean'] = self.processed_df['author'].apply(self.clean_author_name)
        
        # Process tags - FIXED VERSION
        print("Processing tags...")
        self.processed_df['tags_clean'] = self.processed_df['tags'].apply(self.process_tags)
        
        # Remove empty quotes
        before_len = len(self.processed_df)
        self.processed_df = self.processed_df[self.processed_df['quote_clean'].str.len() > 10]
        after_len = len(self.processed_df)
        print(f"Removed {before_len - after_len} quotes with less than 10 characters")
        
        # Create combined text for embedding
        print("Creating combined text for embeddings...")
        self.processed_df['combined_text'] = (
            self.processed_df['quote_clean'] + " [AUTHOR] " + 
            self.processed_df['author_clean'] + " [TAGS] " + 
            self.processed_df['tags_clean'].apply(lambda x: ' '.join(x) if x else '')
        )
        
        # Add reading ease score
        print("Calculating readability scores...")
        self.processed_df['readability'] = self.processed_df['quote_clean'].apply(
            lambda x: flesch_reading_ease(x) if len(x) > 0 else 0
        )
        
        # Add quote length and word count
        self.processed_df['quote_length'] = self.processed_df['quote_clean'].str.len()
        self.processed_df['word_count'] = self.processed_df['quote_clean'].str.split().str.len()
        
        print(f"\nProcessed Dataset Shape: {self.processed_df.shape}")
        print(f"Sample processed data:")
        print(self.processed_df[['quote_clean', 'author_clean', 'tags_clean']].head(2))
        
        return self.processed_df
    
    def create_visualization(self):
        """Create visualizations for the dataset"""
        print("\nCreating visualizations...")
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Quote length distribution
            axes[0, 0].hist(self.processed_df['quote_length'], bins=50, alpha=0.7, color='skyblue')
            axes[0, 0].set_title('Distribution of Quote Lengths')
            axes[0, 0].set_xlabel('Quote Length (characters)')
            axes[0, 0].set_ylabel('Frequency')
            
            # Top authors
            top_authors = self.processed_df['author_clean'].value_counts().head(10)
            axes[0, 1].barh(range(len(top_authors)), top_authors.values, color='lightcoral')
            axes[0, 1].set_yticks(range(len(top_authors)))
            axes[0, 1].set_yticklabels(top_authors.index)
            axes[0, 1].set_title('Top 10 Authors by Quote Count')
            axes[0, 1].set_xlabel('Number of Quotes')
            
            # Readability distribution
            axes[1, 0].hist(self.processed_df['readability'], bins=30, alpha=0.7, color='lightgreen')
            axes[1, 0].set_title('Readability Score Distribution')
            axes[1, 0].set_xlabel('Flesch Reading Ease Score')
            axes[1, 0].set_ylabel('Frequency')
            
            # Tags per quote
            tags_per_quote = self.processed_df['tags_clean'].apply(len)
            axes[1, 1].hist(tags_per_quote, bins=range(0, max(tags_per_quote)+2), alpha=0.7, color='gold')
            axes[1, 1].set_title('Distribution of Tags per Quote')
            axes[1, 1].set_xlabel('Number of Tags')
            axes[1, 1].set_ylabel('Frequency')
            
            plt.tight_layout()
            plt.savefig('dataset_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()
            print("✅ Visualizations saved to dataset_analysis.png")
            
        except Exception as e:
            print(f"⚠️ Visualization creation failed: {e}")
            print("Continuing without visualizations...")
    
    def save_processed_data(self, filename='processed_quotes.jsonl'):
        """Save processed data to JSONL format"""
        print(f"\nSaving processed data to {filename}...")
        
        # Convert to records format
        records = []
        for idx, row in self.processed_df.iterrows():
            record = {
                'id': f"quote_{idx}",
                'quote': row['quote_clean'],
                'author': row['author_clean'],
                'tags': row['tags_clean'],
                'combined_text': row['combined_text'],
                'readability': row['readability'],
                'quote_length': row['quote_length'],
                'word_count': row['word_count']
            }
            records.append(record)
        
        # Save as JSONL
        with open(filename, 'w', encoding='utf-8') as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        
        print(f"✅ Saved {len(records)} processed quotes to {filename}")
        return filename
    
    def get_dataset_stats(self):
        """Get comprehensive dataset statistics"""
        if self.processed_df is None:
            print("Please preprocess data first!")
            return None
        
        # Tag statistics
        all_tags = []
        for tags in self.processed_df['tags_clean']:
            all_tags.extend(tags)
        
        stats = {
            'total_quotes': len(self.processed_df),
            'unique_authors': self.processed_df['author_clean'].nunique(),
            'total_unique_tags': len(set(all_tags)),
            'avg_quote_length': self.processed_df['quote_length'].mean(),
            'avg_word_count': self.processed_df['word_count'].mean(),
            'avg_readability': self.processed_df['readability'].mean(),
            'most_common_tags': Counter(all_tags).most_common(10),
            'top_authors': self.processed_df['author_clean'].value_counts().head(5).to_dict()
        }
        
        return stats

def main():
    """Main execution function"""
    print("🚀 Starting Quote Dataset Processing...")
    
    # Initialize processor
    processor = QuoteDataProcessor()
    
    # Load dataset
    if not processor.load_dataset():
        print("❌ Failed to load dataset. Exiting.")
        return
    
    # Explore dataset
    exploration_stats = processor.explore_dataset()
    
    # Preprocess data
    processed_df = processor.preprocess_data()
    
    # Create visualizations
    try:
        processor.create_visualization()
    except Exception as e:
        print(f"⚠️ Visualization creation failed: {e}")
    
    # Save processed data
    filename = processor.save_processed_data()
    
    # Get final stats
    final_stats = processor.get_dataset_stats()
    
    print(f"\n" + "="*50)
    print("✅ PROCESSING COMPLETE")
    print("="*50)
    print(f"Final Dataset Statistics:")
    for key, value in final_stats.items():
        if key not in ['most_common_tags', 'top_authors']:
            print(f"  {key}: {value}")
    
    print(f"\n📄 Processed data saved to: {filename}")
    print("🎯 Ready for model training and RAG pipeline development!")
    
    return True

if __name__ == "__main__":
    main()