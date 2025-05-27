#!/usr/bin/env python3
"""
RAG Pipeline for Semantic Quote Retrieval
Core implementation that works with existing setup
"""

import json
import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import time
import warnings
warnings.filterwarnings('ignore')

# Try imports with fallbacks
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("⚠️ sentence-transformers not available")
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    print("⚠️ FAISS not available, using basic similarity")
    FAISS_AVAILABLE = False

try:
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    print("⚠️ scikit-learn not available")
    SKLEARN_AVAILABLE = False

@dataclass
class RetrievalResult:
    """Data class for retrieval results"""
    quote: str
    author: str
    tags: List[str]
    similarity_score: float
    quote_id: str

@dataclass
class RAGResponse:
    """Data class for RAG response"""
    query: str
    answer: str
    retrieved_quotes: List[RetrievalResult]
    metadata: Dict[str, Any]

class QuoteRAGPipeline:
    """Main RAG pipeline for quote retrieval"""
    
    def __init__(self, 
                 model_path: str = "all-MiniLM-L6-v2",
                 use_openai: bool = False):
        """
        Initialize the RAG pipeline
        
        Args:
            model_path: Path to sentence transformer model
            use_openai: Whether to use OpenAI for generation (optional)
        """
        self.model_path = model_path
        self.use_openai = use_openai
        self.embedding_model = None
        self.quotes_data = None
        self.embeddings = None
        self.faiss_index = None
        
        # Initialize model
        self._load_embedding_model()
    
    def _load_embedding_model(self):
        """Load the embedding model"""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            print("⚠️ Sentence transformers not available, using basic text matching")
            return
        
        try:
            print("🤖 Loading sentence transformer model...")
            self.embedding_model = SentenceTransformer(self.model_path)
            print("✅ Embedding model loaded successfully!")
        except Exception as e:
            print(f"⚠️ Error loading model: {e}")
            self.embedding_model = None
    
    def load_quotes_data(self, filename: str = 'processed_quotes.jsonl'):
        """Load the processed quotes data"""
        print(f"📂 Loading quotes data from {filename}...")
        
        if not Path(filename).exists():
            print(f"❌ File {filename} not found!")
            return False
        
        try:
            data = []
            with open(filename, 'r', encoding='utf-8') as f:
                for line in f:
                    data.append(json.loads(line.strip()))
            
            self.quotes_data = pd.DataFrame(data)
            print(f"✅ Loaded {len(self.quotes_data)} quotes successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def create_embeddings(self, force_recreate: bool = False):
        """Create embeddings for all quotes"""
        embeddings_file = 'quote_embeddings.pkl'
        
        # Try to load existing embeddings
        if not force_recreate and os.path.exists(embeddings_file):
            try:
                print("📥 Loading existing embeddings...")
                with open(embeddings_file, 'rb') as f:
                    self.embeddings = pickle.load(f)
                print(f"✅ Loaded embeddings for {len(self.embeddings)} quotes")
                return True
            except Exception as e:
                print(f"⚠️ Error loading embeddings: {e}")
        
        # Create new embeddings
        if not self.embedding_model:
            print("⚠️ No embedding model available, skipping embeddings")
            return False
        
        if self.quotes_data is None:
            print("❌ No quotes data loaded!")
            return False
        
        print("🔄 Creating embeddings...")
        texts = self.quotes_data['combined_text'].tolist()
        
        try:
            # Create embeddings in batches
            batch_size = 32
            embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                batch_embeddings = self.embedding_model.encode(batch_texts, show_progress_bar=False)
                embeddings.extend(batch_embeddings)
                
                if (i // batch_size + 1) % 10 == 0:
                    print(f"  Processed {i + len(batch_texts)}/{len(texts)} texts...")
            
            self.embeddings = np.array(embeddings)
            
            # Save embeddings
            with open(embeddings_file, 'wb') as f:
                pickle.dump(self.embeddings, f)
            
            print(f"✅ Created and saved embeddings: {self.embeddings.shape}")
            return True
            
        except Exception as e:
            print(f"❌ Error creating embeddings: {e}")
            return False
    
    def build_faiss_index(self):
        """Build FAISS index for efficient similarity search"""
        if not FAISS_AVAILABLE:
            print("⚠️ FAISS not available, using basic similarity search")
            return True  # Continue without FAISS
        
        if self.embeddings is None:
            print("❌ No embeddings available!")
            return False
        
        try:
            print("🔄 Building FAISS index...")
            
            # Create FAISS index
            dimension = self.embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatIP(dimension)
            
            # Normalize embeddings for cosine similarity
            normalized_embeddings = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)
            
            # Add to index
            self.faiss_index.add(normalized_embeddings.astype('float32'))
            
            print(f"✅ FAISS index built with {self.faiss_index.ntotal} vectors")
            
            # Save index
            faiss.write_index(self.faiss_index, 'quotes_faiss_index.bin')
            print("💾 FAISS index saved")
            
            return True
            
        except Exception as e:
            print(f"⚠️ FAISS index creation failed: {e}")
            return True  # Continue without FAISS
    
    def load_faiss_index(self):
        """Load existing FAISS index"""
        if not FAISS_AVAILABLE:
            return True  # Skip if FAISS not available
        
        try:
            self.faiss_index = faiss.read_index('quotes_faiss_index.bin')
            print(f"✅ Loaded FAISS index with {self.faiss_index.ntotal} vectors")
            return True
        except Exception as e:
            print(f"⚠️ Could not load FAISS index: {e}")
            return False
    
    def retrieve_quotes(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """
        Retrieve relevant quotes for a given query
        
        Args:
            query: Natural language query
            top_k: Number of quotes to retrieve
            
        Returns:
            List of RetrievalResult objects
        """
        if self.quotes_data is None:
            print("❌ No quotes data loaded!")
            return []
        
        # Try FAISS search first
        if self.faiss_index is not None and self.embedding_model is not None:
            return self._faiss_search(query, top_k)
        
        # Fall back to basic search
        return self._basic_search(query, top_k)
    
    def _faiss_search(self, query: str, top_k: int) -> List[RetrievalResult]:
        """FAISS-based semantic search"""
        try:
            # Encode query
            query_embedding = self.embedding_model.encode([query])
            query_embedding = query_embedding / np.linalg.norm(query_embedding)
            
            # Search in FAISS index
            similarities, indices = self.faiss_index.search(
                query_embedding.astype('float32'), top_k
            )
            
            # Create results
            results = []
            for similarity, idx in zip(similarities[0], indices[0]):
                if idx < len(self.quotes_data):
                    row = self.quotes_data.iloc[idx]
                    result = RetrievalResult(
                        quote=row['quote'],
                        author=row['author'],
                        tags=row['tags'] if isinstance(row['tags'], list) else [],
                        similarity_score=float(similarity),
                        quote_id=row['id']
                    )
                    results.append(result)
            
            return results
            
        except Exception as e:
            print(f"⚠️ FAISS search failed: {e}")
            return self._basic_search(query, top_k)
    
    def _basic_search(self, query: str, top_k: int) -> List[RetrievalResult]:
        """Basic keyword-based search fallback"""
        try:
            query_words = set(query.lower().split())
            matches = []
            
            for _, row in self.quotes_data.iterrows():
                # Check quote content
                quote_words = set(str(row['quote']).lower().split())
                quote_score = len(query_words.intersection(quote_words)) / max(len(query_words), 1)
                
                # Check tags
                tags = row['tags'] if isinstance(row['tags'], list) else []
                tag_words = set()
                for tag in tags:
                    tag_words.update(str(tag).lower().split('-'))
                tag_score = len(query_words.intersection(tag_words)) / max(len(query_words), 1)
                
                # Check author
                author_words = set(str(row['author']).lower().split())
                author_score = len(query_words.intersection(author_words)) / max(len(query_words), 1)
                
                # Combined score
                total_score = (quote_score * 0.6) + (tag_score * 0.3) + (author_score * 0.1)
                
                if total_score > 0:
                    matches.append({
                        'row': row,
                        'score': total_score
                    })
            
            # Sort by score and create results
            matches.sort(key=lambda x: x['score'], reverse=True)
            
            results = []
            for match in matches[:top_k]:
                row = match['row']
                result = RetrievalResult(
                    quote=row['quote'],
                    author=row['author'],
                    tags=row['tags'] if isinstance(row['tags'], list) else [],
                    similarity_score=match['score'],
                    quote_id=row['id']
                )
                results.append(result)
            
            return results
            
        except Exception as e:
            print(f"❌ Basic search failed: {e}")
            return []
    
    def generate_answer(self, query: str, retrieved_quotes: List[RetrievalResult]) -> str:
        """Generate answer based on retrieved quotes"""
        if not retrieved_quotes:
            return "I couldn't find any relevant quotes for your query. Try different keywords or browse by author/tags."
        
        # Simple template-based generation
        answer_parts = [f"Here are relevant quotes for '{query}':"]
        
        for i, result in enumerate(retrieved_quotes, 1):
            answer_parts.append(f"\n{i}. \"{result.quote}\" — {result.author}")
            if result.tags:
                answer_parts.append(f"   Tags: {', '.join(result.tags[:5])}...")  # Limit tags
        
        # Add summary
        authors = list(set([r.author for r in retrieved_quotes]))
        if len(authors) > 1:
            answer_parts.append(f"\nThese insights come from {len(authors)} different authors including {', '.join(authors[:3])}{'...' if len(authors) > 3 else ''}.")
        else:
            answer_parts.append(f"\nAll quotes are from {authors[0]}.")
        
        return "\n".join(answer_parts)
    
    def process_query(self, query: str, top_k: int = 5) -> RAGResponse:
        """
        Process a complete RAG query
        
        Args:
            query: User's natural language query
            top_k: Number of quotes to retrieve
            
        Returns:
            RAGResponse object with answer and metadata
        """
        start_time = time.time()
        
        # Retrieve relevant quotes
        retrieved_quotes = self.retrieve_quotes(query, top_k)
        
        # Generate answer
        answer = self.generate_answer(query, retrieved_quotes)
        
        end_time = time.time()
        
        # Create response
        response = RAGResponse(
            query=query,
            answer=answer,
            retrieved_quotes=retrieved_quotes,
            metadata={
                'processing_time': end_time - start_time,
                'num_retrieved': len(retrieved_quotes),
                'search_method': 'FAISS' if self.faiss_index else 'Basic',
                'model_used': self.model_path if self.embedding_model else 'Keyword matching'
            }
        )
        
        return response
    
    def get_author_quotes(self, author_name: str, limit: int = 10) -> List[RetrievalResult]:
        """Get quotes by a specific author"""
        if self.quotes_data is None:
            return []
        
        try:
            author_quotes = self.quotes_data[
                self.quotes_data['author'].str.contains(author_name, case=False, na=False)
            ]
            
            results = []
            for _, row in author_quotes.head(limit).iterrows():
                result = RetrievalResult(
                    quote=row['quote'],
                    author=row['author'],
                    tags=row['tags'] if isinstance(row['tags'], list) else [],
                    similarity_score=1.0,  # Perfect match for author
                    quote_id=row['id']
                )
                results.append(result)
            
            return results
            
        except Exception as e:
            print(f"❌ Error getting author quotes: {e}")
            return []
    
    def get_tag_quotes(self, tag: str, limit: int = 10) -> List[RetrievalResult]:
        """Get quotes with a specific tag"""
        if self.quotes_data is None:
            return []
        
        try:
            tag_quotes = self.quotes_data[
                self.quotes_data['tags'].apply(
                    lambda tags: tag.lower() in [str(t).lower() for t in tags] if isinstance(tags, list) else False
                )
            ]
            
            results = []
            for _, row in tag_quotes.head(limit).iterrows():
                result = RetrievalResult(
                    quote=row['quote'],
                    author=row['author'],
                    tags=row['tags'] if isinstance(row['tags'], list) else [],
                    similarity_score=1.0,  # Perfect match for tag
                    quote_id=row['id']
                )
                results.append(result)
            
            return results
            
        except Exception as e:
            print(f"❌ Error getting tag quotes: {e}")
            return []
    
    def get_random_quotes(self, limit: int = 5) -> List[RetrievalResult]:
        """Get random quotes"""
        if self.quotes_data is None:
            return []
        
        try:
            random_quotes = self.quotes_data.sample(n=min(limit, len(self.quotes_data)))
            
            results = []
            for _, row in random_quotes.iterrows():
                result = RetrievalResult(
                    quote=row['quote'],
                    author=row['author'],
                    tags=row['tags'] if isinstance(row['tags'], list) else [],
                    similarity_score=1.0,
                    quote_id=row['id']
                )
                results.append(result)
            
            return results
            
        except Exception as e:
            print(f"❌ Error getting random quotes: {e}")
            return []

def main():
    """Test the RAG pipeline"""
    print("🚀 Testing RAG Pipeline...")
    
    # Initialize pipeline
    rag = QuoteRAGPipeline()
    
    # Load data
    if not rag.load_quotes_data():
        print("❌ Failed to load data")
        return
    
    # Create embeddings and index
    rag.create_embeddings()
    rag.build_faiss_index()
    
    # Test queries
    test_queries = [
        "quotes about love and happiness",
        "wisdom from great thinkers",
        "funny quotes about life"
    ]
    
    print("\n🔍 Testing queries...")
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        response = rag.process_query(query, top_k=3)
        print(f"Found {len(response.retrieved_quotes)} quotes")
        for i, result in enumerate(response.retrieved_quotes, 1):
            print(f"  {i}. \"{result.quote[:60]}...\" - {result.author}")
    
    print("\n✅ RAG pipeline test completed!")

if __name__ == "__main__":
    main()