#!/usr/bin/env python3
"""
RAG Pipeline for Semantic Quote Retrieval
Implements retrieval-augmented generation using fine-tuned embeddings and LLM
"""

import json
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import openai
from typing import List, Dict, Tuple, Optional, Any
import os
import pickle
from dataclasses import dataclass
import re
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# For open-source LLM integration
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
except ImportError:
    print("Transformers not available for open-source models")

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
    def __init__(self, 
                 model_path: str = "./fine_tuned_quote_model",
                 openai_api_key: Optional[str] = None,
                 use_openai: bool = True):
        """
        Initialize the RAG pipeline
        
        Args:
            model_path: Path to fine-tuned sentence transformer
            openai_api_key: OpenAI API key (if using OpenAI models)
            use_openai: Whether to use OpenAI or open-source LLM
        """
        self.model_path = model_path
        self.use_openai = use_openai
        self.embedding_model = None
        self.quotes_data = None
        self.embeddings = None
        self.faiss_index = None
        self.llm_pipeline = None
        
        # Initialize OpenAI if needed
        if use_openai and openai_api_key:
            openai.api_key = openai_api_key
        
        # Load components
        self._load_embedding_model()
        self._initialize_llm()
    
    def _load_embedding_model(self):
        """Load the fine-tuned embedding model"""
        print("Loading fine-tuned embedding model...")
        try:
            self.embedding_model = SentenceTransformer(self.model_path)
            print("Embedding model loaded successfully!")
        except Exception as e:
            print(f"Error loading embedding model: {e}")
            print("Falling back to base model...")
            self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    def _initialize_llm(self):
        """Initialize the language model for generation"""
        print("Initializing language model...")
        
        if self.use_openai:
            print("Using OpenAI GPT for generation")
            # OpenAI client is initialized with API key
        else:
            print("Initializing open-source LLM...")
            try:
                # Use a lightweight model for generation
                model_name = "microsoft/DialoGPT-medium"
                self.llm_pipeline = pipeline(
                    "text-generation",
                    model=model_name,
                    tokenizer=model_name,
                    max_length=512,
                    device=-1  # CPU
                )
                print("Open-source LLM loaded successfully!")
            except Exception as e:
                print(f"Error loading open-source LLM: {e}")
                self.llm_pipeline = None
    
    def load_quotes_data(self, filename: str = 'processed_quotes.jsonl'):
        """Load the processed quotes data"""
        print(f"Loading quotes data from {filename}...")
        
        try:
            data = []
            with open(filename, 'r', encoding='utf-8') as f:
                for line in f:
                    data.append(json.loads(line.strip()))
            
            self.quotes_data = pd.DataFrame(data)
            print(f"Loaded {len(self.quotes_data)} quotes successfully!")
            return True
            
        except FileNotFoundError:
            print(f"File {filename} not found. Please run data preprocessing first.")
            return False
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def create_embeddings(self, force_recreate: bool = False):
        """Create embeddings for all quotes"""
        embeddings_file = 'quote_embeddings.pkl'
        
        if not force_recreate and os.path.exists(embeddings_file):
            print("Loading existing embeddings...")
            try:
                with open(embeddings_file, 'rb') as f:
                    self.embeddings = pickle.load(f)
                print(f"Loaded embeddings for {len(self.embeddings)} quotes")
                return True
            except Exception as e:
                print(f"Error loading embeddings: {e}")
        
        print("Creating embeddings for all quotes...")
        
        if self.quotes_data is None:
            print("No quotes data loaded!")
            return False
        
        # Get texts for embedding
        texts = self.quotes_data['combined_text'].tolist()
        
        # Create embeddings in batches
        batch_size = 32
        embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size)):
            batch_texts = texts[i:i+batch_size]
            batch_embeddings = self.embedding_model.encode(batch_texts)
            embeddings.extend(batch_embeddings)
        
        self.embeddings = np.array(embeddings)
        
        # Save embeddings
        with open(embeddings_file, 'wb') as f:
            pickle.dump(self.embeddings, f)
        
        print(f"Created and saved embeddings for {len(self.embeddings)} quotes")
        return True
    
    def build_faiss_index(self):
        """Build FAISS index for efficient similarity search"""
        print("Building FAISS index...")
        
        if self.embeddings is None:
            print("No embeddings available! Please create embeddings first.")
            return False
        
        # Create FAISS index
        dimension = self.embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        normalized_embeddings = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        
        # Add to index
        self.faiss_index.add(normalized_embeddings.astype('float32'))
        
        print(f"FAISS index built with {self.faiss_index.ntotal} vectors")
        
        # Save index
        faiss.write_index(self.faiss_index, 'quotes_faiss_index.bin')
        print("FAISS index saved to quotes_faiss_index.bin")
        
        return True
    
    def load_faiss_index(self):
        """Load existing FAISS index"""
        try:
            self.faiss_index = faiss.read_index('quotes_faiss_index.bin')
            print(f"Loaded FAISS index with {self.faiss_index.ntotal} vectors")
            return True
        except Exception as e:
            print(f"Error loading FAISS index: {e}")
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
        if self.faiss_index is None:
            print("FAISS index not loaded!")
            return []
        
        # Encode query
        query_embedding = self.embedding_model.encode([query])
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        # Search in FAISS index
        similarities, indices = self.faiss_index.search(
            query_embedding.astype('float32'), top_k
        )
        
        # Create results
        results = []
        for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
            if idx < len(self.quotes_data):
                row = self.quotes_data.iloc[idx]
                result = RetrievalResult(
                    quote=row['quote'],
                    author=row['author'],
                    tags=row['tags'],
                    similarity_score=float(similarity),
                    quote_id=row['id']
                )
                results.append(result)
        
        return results
    
    def generate_response_openai(self, query: str, retrieved_quotes: List[RetrievalResult]) -> str:
        """Generate response using OpenAI GPT"""
        # Prepare context
        context_quotes = []
        for result in retrieved_quotes:
            context_quotes.append(f'"{result.quote}" - {result.author} (Tags: {", ".join(result.tags)})')
        
        context = "\n".join(context_quotes)
        
        # Create prompt
        prompt = f"""Based on the following quotes, provide a comprehensive answer to the user's query.

Query: {query}

Relevant Quotes:
{context}

Instructions:
1. Directly answer the user's query using the provided quotes
2. Reference specific quotes and authors when relevant
3. Provide insights that connect the quotes to the query
4. Format your response in a clear, engaging manner
5. If the query asks for specific quotes, list them clearly

Response:"""
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a knowledgeable assistant that helps users find and understand relevant quotes. Provide clear, insightful responses based on the provided quote context."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Error with OpenAI API: {e}")
            return self._generate_fallback_response(query, retrieved_quotes)
    
    def generate_response_opensource(self, query: str, retrieved_quotes: List[RetrievalResult]) -> str:
        """Generate response using open-source LLM"""
        if self.llm_pipeline is None:
            return self._generate_fallback_response(query, retrieved_quotes)
        
        # Prepare context (shorter for open-source models)
        context_quotes = []
        for result in retrieved_quotes[:3]:  # Limit context for smaller models
            context_quotes.append(f'"{result.quote}" - {result.author}')
        
        context = "\n".join(context_quotes)
        
        # Create prompt
        prompt = f"Query: {query}\n\nRelevant quotes:\n{context}\n\nAnswer:"
        
        try:
            response = self.llm_pipeline(
                prompt,
                max_length=len(prompt.split()) + 150,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True
            )
            
            generated_text = response[0]['generated_text']
            answer = generated_text[len(prompt):].strip()
            
            return answer if answer else self._generate_fallback_response(query, retrieved_quotes)
            
        except Exception as e:
            print(f"Error with open-source LLM: {e}")
            return self._generate_fallback_response(query, retrieved_quotes)
    
    def _generate_fallback_response(self, query: str, retrieved_quotes: List[RetrievalResult]) -> str:
        """Generate a simple fallback response"""
        if not retrieved_quotes:
            return "I couldn't find any relevant quotes for your query."
        
        response_parts = [f"Here are relevant quotes for your query about '{query}':"]
        
        for i, result in enumerate(retrieved_quotes, 1):
            response_parts.append(f"\n{i}. \"{result.quote}\" - {result.author}")
            if result.tags:
                response_parts.append(f"   Tags: {', '.join(result.tags)}")
        
        return "\n".join(response_parts)
    
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
        
        # Generate response
        if self.use_openai:
            answer = self.generate_response_openai(query, retrieved_quotes)
        else:
            answer = self.generate_response_opensource(query, retrieved_quotes)
        
        end_time = time.time()
        
        # Create response object
        response = RAGResponse(
            query=query,
            answer=answer,
            retrieved_quotes=retrieved_quotes,
            metadata={
                'processing_time': end_time - start_time,
                'num_retrieved': len(retrieved_quotes),
                'retrieval_method': 'FAISS + Fine-tuned embeddings',
                'generation_method': 'OpenAI GPT' if self.use_openai else 'Open-source LLM'
            }
        )
        
        return response
    
    def get_query_suggestions(self, partial_query: str, limit: int = 5) -> List[str]:
        """Get query suggestions based on available data"""
        # Simple implementation - could be enhanced
        suggestions = [
            f"quotes about {partial_query} by famous authors",
            f"inspirational quotes on {partial_query}",
            f"philosophical quotes about {partial_query}",
            f"motivational {partial_query} quotes",
            f"famous {partial_query} sayings"
        ]
        
        return suggestions[:limit]
    
    def get_author_quotes(self, author_name: str, limit: int = 10) -> List[RetrievalResult]:
        """Get quotes by a specific author"""
        author_quotes = self.quotes_data[
            self.quotes_data['author'].str.contains(author_name, case=False, na=False)
        ]
        
        results = []
        for _, row in author_quotes.head(limit).iterrows():
            result = RetrievalResult(
                quote=row['quote'],
                author=row['author'],
                tags=row['tags'],
                similarity_score=1.0,  # Perfect match for author
                quote_id=row['id']
            )
            results.append(result)
        
        return results
    
    def get_tag_quotes(self, tag: str, limit: int = 10) -> List[RetrievalResult]:
        """Get quotes with a specific tag"""
        tag_quotes = self.quotes_data[
            self.quotes_data['tags'].apply(
                lambda tags: tag.lower() in [t.lower() for t in tags] if tags else False
            )
        ]
        
        results = []
        for _, row in tag_quotes.head(limit).iterrows():
            result = RetrievalResult(
                quote=row['quote'],
                author=row['author'],
                tags=row['tags'],
                similarity_score=1.0,  # Perfect match for tag
                quote_id=row['id']
            )
            results.append(result)
        
        return results
    
    def export_results_json(self, response: RAGResponse, filename: str = None) -> str:
        """Export RAG response to JSON format"""
        if filename is None:
            timestamp = int(time.time())
            filename = f"rag_response_{timestamp}.json"
        
        # Convert to JSON-serializable format
        export_data = {
            'query': response.query,
            'answer': response.answer,
            'retrieved_quotes': [
                {
                    'quote': result.quote,
                    'author': result.author,
                    'tags': result.tags,
                    'similarity_score': result.similarity_score,
                    'quote_id': result.quote_id
                }
                for result in response.retrieved_quotes
            ],
            'metadata': response.metadata
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f"Results exported to {filename}")
        return filename

def main():
    """Main function to demonstrate RAG pipeline"""
    print("Initializing Quote RAG Pipeline...")
    
    # Initialize pipeline (using open-source for demo)
    rag = QuoteRAGPipeline(use_openai=False)  # Set to True if you have OpenAI API key
    
    # Load data
    if not rag.load_quotes_data():
        return
    
    # Create embeddings and index
    if not rag.create_embeddings():
        return
    
    if not rag.build_faiss_index():
        return
    
    # Test queries
    test_queries = [
        "quotes about hope and inspiration",
        "Oscar Wilde funny quotes",
        "motivational quotes about success",
        "quotes about love and relationships",
        "philosophical quotes about life and death"
    ]
    
    print("\n" + "="*60)
    print("TESTING RAG PIPELINE")
    print("="*60)
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        print("-" * 50)
        
        response = rag.process_query(query, top_k=3)
        
        print(f"\nAnswer:\n{response.answer}")
        
        print(f"\nRetrieved Quotes:")
        for i, result in enumerate(response.retrieved_quotes, 1):
            print(f"{i}. \"{result.quote}\" - {result.author}")
            print(f"   Similarity: {result.similarity_score:.4f}")
            print(f"   Tags: {', '.join(result.tags)}")
        
        print(f"\nProcessing time: {response.metadata['processing_time']:.2f}s")
        
        # Export results
        filename = rag.export_results_json(response)
        print(f"Results saved to: {filename}")
    
    print("\n" + "="*60)
    print("RAG PIPELINE TESTING COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()