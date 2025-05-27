#!/usr/bin/env python3
"""
FIXED Streamlit Application for RAG-Based Semantic Quote Retrieval
Interactive web interface for the quote retrieval system - Windows compatible
"""

import streamlit as st
import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from typing import List, Dict, Any
import base64
from io import BytesIO

# Import our RAG pipeline
try:
    from rag_pipeline import QuoteRAGPipeline, RAGResponse, RetrievalResult
except ImportError:
    st.error("RAG pipeline not found. Make sure rag_pipeline.py is in the same directory.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Semantic Quote Retrieval System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1e3a8a;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .quote-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .quote-text {
        font-size: 1.2rem;
        font-style: italic;
        margin-bottom: 1rem;
        line-height: 1.6;
    }
    
    .quote-author {
        font-size: 1rem;
        font-weight: bold;
        text-align: right;
        margin-bottom: 0.5rem;
    }
    
    .quote-tags {
        font-size: 0.9rem;
        opacity: 0.8;
    }
    
    .similarity-score {
        background: rgba(255, 255, 255, 0.2);
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        display: inline-block;
        margin-top: 0.5rem;
    }
    
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1e3a8a;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_rag_pipeline():
    """Load and cache the RAG pipeline"""
    try:
        rag = QuoteRAGPipeline(use_openai=False)
        
        # Load data
        if not rag.load_quotes_data():
            st.error("Failed to load quotes data. Please ensure processed_quotes.jsonl exists.")
            return None
        
        # Load or create embeddings
        if not rag.create_embeddings():
            st.error("Failed to create embeddings.")
            return None
        
        # Load or build FAISS index
        if not rag.load_faiss_index():
            if not rag.build_faiss_index():
                st.error("Failed to build FAISS index.")
                return None
        
        return rag
    except Exception as e:
        st.error(f"Error loading RAG pipeline: {str(e)}")
        return None

def create_download_link(data: Dict, filename: str, text: str) -> str:
    """Create a download link for JSON data"""
    json_str = json.dumps(data, indent=2, ensure_ascii=False)
    b64 = base64.b64encode(json_str.encode()).decode()
    href = f'<a href="data:application/json;base64,{b64}" download="{filename}">{text}</a>'
    return href

def display_quote_card(result: RetrievalResult, index: int):
    """Display a single quote in a styled card"""
    st.markdown(f"""
    <div class="quote-card">
        <div class="quote-text">"{result.quote}"</div>
        <div class="quote-author">— {result.author}</div>
        <div class="quote-tags">Tags: {', '.join(result.tags) if result.tags else 'None'}</div>
        <div class="similarity-score">Similarity: {result.similarity_score:.4f}</div>
    </div>
    """, unsafe_allow_html=True)

def create_visualization(results: List[RetrievalResult], query: str):
    """Create visualizations for the results"""
    if not results:
        return
    
    # Similarity scores chart
    scores = [r.similarity_score for r in results]
    authors = [r.author for r in results]
    
    fig_scores = px.bar(
        x=authors,
        y=scores,
        title=f"Similarity Scores for Query: '{query}'",
        labels={'x': 'Authors', 'y': 'Similarity Score'},
        color=scores,
        color_continuous_scale='Blues'
    )
    fig_scores.update_layout(showlegend=False, xaxis_tickangle=-45)
    
    st.plotly_chart(fig_scores, use_container_width=True)
    
    # Tags distribution
    all_tags = []
    for result in results:
        if result.tags:
            all_tags.extend(result.tags)
    
    if all_tags:
        tag_counts = pd.Series(all_tags).value_counts().head(10)
        
        fig_tags = px.pie(
            values=tag_counts.values,
            names=tag_counts.index,
            title="Tag Distribution in Retrieved Quotes"
        )
        
        st.plotly_chart(fig_tags, use_container_width=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">📚 Semantic Quote Retrieval System</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Initialize session state for query management
    if 'search_query' not in st.session_state:
        st.session_state.search_query = ""
    
    # Initialize RAG pipeline
    if 'rag_pipeline' not in st.session_state:
        with st.spinner("Loading RAG pipeline... This may take a moment."):
            st.session_state.rag_pipeline = load_rag_pipeline()
    
    if st.session_state.rag_pipeline is None:
        st.error("Failed to load RAG pipeline. Please check your setup.")
        return
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Settings")
        
        # Number of results
        top_k = st.slider("Number of quotes to retrieve:", 1, 20, 5)
        
        # Dataset info
        st.header("📊 Dataset Info")
        
        if hasattr(st.session_state.rag_pipeline, 'quotes_data'):
            df = st.session_state.rag_pipeline.quotes_data
            st.metric("Total Quotes", len(df))
            st.metric("Unique Authors", df['author'].nunique())
            
            # Top authors
            top_authors = df['author'].value_counts().head(5)
            st.write("**Top Authors:**")
            for author, count in top_authors.items():
                st.write(f"• {author}: {count}")
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Search Quotes", "👤 Browse by Author", "🏷️ Browse by Tags", "📈 Analytics"])
    
    with tab1:
        st.header("Search for Quotes")
        
        # Search input - FIXED: Use form to avoid session state issues
        with st.form("search_form"):
            query = st.text_input(
                "Enter your query:",
                placeholder="e.g., 'quotes about hope by Oscar Wilde', 'motivational quotes', 'philosophical wisdom'",
                value=st.session_state.search_query
            )
            search_button = st.form_submit_button("🔍 Search", type="primary")
        
        # Example queries - FIXED: Use session state properly
        st.write("**Try these example queries:**")
        example_queries = [
            "quotes about hope and inspiration",
            "Oscar Wilde funny quotes", 
            "motivational quotes about success",
            "philosophical quotes about life",
            "quotes about love and relationships"
        ]
        
        example_cols = st.columns(len(example_queries))
        for i, example in enumerate(example_queries):
            if example_cols[i].button(f"'{example[:20]}...'", key=f"example_{i}"):
                st.session_state.search_query = example
                st.rerun()
        
        # Perform search
        if search_button and query:
            with st.spinner("Searching for relevant quotes..."):
                try:
                    st.session_state.search_query = query
                    response = st.session_state.rag_pipeline.process_query(query, top_k=top_k)
                    
                    # Display generated answer
                    st.subheader("📝 Generated Answer")
                    st.write(response.answer)
                    
                    # Display metadata
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Processing Time", f"{response.metadata['processing_time']:.2f}s")
                    with col2:
                        st.metric("Retrieved Quotes", response.metadata['num_retrieved'])
                    with col3:
                        st.metric("Method", response.metadata.get('search_method', 'Search'))
                    
                    # Display retrieved quotes
                    st.subheader("📚 Retrieved Quotes")
                    
                    for i, result in enumerate(response.retrieved_quotes):
                        display_quote_card(result, i)
                    
                    # Visualizations
                    st.subheader("📊 Analysis")
                    create_visualization(response.retrieved_quotes, query)
                    
                    # JSON download
                    st.subheader("💾 Export Results")
                    
                    export_data = {
                        'query': response.query,
                        'answer': response.answer,
                        'retrieved_quotes': [
                            {
                                'quote': r.quote,
                                'author': r.author,
                                'tags': r.tags,
                                'similarity_score': r.similarity_score,
                                'quote_id': r.quote_id
                            }
                            for r in response.retrieved_quotes
                        ],
                        'metadata': response.metadata
                    }
                    
                    filename = f"quote_search_{int(time.time())}.json"
                    download_link = create_download_link(export_data, filename, "📄 Download Results as JSON")
                    st.markdown(download_link, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"Error during search: {str(e)}")
    
    with tab2:
        st.header("Browse Quotes by Author")
        
        if hasattr(st.session_state.rag_pipeline, 'quotes_data'):
            df = st.session_state.rag_pipeline.quotes_data
            
            # Author selection
            authors = sorted(df['author'].unique())
            selected_author = st.selectbox("Select an author:", authors)
            
            max_quotes = st.slider("Maximum quotes to show:", 1, 50, 10)
            
            if st.button("📖 Get Author Quotes", type="primary"):
                try:
                    author_results = st.session_state.rag_pipeline.get_author_quotes(
                        selected_author, limit=max_quotes
                    )
                    
                    if author_results:
                        st.subheader(f"Quotes by {selected_author}")
                        
                        for i, result in enumerate(author_results):
                            display_quote_card(result, i)
                        
                        # Export option
                        export_data = {
                            'author': selected_author,
                            'quotes': [
                                {
                                    'quote': r.quote,
                                    'tags': r.tags,
                                    'quote_id': r.quote_id
                                }
                                for r in author_results
                            ]
                        }
                        
                        filename = f"{selected_author.replace(' ', '_')}_quotes.json"
                        download_link = create_download_link(export_data, filename, f"📄 Download {selected_author}'s Quotes")
                        st.markdown(download_link, unsafe_allow_html=True)
                    else:
                        st.info(f"No quotes found for {selected_author}")
                        
                except Exception as e:
                    st.error(f"Error retrieving author quotes: {str(e)}")
    
    with tab3:
        st.header("Browse Quotes by Tags")
        
        if hasattr(st.session_state.rag_pipeline, 'quotes_data'):
            df = st.session_state.rag_pipeline.quotes_data
            
            # Get all unique tags
            all_tags = []
            for tags in df['tags']:
                if isinstance(tags, list):
                    all_tags.extend(tags)
            
            unique_tags = sorted(list(set(all_tags)))
            
            # Tag selection
            selected_tag = st.selectbox("Select a tag:", unique_tags)
            
            max_quotes = st.slider("Maximum quotes to show:", 1, 50, 10, key="tag_max")
            
            if st.button("🏷️ Get Tagged Quotes", type="primary"):
                try:
                    tag_results = st.session_state.rag_pipeline.get_tag_quotes(
                        selected_tag, limit=max_quotes
                    )
                    
                    if tag_results:
                        st.subheader(f"Quotes tagged with '{selected_tag}'")
                        
                        for i, result in enumerate(tag_results):
                            display_quote_card(result, i)
                        
                        # Export option
                        export_data = {
                            'tag': selected_tag,
                            'quotes': [
                                {
                                    'quote': r.quote,
                                    'author': r.author,
                                    'tags': r.tags,
                                    'quote_id': r.quote_id
                                }
                                for r in tag_results
                            ]
                        }
                        
                        filename = f"{selected_tag}_quotes.json"
                        download_link = create_download_link(export_data, filename, f"📄 Download '{selected_tag}' Quotes")
                        st.markdown(download_link, unsafe_allow_html=True)
                    else:
                        st.info(f"No quotes found for tag '{selected_tag}'")
                        
                except Exception as e:
                    st.error(f"Error retrieving tagged quotes: {str(e)}")
    
    with tab4:
        st.header("Dataset Analytics")
        
        if hasattr(st.session_state.rag_pipeline, 'quotes_data'):
            df = st.session_state.rag_pipeline.quotes_data
            
            # Dataset overview
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Quotes", len(df))
            with col2:
                st.metric("Unique Authors", df['author'].nunique())
            with col3:
                avg_length = df['quote'].str.len().mean()
                st.metric("Avg Quote Length", f"{avg_length:.0f} chars")
            with col4:
                all_tags = []
                for tags in df['tags']:
                    if isinstance(tags, list):
                        all_tags.extend(tags)
                st.metric("Unique Tags", len(set(all_tags)))
            
            # Author distribution
            st.subheader("📊 Top Authors by Quote Count")
            top_authors = df['author'].value_counts().head(15)
            
            fig_authors = px.bar(
                x=top_authors.values,
                y=top_authors.index,
                orientation='h',
                title="Top 15 Authors by Number of Quotes",
                labels={'x': 'Number of Quotes', 'y': 'Authors'}
            )
            fig_authors.update_layout(height=600)
            st.plotly_chart(fig_authors, use_container_width=True)
            
            # Tag distribution
            st.subheader("🏷️ Most Common Tags")
            from collections import Counter
            tag_counts = Counter(all_tags).most_common(20)
            
            fig_tags = px.bar(
                x=[tag for tag, count in tag_counts],
                y=[count for tag, count in tag_counts],
                title="Top 20 Most Common Tags",
                labels={'x': 'Tags', 'y': 'Frequency'}
            )
            fig_tags.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_tags, use_container_width=True)
            
            # Quote length distribution - FIXED: Remove bins parameter
            st.subheader("📏 Quote Length Distribution")
            
            try:
                fig_length = px.histogram(
                    df,
                    x=df['quote'].str.len(),
                    title="Distribution of Quote Lengths",
                    labels={'x': 'Quote Length (characters)', 'count': 'Frequency'},
                    nbins=50  # FIXED: Use nbins instead of bins
                )
                st.plotly_chart(fig_length, use_container_width=True)
            except Exception as e:
                st.write("Quote length analysis temporarily unavailable")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p>📚 Semantic Quote Retrieval System powered by RAG (Retrieval Augmented Generation)</p>
        <p>Built with Streamlit, Sentence Transformers, and FAISS</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()