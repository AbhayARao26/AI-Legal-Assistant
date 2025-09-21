"""
Streamlit Frontend for Legal AI Document Analysis
"""
import streamlit as st
import requests
import json
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Configuration
API_BASE_URL = "http://localhost:8000"

# Page configuration
st.set_page_config(
    page_title="Legal AI Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .section-header {
        font-size: 1.5rem;
        color: #2c5aa0;
        border-bottom: 2px solid #e1e5e9;
        padding-bottom: 0.5rem;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    
    .highlight-box {
        background-color: #f8f9fa;
        border-left: 4px solid #007bff;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.25rem;
    }
    
    .legal-term {
        background-color: #fff3cd;
        padding: 0.2rem 0.4rem;
        border-radius: 0.25rem;
        border: 1px solid #ffeaa7;
        cursor: help;
    }
    
    .citation {
        background-color: #e7f3ff;
        padding: 0.5rem;
        margin: 0.5rem 0;
        border-radius: 0.25rem;
        font-size: 0.9rem;
        border-left: 3px solid #007bff;
    }
    
    .confidence-high { color: #28a745; font-weight: bold; }
    .confidence-medium { color: #ffc107; font-weight: bold; }
    .confidence-low { color: #dc3545; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# Session state initialization
if 'session_id' not in st.session_state:
    st.session_state.session_id = None
if 'uploaded_documents' not in st.session_state:
    st.session_state.uploaded_documents = []
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'current_document' not in st.session_state:
    st.session_state.current_document = None


def create_session():
    """Create a new session"""
    try:
        response = requests.post(f"{API_BASE_URL}/session")
        if response.status_code == 200:
            session_data = response.json()
            st.session_state.session_id = session_data['session_id']
            return True
    except Exception as e:
        st.error(f"Failed to create session: {str(e)}")
    return False


def upload_document(file):
    """Upload a document to the API"""
    try:
        files = {"file": (file.name, file.getvalue(), file.type)}
        data = {"user_id": st.session_state.session_id}
        
        response = requests.post(f"{API_BASE_URL}/upload_document", files=files, data=data)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Upload failed: {response.text}")
            return None
    except Exception as e:
        st.error(f"Upload error: {str(e)}")
        return None


def get_document_info(document_id: str):
    """Get document information"""
    try:
        response = requests.get(f"{API_BASE_URL}/document/{document_id}")
        if response.status_code == 200:
            return response.json()
        return None
    except Exception:
        return None


def summarize_document(document_id: str, summary_type: str = "overall", section_name: str = None):
    """Get document summary"""
    try:
        data = {
            "document_id": document_id,
            "summary_type": summary_type,
            "section_name": section_name
        }
        
        response = requests.post(f"{API_BASE_URL}/summarize", json=data)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Summarization failed: {response.text}")
            return None
    except Exception as e:
        st.error(f"Summarization error: {str(e)}")
        return None


def extract_clauses(document_id: str, clause_types: List[str] = None):
    """Extract clauses from document"""
    try:
        data = {
            "document_id": document_id,
            "clause_types": clause_types
        }
        
        response = requests.post(f"{API_BASE_URL}/extract_clauses", json=data)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Clause extraction failed: {response.text}")
            return None
    except Exception as e:
        st.error(f"Clause extraction error: {str(e)}")
        return None


def define_terms(terms: List[str], document_id: str = None, context: str = None):
    """Define legal terms"""
    try:
        data = {
            "terms": terms,
            "document_id": document_id,
            "context": context
        }
        
        response = requests.post(f"{API_BASE_URL}/define_terms", json=data)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Term definition failed: {response.text}")
            return None
    except Exception as e:
        st.error(f"Term definition error: {str(e)}")
        return None


def ask_question(question: str, document_id: str):
    """Ask a question about the document"""
    try:
        data = {
            "question": question,
            "document_id": document_id,
            "conversation_history": st.session_state.conversation_history
        }
        
        response = requests.post(f"{API_BASE_URL}/ask", json=data)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Question answering failed: {response.text}")
            return None
    except Exception as e:
        st.error(f"Question answering error: {str(e)}")
        return None


def format_confidence_score(score: float) -> str:
    """Format confidence score with color coding"""
    if score >= 0.8:
        return f'<span class="confidence-high">{score:.1%}</span>'
    elif score >= 0.6:
        return f'<span class="confidence-medium">{score:.1%}</span>'
    else:
        return f'<span class="confidence-low">{score:.1%}</span>'


def render_citations(citations: List[Dict]):
    """Render citations in a formatted way"""
    if not citations:
        return
    
    st.markdown("#### 📚 Sources:")
    for i, citation in enumerate(citations):
        with st.expander(f"Source {i+1} (Relevance: {citation.get('relevance_score', 0):.1%})"):
            st.markdown(f'<div class="citation">{citation.get("text", "")}</div>', unsafe_allow_html=True)


def render_legal_term_tooltip(term: str, definition: str):
    """Render a legal term with tooltip"""
    return f'<span class="legal-term" title="{definition}">{term}</span>'


def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">⚖️ Legal AI Assistant</h1>', unsafe_allow_html=True)
    st.markdown("### Transform complex legal documents into clear, accessible insights")
    
    # Create session if not exists
    if not st.session_state.session_id:
        if st.button("Start New Session", type="primary"):
            if create_session():
                st.success("Session created successfully!")
                st.rerun()
    
    if not st.session_state.session_id:
        st.warning("Please start a new session to continue.")
        return
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📁 Document Management")
        
        # Upload section
        uploaded_file = st.file_uploader(
            "Upload Legal Document",
            type=['pdf', 'docx', 'txt'],
            help="Upload a legal document for analysis"
        )
        
        if uploaded_file and st.button("Process Document"):
            with st.spinner("Uploading and processing document..."):
                result = upload_document(uploaded_file)
                if result:
                    st.success("Document uploaded successfully!")
                    st.session_state.uploaded_documents.append(result)
                    st.session_state.current_document = result['id']
                    st.rerun()
        
        # Document selector
        if st.session_state.uploaded_documents:
            st.markdown("### 📄 Select Document")
            doc_options = {doc['filename']: doc['id'] for doc in st.session_state.uploaded_documents}
            selected_doc_name = st.selectbox("Choose document:", list(doc_options.keys()))
            
            if selected_doc_name:
                st.session_state.current_document = doc_options[selected_doc_name]
                
                # Show document status
                doc_info = get_document_info(st.session_state.current_document)
                if doc_info:
                    st.info(f"Status: {doc_info['processing_status'].title()}")
                    if doc_info['processing_status'] == 'completed':
                        st.success("Ready for analysis!")
    
    # Main content area
    if not st.session_state.current_document:
        st.info("👆 Please upload a document to get started")
        return
    
    # Check document status
    doc_info = get_document_info(st.session_state.current_document)
    if not doc_info or doc_info['processing_status'] != 'completed':
        st.warning("Document is still being processed. Please wait...")
        return
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📄 Document Summary", 
        "📜 Clauses & Terms", 
        "📖 Legal Definitions", 
        "❓ Ask Questions",
        "📊 Document Analysis"
    ])
    
    # Document Summary Tab
    with tab1:
        st.markdown('<div class="section-header">Document Summary</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        
        with col2:
            summary_type = st.selectbox(
                "Summary Type:",
                ["overall", "section", "clause"],
                help="Choose the level of detail for the summary"
            )
            
            section_name = None
            if summary_type == "section":
                section_name = st.text_input("Section Name (optional):")
        
        if st.button("Generate Summary", type="primary"):
            with st.spinner("Generating summary..."):
                summary_result = summarize_document(
                    st.session_state.current_document, 
                    summary_type, 
                    section_name
                )
                
                if summary_result:
                    st.markdown("### Summary")
                    st.markdown(summary_result['summary_text'])
                    
                    if summary_result.get('key_points'):
                        st.markdown("### Key Points")
                        for point in summary_result['key_points']:
                            st.markdown(f"• {point}")
    
    # Clauses & Terms Tab
    with tab2:
        st.markdown('<div class="section-header">Clauses & Terms Extraction</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        
        with col2:
            clause_types = st.multiselect(
                "Select Clause Types:",
                ["obligation", "right", "penalty", "timeline", "termination", "renewal", 
                 "payment", "indemnification", "liability", "confidentiality"],
                help="Leave empty to extract all types"
            )
        
        if st.button("Extract Clauses", type="primary"):
            with st.spinner("Extracting clauses..."):
                clauses_result = extract_clauses(
                    st.session_state.current_document,
                    clause_types if clause_types else None
                )
                
                if clauses_result:
                    clauses = clauses_result['clauses']
                    
                    if clauses:
                        # Group clauses by type
                        clause_groups = {}
                        for clause in clauses:
                            clause_type = clause['clause_type']
                            if clause_type not in clause_groups:
                                clause_groups[clause_type] = []
                            clause_groups[clause_type].append(clause)
                        
                        for clause_type, clause_list in clause_groups.items():
                            with st.expander(f"{clause_type.title()} Clauses ({len(clause_list)})"):
                                for clause in clause_list:
                                    st.markdown(f"**Importance:** {clause['importance_score']}/10")
                                    st.markdown(f"**Text:** {clause['clause_text']}")
                                    if clause.get('simplified_explanation'):
                                        st.markdown(f"**Explanation:** {clause['simplified_explanation']}")
                                    st.markdown("---")
                    else:
                        st.info("No clauses found for the selected types.")
    
    # Legal Definitions Tab
    with tab3:
        st.markdown('<div class="section-header">Legal Definitions</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            terms_input = st.text_area(
                "Enter legal terms to define (one per line):",
                placeholder="indemnification\nforce majeure\narbitration"
            )
        
        with col2:
            context_input = st.text_area(
                "Additional Context (optional):",
                placeholder="Provide context for better definitions"
            )
        
        if st.button("Define Terms", type="primary") and terms_input:
            terms = [term.strip() for term in terms_input.split('\n') if term.strip()]
            
            with st.spinner("Defining terms..."):
                definitions_result = define_terms(
                    terms,
                    st.session_state.current_document,
                    context_input
                )
                
                if definitions_result:
                    for definition in definitions_result['definitions']:
                        with st.expander(f"📖 {definition['term']}"):
                            st.markdown(f"**Legal Definition:** {definition['legal_definition']}")
                            st.markdown(f"**Simple Explanation:** {definition['simple_definition']}")
                            
                            if definition.get('examples'):
                                st.markdown("**Examples:**")
                                for example in definition['examples']:
                                    st.markdown(f"• {example}")
                            
                            if definition.get('related_terms'):
                                st.markdown(f"**Related Terms:** {', '.join(definition['related_terms'])}")
    
    # Ask Questions Tab
    with tab4:
        st.markdown('<div class="section-header">Ask Questions</div>', unsafe_allow_html=True)
        
        # Question input
        question = st.text_input(
            "Ask a question about the document:",
            placeholder="What are my obligations under this contract?"
        )
        
        if st.button("Ask Question", type="primary") and question:
            with st.spinner("Processing question..."):
                qa_result = ask_question(question, st.session_state.current_document)
                
                if qa_result:
                    # Add to conversation history
                    st.session_state.conversation_history.append({
                        'question': question,
                        'answer': qa_result['answer'],
                        'timestamp': datetime.now().isoformat()
                    })
                    
                    # Display answer
                    st.markdown("### Answer")
                    st.markdown(qa_result['answer'])
                    
                    # Display confidence
                    confidence_html = format_confidence_score(qa_result['confidence_score'])
                    st.markdown(f"**Confidence:** {confidence_html}", unsafe_allow_html=True)
                    
                    # Display citations
                    if qa_result.get('citations'):
                        render_citations(qa_result['citations'])
                    
                    # Display related questions
                    if qa_result.get('related_questions'):
                        st.markdown("### Related Questions")
                        for related_q in qa_result['related_questions']:
                            if st.button(f"💡 {related_q}", key=f"related_{hash(related_q)}"):
                                st.session_state.question_to_ask = related_q
                                st.rerun()
        
        # Conversation history
        if st.session_state.conversation_history:
            st.markdown("### Conversation History")
            for i, conv in enumerate(reversed(st.session_state.conversation_history[-5:])):
                with st.expander(f"Q: {conv['question'][:50]}..."):
                    st.markdown(f"**Question:** {conv['question']}")
                    st.markdown(f"**Answer:** {conv['answer']}")
                    st.markdown(f"**Time:** {conv['timestamp']}")
    
    # Document Analysis Tab
    with tab5:
        st.markdown('<div class="section-header">Document Analysis Dashboard</div>', unsafe_allow_html=True)
        
        if doc_info:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("File Size", f"{doc_info['file_size'] / 1024:.1f} KB")
            
            with col2:
                st.metric("File Type", doc_info['file_type'].upper())
            
            with col3:
                if 'word_count' in doc_info.get('metadata', {}):
                    st.metric("Word Count", doc_info['metadata']['word_count'])
            
            with col4:
                if 'estimated_pages' in doc_info.get('metadata', {}):
                    st.metric("Est. Pages", doc_info['metadata']['estimated_pages'])
            
            # Document preview
            if doc_info.get('content_preview'):
                st.markdown("### Document Preview")
                with st.expander("Show document preview"):
                    st.text(doc_info['content_preview'])
        
        # Analytics placeholder
        st.markdown("### Document Complexity Analysis")
        st.info("Advanced analytics features coming soon!")


if __name__ == "__main__":
    main()