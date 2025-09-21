"""
Document Retrieval & Analysis Agent
Handles document ingestion, chunking, embedding, and vector storage
"""
import os
import uuid
from typing import List, Dict, Any, Optional
import asyncio
from datetime import datetime

# Document processing imports
import PyPDF2
from docx import Document as DocxDocument
import io
from sentence_transformers import SentenceTransformer
import pinecone
from pinecone import Pinecone, ServerlessSpec
import google.generativeai as genai

from .base_agent import BaseAgent, AgentInput, AgentOutput, AgentState
from models.schemas import DocumentType, ProcessingStatus
from config import settings


class DocumentRetrievalAgent(BaseAgent):
    """Agent responsible for document processing and retrieval"""
    
    def __init__(self):
        super().__init__("DocumentRetrievalAgent")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize Pinecone with new API
        self.pc = Pinecone(api_key=settings.pinecone_api_key)
        
        # Create or connect to index
        index_names = [index.name for index in self.pc.list_indexes()]
        if settings.pinecone_index_name not in index_names:
            self.pc.create_index(
                name=settings.pinecone_index_name,
                dimension=384,  # all-MiniLM-L6-v2 dimension
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )
        
        self.index = self.pc.Index(settings.pinecone_index_name)
        
        # Initialize Gemini
        genai.configure(api_key=settings.google_api_key)
        self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
    
    async def run(self, input_data: AgentInput, state: AgentState) -> AgentOutput:
        """Main method to process document operations"""
        try:
            action = input_data.parameters.get('action', 'process')
            
            if action == 'process':
                return await self._process_document(input_data, state)
            elif action == 'search':
                return await self._search_documents(input_data, state)
            elif action == 'get_chunks':
                return await self._get_document_chunks(input_data, state)
            else:
                return self.create_error_output(f"Unknown action: {action}")
                
        except Exception as e:
            return self.create_error_output(f"Error in DocumentRetrievalAgent: {str(e)}")
    
    async def _process_document(self, input_data: AgentInput, state: AgentState) -> AgentOutput:
        """Process and store a document"""
        try:
            file_path = input_data.parameters.get('file_path')
            file_type = input_data.parameters.get('file_type')
            document_id = input_data.parameters.get('document_id')
            
            if not all([file_path, file_type, document_id]):
                return self.create_error_output("Missing required parameters: file_path, file_type, document_id")
            
            # Extract text from document
            text_content = await self._extract_text(file_path, file_type)
            
            # Split into chunks
            chunks = self._split_text(text_content)
            
            # Generate embeddings and store in Pinecone
            await self._store_embeddings(document_id, chunks)
            
            # Extract metadata
            metadata = await self._extract_metadata(text_content, file_type)
            
            result = {
                'document_id': document_id,
                'processing_status': ProcessingStatus.COMPLETED,
                'chunks_count': len(chunks),
                'metadata': metadata,
                'content_preview': text_content[:500] + "..." if len(text_content) > 500 else text_content
            }
            
            return self.create_success_output(result)
            
        except Exception as e:
            return self.create_error_output(f"Error processing document: {str(e)}")
    
    async def _extract_text(self, file_path: str, file_type: str) -> str:
        """Extract text from different document types"""
        try:
            if file_type.lower() == 'pdf':
                return await self._extract_pdf_text(file_path)
            elif file_type.lower() == 'docx':
                return await self._extract_docx_text(file_path)
            elif file_type.lower() == 'txt':
                return await self._extract_txt_text(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
        except Exception as e:
            raise Exception(f"Text extraction failed: {str(e)}")
    
    async def _extract_pdf_text(self, file_path: str) -> str:
        """Extract text from PDF"""
        text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text
    
    async def _extract_docx_text(self, file_path: str) -> str:
        """Extract text from DOCX"""
        doc = DocxDocument(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    
    async def _extract_txt_text(self, file_path: str) -> str:
        """Extract text from TXT"""
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    
    def _split_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[Dict[str, Any]]:
        """Split text into overlapping chunks"""
        chunks = []
        words = text.split()
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            chunk_text = ' '.join(chunk_words)
            
            chunks.append({
                'text': chunk_text,
                'chunk_id': str(uuid.uuid4()),
                'chunk_index': len(chunks),
                'word_count': len(chunk_words)
            })
        
        return chunks
    
    async def _store_embeddings(self, document_id: str, chunks: List[Dict[str, Any]]):
        """Generate embeddings and store in Pinecone"""
        vectors = []
        
        for chunk in chunks:
            embedding = self.embedding_model.encode(chunk['text']).tolist()
            
            vectors.append({
                'id': f"{document_id}_{chunk['chunk_id']}",
                'values': embedding,
                'metadata': {
                    'document_id': document_id,
                    'chunk_id': chunk['chunk_id'],
                    'chunk_index': chunk['chunk_index'],
                    'text': chunk['text'],
                    'word_count': chunk['word_count'],
                    'timestamp': datetime.now().isoformat()
                }
            })
        
        # Upsert in batches
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            self.index.upsert(vectors=batch)
    
    async def _extract_metadata(self, text: str, file_type: str) -> Dict[str, Any]:
        """Extract metadata from document using Gemini"""
        try:
            prompt = f"""
            Analyze this legal document and extract the following metadata:
            1. Document type (contract, agreement, policy, etc.)
            2. Parties involved
            3. Key dates (effective date, expiration, etc.)
            4. Main subject/purpose
            5. Jurisdiction
            6. Document structure (number of sections, pages estimate)
            
            Document text (first 2000 characters):
            {text[:2000]}
            
            Return the metadata as a JSON object.
            """
            
            response = await self.gemini_model.generate_content_async(prompt)
            
            # Parse the response and extract JSON
            metadata_text = response.text
            
            # Basic metadata extraction
            word_count = len(text.split())
            char_count = len(text)
            
            return {
                'file_type': file_type,
                'word_count': word_count,
                'character_count': char_count,
                'estimated_pages': max(1, word_count // 250),  # Rough estimate
                'analysis': metadata_text,
                'processed_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'file_type': file_type,
                'word_count': len(text.split()),
                'character_count': len(text),
                'error': f"Metadata extraction failed: {str(e)}",
                'processed_at': datetime.now().isoformat()
            }
    
    async def _search_documents(self, input_data: AgentInput, state: AgentState) -> AgentOutput:
        """Search for relevant document chunks"""
        try:
            query = input_data.query
            document_id = input_data.document_id
            top_k = input_data.parameters.get('top_k', 5)
            
            # Generate query embedding
            query_embedding = self.embedding_model.encode(query).tolist()
            
            # Search in Pinecone
            filter_dict = {}
            if document_id:
                filter_dict['document_id'] = document_id
            
            search_results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                filter=filter_dict if filter_dict else None
            )
            
            # Format results
            results = []
            for match in search_results['matches']:
                results.append({
                    'text': match['metadata']['text'],
                    'relevance_score': match['score'],
                    'chunk_id': match['metadata']['chunk_id'],
                    'document_id': match['metadata']['document_id'],
                    'chunk_index': match['metadata']['chunk_index']
                })
            
            return self.create_success_output({
                'query': query,
                'results': results,
                'total_found': len(results)
            })
            
        except Exception as e:
            return self.create_error_output(f"Search failed: {str(e)}")
    
    async def _get_document_chunks(self, input_data: AgentInput, state: AgentState) -> AgentOutput:
        """Get all chunks for a specific document"""
        try:
            document_id = input_data.document_id
            if not document_id:
                return self.create_error_output("Document ID is required")
            
            # Query all chunks for the document
            results = self.index.query(
                vector=[0] * 384,  # Dummy vector
                top_k=10000,  # Large number to get all chunks
                include_metadata=True,
                filter={'document_id': document_id}
            )
            
            chunks = []
            for match in results['matches']:
                chunks.append({
                    'chunk_id': match['metadata']['chunk_id'],
                    'chunk_index': match['metadata']['chunk_index'],
                    'text': match['metadata']['text'],
                    'word_count': match['metadata']['word_count']
                })
            
            # Sort by chunk index
            chunks.sort(key=lambda x: x['chunk_index'])
            
            return self.create_success_output({
                'document_id': document_id,
                'chunks': chunks,
                'total_chunks': len(chunks)
            })
            
        except Exception as e:
            return self.create_error_output(f"Failed to get document chunks: {str(e)}")