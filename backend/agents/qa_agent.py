"""
Q&A Agent
Handles natural language questions about legal documents with retrieval and citations
"""
import re
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
try:
    import google.generativeai as genai
except ImportError:
    print("Warning: google.generativeai not installed")
    genai = None

from .base_agent import BaseAgent, AgentInput, AgentOutput, AgentState
from .document_retrieval_agent import DocumentRetrievalAgent
from models.database import get_db, Document
from config import settings


class QAAgent(BaseAgent):
    """Agent responsible for answering questions about legal documents"""
    
    def __init__(self):
        super().__init__("QAAgent")
        
        # Initialize Gemini if available
        if genai and settings.google_api_key:
            try:
                genai.configure(api_key=settings.google_api_key)
                self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
            except Exception as e:
                print(f"Warning: Failed to initialize Gemini: {str(e)}")
                self.gemini_model = None
        else:
            print("Warning: Google GenerativeAI not available or API key not set")
            self.gemini_model = None
        
        # Initialize document retrieval agent
        self.retrieval_agent = DocumentRetrievalAgent()
    
    async def _get_document_text(self, document_id: str) -> str:
        """Get document text from extracted file"""
        try:
            print(f"Q&A Agent: Getting document text for ID: {document_id}")
            
            # Get document record from database to find file path
            db = next(get_db())
            doc_record = db.query(Document).filter(Document.id == document_id).first()
            
            if not doc_record:
                print(f"Q&A Agent: No document record found for ID: {document_id}")
                return ""
            
            print(f"Q&A Agent: Found document record - filename: {doc_record.filename}")
            print(f"Q&A Agent: Document metadata: {doc_record.document_metadata}")
            
            # Try to get extracted text path from metadata
            metadata = doc_record.document_metadata or {}
            text_storage_path = metadata.get('text_storage_path')
            
            if text_storage_path and os.path.exists(text_storage_path):
                print(f"Q&A Agent: Found text storage path: {text_storage_path}")
                with open(text_storage_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                    print(f"Q&A Agent: Read {len(text)} characters from stored text file")
                    return text
            else:
                print(f"Q&A Agent: Text storage path not found or doesn't exist: {text_storage_path}")
            
            # Fallback: try to construct path
            upload_path = f"uploads/{doc_record.filename}"
            extracted_path = f"{upload_path}.extracted.txt"
            
            print(f"Q&A Agent: Trying fallback path: {extracted_path}")
            
            if os.path.exists(extracted_path):
                print(f"Q&A Agent: Found fallback extracted text file")
                with open(extracted_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                    print(f"Q&A Agent: Read {len(text)} characters from fallback text file")
                    return text
            else:
                print(f"Q&A Agent: Fallback text file doesn't exist: {extracted_path}")
            
            return ""
            
        except Exception as e:
            print(f"Q&A Agent: Error getting document text: {str(e)}")
            return ""
        finally:
            if 'db' in locals():
                db.close()
    
    async def _generate_with_gemini(self, prompt: str) -> str:
        """Generate content with Gemini, with fallback if not available"""
        if not self.gemini_model:
            return "Q&A service is not available. Please configure Google AI API key."
        
        try:
            response = await self.gemini_model.generate_content_async(prompt)
            return response.text
        except Exception as e:
            return f"Error generating answer: {str(e)}"
    
    async def run(self, input_data: AgentInput, state: AgentState) -> AgentOutput:
        """Main method to answer questions about documents"""
        try:
            question = input_data.query
            document_id = input_data.document_id
            conversation_history = input_data.parameters.get('conversation_history', [])
            
            if not question.strip():
                return self.create_error_output("Question cannot be empty")
            
            if not document_id:
                return self.create_error_output("Document ID is required for Q&A")
            
            # Answer the question
            return await self._answer_question(question, document_id, conversation_history, state)
                
        except Exception as e:
            return self.create_error_output(f"Error in QAAgent: {str(e)}")
    
    async def _answer_question(self, question: str, document_id: str, conversation_history: List[Dict], state: AgentState) -> AgentOutput:
        """Answer a question about the document"""
        try:
            # Analyze the question to understand intent
            question_analysis = await self._analyze_question(question)
            
            # Retrieve relevant content
            relevant_content = await self._retrieve_relevant_content(question, document_id)
            
            if not relevant_content:
                return self.create_error_output("No relevant content found for the question")
            
            # Generate context-aware answer
            answer_data = await self._generate_answer(
                question, 
                relevant_content, 
                conversation_history, 
                question_analysis
            )
            
            # Generate related questions
            related_questions = await self._generate_related_questions(question, relevant_content)
            
            # Create citations
            citations = self._create_citations(relevant_content)
            
            result = {
                'question': question,
                'answer': answer_data['answer'],
                'confidence_score': answer_data['confidence'],
                'citations': citations,
                'related_questions': related_questions,
                'question_type': question_analysis['type'],
                'answer_timestamp': datetime.now().isoformat(),
                'sources_used': len(relevant_content),
                'reasoning': answer_data.get('reasoning', '')
            }
            
            return self.create_success_output(result, citations=citations)
            
        except Exception as e:
            return self.create_error_output(f"Failed to answer question: {str(e)}")
    
    async def _analyze_question(self, question: str) -> Dict[str, Any]:
        """Analyze the question to understand its type and intent"""
        try:
            # Use patterns to classify question types
            question_lower = question.lower()
            
            question_types = {
                'definition': ['what is', 'what does', 'define', 'meaning of', 'explain'],
                'obligation': ['what must', 'what should', 'obligations', 'required to', 'have to'],
                'consequence': ['what happens if', 'penalty', 'consequences', 'breach', 'violation'],
                'timeline': ['when', 'deadline', 'how long', 'duration', 'timeline'],
                'procedure': ['how to', 'process', 'procedure', 'steps'],
                'comparison': ['difference', 'compare', 'versus', 'vs'],
                'termination': ['terminate', 'end', 'cancel', 'exit'],
                'payment': ['pay', 'cost', 'fee', 'price', 'amount'],
                'rights': ['rights', 'can i', 'allowed to', 'permitted'],
                'general': []  # fallback
            }
            
            detected_type = 'general'
            for q_type, keywords in question_types.items():
                if any(keyword in question_lower for keyword in keywords):
                    detected_type = q_type
                    break
            
            # Extract key entities (simple approach)
            entities = re.findall(r'\b[A-Z][a-z]+\b', question)
            
            # Determine complexity
            complexity = 'simple'
            if len(question.split()) > 15 or '?' in question[:-1]:  # Multiple questions
                complexity = 'complex'
            
            return {
                'type': detected_type,
                'entities': entities,
                'complexity': complexity,
                'word_count': len(question.split()),
                'has_conditions': any(word in question_lower for word in ['if', 'when', 'unless', 'provided that'])
            }
            
        except Exception:
            return {
                'type': 'general',
                'entities': [],
                'complexity': 'simple',
                'word_count': len(question.split()),
                'has_conditions': False
            }
    
    async def _retrieve_relevant_content(self, question: str, document_id: str) -> List[Dict[str, Any]]:
        """Retrieve relevant content for answering the question"""
        try:
            print(f"Q&A Agent: Retrieving content for document {document_id}, question: {question[:100]}...")
            
            # Get full document text
            full_text = await self._get_document_text(document_id)
            
            if not full_text:
                print(f"Q&A Agent: No document text found for document {document_id}")
                return []
            
            print(f"Q&A Agent: Found document text, length: {len(full_text)} characters")
            
            # Extract keywords from question
            question_keywords = self._extract_keywords(question)
            print(f"Q&A Agent: Extracted keywords: {question_keywords}")
            
            # Split document into paragraphs for relevance scoring
            paragraphs = [p.strip() for p in full_text.split('\n\n') if p.strip() and len(p.strip()) > 50]
            print(f"Q&A Agent: Split into {len(paragraphs)} paragraphs")
            
            relevant_content = []
            
            for i, paragraph in enumerate(paragraphs):
                paragraph_lower = paragraph.lower()
                question_lower = question.lower()
                
                # Calculate relevance score
                relevance_score = 0.0
                
                # Direct question phrase matching (highest weight)
                question_words = question_lower.split()
                for word in question_words:
                    if word in paragraph_lower and len(word) > 3:
                        relevance_score += 2.0
                
                # Keyword matching (medium weight)
                for keyword in question_keywords:
                    if keyword.lower() in paragraph_lower:
                        relevance_score += 1.0
                
                # Normalize by paragraph length (to avoid bias toward long paragraphs)
                relevance_score = relevance_score / max(len(paragraph.split()) / 50, 1)
                
                if relevance_score > 0.5:  # Threshold for relevance
                    relevant_content.append({
                        'text': paragraph,
                        'relevance_score': min(relevance_score, 10.0),  # Cap at 10
                        'chunk_id': f'para_{i}',
                        'start_char': full_text.find(paragraph),
                        'word_count': len(paragraph.split())
                    })
            
            print(f"Q&A Agent: Found {len(relevant_content)} relevant paragraphs")
            
            # If no content found with strict threshold, lower the threshold
            if not relevant_content:
                print("Q&A Agent: No content found with strict threshold, trying with lower threshold...")
                for i, paragraph in enumerate(paragraphs[:20]):  # Check first 20 paragraphs
                    paragraph_lower = paragraph.lower()
                    question_lower = question.lower()
                    
                    # Calculate relevance score with more lenient criteria
                    relevance_score = 0.0
                    
                    # Check for any word overlap
                    question_words = [w for w in question_lower.split() if len(w) > 2]
                    for word in question_words:
                        if word in paragraph_lower:
                            relevance_score += 1.0
                    
                    if relevance_score > 0.1 or i < 5:  # Very low threshold or first 5 paragraphs
                        relevant_content.append({
                            'text': paragraph,
                            'relevance_score': max(relevance_score, 0.1),
                            'chunk_id': f'para_{i}',
                            'start_char': full_text.find(paragraph),
                            'word_count': len(paragraph.split())
                        })
                
                print(f"Q&A Agent: With lower threshold, found {len(relevant_content)} relevant paragraphs")
            
            # Sort by relevance and return top results
            relevant_content.sort(key=lambda x: x['relevance_score'], reverse=True)
            return relevant_content[:10]  # Top 10 most relevant paragraphs
            
        except Exception as e:
            print(f"Q&A Agent: Error retrieving relevant content: {str(e)}")
            return []
            
            return relevant_chunks[:10]  # Top 10 most relevant chunks
            
        except Exception as e:
            return []
    
    def _extract_keywords(self, question: str) -> List[str]:
        """Extract important keywords from the question"""
        # Remove common stop words
        stop_words = {
            'what', 'how', 'when', 'where', 'why', 'who', 'which', 'can', 'is', 'are', 
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 
            'with', 'by', 'about', 'this', 'that', 'these', 'those', 'i', 'me', 'my', 
            'mine', 'we', 'us', 'our', 'ours', 'if', 'then', 'than', 'do', 'does', 'did'
        }
        
        words = re.findall(r'\b\w+\b', question.lower())
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        return keywords
    
    async def _generate_answer(self, question: str, relevant_content: List[Dict], conversation_history: List[Dict], question_analysis: Dict) -> Dict[str, Any]:
        """Generate a comprehensive answer using relevant content"""
        try:
            # Combine relevant content
            context_text = ""
            for i, chunk in enumerate(relevant_content):
                context_text += f"[Source {i+1}]: {chunk['text']}\n\n"
            
            # Build conversation context
            conversation_context = ""
            if conversation_history:
                recent_history = conversation_history[-3:]  # Last 3 interactions
                for interaction in recent_history:
                    if 'input' in interaction and 'output' in interaction:
                        conv_q = interaction['input'].get('query', '')
                        conv_a = interaction['output'].get('result', {}).get('answer', '')
                        conversation_context += f"Previous Q: {conv_q[:100]}...\nPrevious A: {conv_a[:100]}...\n\n"
            
            # Create specialized prompt based on question type
            prompt = self._create_specialized_prompt(question, question_analysis, context_text, conversation_context)
            
            answer_text = await self._generate_with_gemini(prompt)
            
            # Calculate confidence score
            confidence = self._calculate_answer_confidence(question, answer_text, relevant_content)
            
            # Extract reasoning if present
            reasoning = self._extract_reasoning(answer_text)
            
            return {
                'answer': answer_text,
                'confidence': confidence,
                'reasoning': reasoning
            }
            
        except Exception as e:
            return {
                'answer': f"I apologize, but I encountered an error while processing your question: {str(e)}",
                'confidence': 0.0,
                'reasoning': "Error in answer generation"
            }
    
    def _create_specialized_prompt(self, question: str, question_analysis: Dict, context: str, conversation_context: str) -> str:
        """Create a specialized prompt based on question type"""
        
        base_instructions = """
        You are a legal document assistant. Answer the user's question based ONLY on the provided document content.
        Be precise, clear, and helpful. If the document doesn't contain enough information to answer fully, say so.
        Always cite which source(s) you're referring to using [Source X] notation.
        """
        
        question_type = question_analysis['type']
        
        type_specific_instructions = {
            'definition': "Focus on providing clear definitions and explanations of legal terms or concepts.",
            'obligation': "Clearly identify who has what obligations and what the consequences are for not meeting them.",
            'consequence': "Explain what happens in specific scenarios, including penalties, procedures, and outcomes.",
            'timeline': "Provide specific dates, deadlines, and timeframes. Be precise about timing requirements.",
            'procedure': "Outline step-by-step processes and requirements clearly.",
            'termination': "Explain termination conditions, procedures, notice requirements, and consequences.",
            'payment': "Detail payment amounts, schedules, methods, and any penalties for late payment.",
            'rights': "Clearly explain what rights parties have and any limitations or conditions.",
            'comparison': "Compare and contrast different options, terms, or scenarios clearly.",
            'general': "Provide a comprehensive answer addressing all aspects of the question."
        }
        
        specific_instruction = type_specific_instructions.get(question_type, type_specific_instructions['general'])
        
        prompt = f"""
        {base_instructions}
        
        {specific_instruction}
        
        Question: {question}
        
        Previous conversation context:
        {conversation_context}
        
        Relevant document content:
        {context}
        
        Please provide a clear, accurate answer based on the document content above.
        """
        
        return prompt
    
    def _calculate_answer_confidence(self, question: str, answer: str, relevant_content: List[Dict]) -> float:
        """Calculate confidence score for the answer"""
        confidence = 0.5  # Base confidence
        
        # Higher confidence if we found highly relevant content
        if relevant_content:
            avg_relevance = sum(chunk['relevance_score'] for chunk in relevant_content) / len(relevant_content)
            confidence += avg_relevance * 0.3
        
        # Higher confidence if answer is detailed and structured
        if len(answer.split()) > 20 and '[Source' in answer:
            confidence += 0.2
        
        # Lower confidence if answer contains uncertainty phrases
        uncertainty_phrases = ['might', 'possibly', 'unclear', 'not enough information', 'may be']
        if any(phrase in answer.lower() for phrase in uncertainty_phrases):
            confidence -= 0.2
        
        # Higher confidence if answer directly addresses question keywords
        question_keywords = self._extract_keywords(question)
        answer_lower = answer.lower()
        keyword_coverage = sum(1 for keyword in question_keywords if keyword in answer_lower) / max(len(question_keywords), 1)
        confidence += keyword_coverage * 0.2
        
        return max(0.0, min(confidence, 1.0))
    
    def _extract_reasoning(self, answer: str) -> str:
        """Extract reasoning or explanation from the answer"""
        # Look for explanation patterns
        reasoning_indicators = ['because', 'since', 'as stated', 'according to', 'based on']
        
        sentences = answer.split('.')
        reasoning_sentences = []
        
        for sentence in sentences:
            if any(indicator in sentence.lower() for indicator in reasoning_indicators):
                reasoning_sentences.append(sentence.strip())
        
        return '. '.join(reasoning_sentences) if reasoning_sentences else ""
    
    async def _generate_related_questions(self, original_question: str, relevant_content: List[Dict]) -> List[str]:
        """Generate related questions that users might ask"""
        try:
            # Combine some relevant content for context
            context_sample = ""
            for chunk in relevant_content[:3]:
                context_sample += chunk['text'][:200] + "... "
            
            prompt = f"""
            Based on the original question and the document content, suggest 3-4 related questions that users might want to ask next.
            Make the questions specific and practical.
            
            Original question: {original_question}
            Document context: {context_sample}
            
            Return only the questions, one per line, without numbering or bullets.
            """
            
            response_text = await self._generate_with_gemini(prompt)
            
            # Parse related questions
            questions = []
            for line in response_text.split('\n'):
                line = line.strip()
                if line and not line.startswith(('1.', '2.', '3.', '4.', '-', '•')):
                    # Clean up formatting
                    line = re.sub(r'^\d+\.?\s*', '', line)  # Remove numbering
                    line = re.sub(r'^[-•]\s*', '', line)    # Remove bullets
                    if line.endswith('?') or len(line) > 10:
                        questions.append(line)
                elif line.startswith(('1.', '2.', '3.', '4.', '-', '•')):
                    # Remove formatting and add
                    clean_line = re.sub(r'^[-•\d\.]\s*', '', line)
                    if clean_line:
                        questions.append(clean_line)
            
            return questions[:4]  # Return max 4 questions
            
        except Exception:
            # Fallback: generate generic related questions based on question type
            return self._generate_fallback_related_questions(original_question)
    
    def _generate_fallback_related_questions(self, original_question: str) -> List[str]:
        """Generate fallback related questions"""
        question_lower = original_question.lower()
        
        generic_questions = []
        
        if any(word in question_lower for word in ['payment', 'pay', 'cost']):
            generic_questions = [
                "What happens if payment is late?",
                "What are the accepted payment methods?",
                "Are there any additional fees or charges?"
            ]
        elif any(word in question_lower for word in ['terminate', 'end', 'cancel']):
            generic_questions = [
                "What notice is required for termination?",
                "What are the consequences of early termination?",
                "Can the agreement be renewed after termination?"
            ]
        elif any(word in question_lower for word in ['breach', 'violation']):
            generic_questions = [
                "What constitutes a material breach?",
                "What remedies are available for breach?",
                "Is there a cure period for violations?"
            ]
        else:
            generic_questions = [
                "What are the main obligations in this document?",
                "What are the termination conditions?",
                "What happens in case of a dispute?"
            ]
        
        return generic_questions
    
    def _create_citations(self, relevant_content: List[Dict]) -> List[Dict[str, Any]]:
        """Create properly formatted citations"""
        citations = []
        
        # Find max relevance score for normalization
        max_score = max(chunk['relevance_score'] for chunk in relevant_content) if relevant_content else 1.0
        
        for i, chunk in enumerate(relevant_content):
            # Normalize relevance score to 0-1 range
            normalized_score = min(chunk['relevance_score'] / max_score, 1.0) if max_score > 0 else 0.1
            
            citation = {
                'text': chunk['text'][:200] + "..." if len(chunk['text']) > 200 else chunk['text'],
                'relevance_score': normalized_score,
                'chunk_id': chunk['chunk_id'],
                'source_number': i + 1,
                'page_number': None,  # Would need to be extracted from metadata
                'section': None       # Would need to be extracted from metadata
            }
            citations.append(citation)
        
        return citations