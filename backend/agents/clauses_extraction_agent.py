"""
Clauses & Terms Extraction Agent
Extracts structured information like obligations, rights, penalties, timelines, etc.
"""
import json
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
from models.schemas import ClauseType
from models.database import get_db, Document
from config import settings


class ClausesExtractionAgent(BaseAgent):
    """Agent responsible for extracting structured clauses and terms"""
    
    def __init__(self):
        super().__init__("ClausesExtractionAgent")
        
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
            # Get document record from database to find file path
            db = next(get_db())
            doc_record = db.query(Document).filter(Document.id == document_id).first()
            
            if not doc_record:
                return ""
            
            # Try to get extracted text path from metadata
            metadata = doc_record.document_metadata or {}
            text_storage_path = metadata.get('text_storage_path')
            
            if text_storage_path and os.path.exists(text_storage_path):
                with open(text_storage_path, 'r', encoding='utf-8') as f:
                    return f.read()
            
            # Fallback: try to construct path
            upload_path = f"uploads/{doc_record.filename}"
            extracted_path = f"{upload_path}.extracted.txt"
            
            if os.path.exists(extracted_path):
                with open(extracted_path, 'r', encoding='utf-8') as f:
                    return f.read()
            
            return ""
            
        except Exception as e:
            print(f"Error getting document text: {str(e)}")
            return ""
        finally:
            if 'db' in locals():
                db.close()
    
    async def _generate_with_gemini(self, prompt: str) -> str:
        """Generate content with Gemini, with fallback if not available"""
        if not self.gemini_model:
            return "Clause extraction service is not available. Please configure Google AI API key."
        
        try:
            response = await self.gemini_model.generate_content_async(prompt)
            return response.text
        except Exception as e:
            return f"Error extracting clauses: {str(e)}"
    
    async def run(self, input_data: AgentInput, state: AgentState) -> AgentOutput:
        """Main method to extract clauses and terms"""
        try:
            clause_types = input_data.parameters.get('clause_types', [])
            
            if not clause_types:
                # Extract all types
                return await self._extract_all_clauses(input_data, state)
            else:
                # Extract specific clause types
                return await self._extract_specific_clauses(input_data, state, clause_types)
                
        except Exception as e:
            return self.create_error_output(f"Error in ClausesExtractionAgent: {str(e)}")
    
    async def _extract_all_clauses(self, input_data: AgentInput, state: AgentState) -> AgentOutput:
        """Extract all types of clauses from the document"""
        try:
            document_id = input_data.document_id
            if not document_id:
                return self.create_error_output("Document ID is required")
            
            # Get document text
            full_text = await self._get_document_text(document_id)
            
            if not full_text:
                return self.create_error_output("No document text found. Please ensure the document has been processed.")
            
            # Extract all clause types
            extracted_clauses = {}
            
            for clause_type in ClauseType:
                clauses = await self._extract_clause_type(full_text, clause_type)
                if clauses:
                    extracted_clauses[clause_type.value] = clauses
            
            # Also extract general legal terms and conditions
            general_terms = await self._extract_general_terms(full_text)
            if general_terms:
                extracted_clauses['general_terms'] = general_terms
            
            result = {
                'document_id': document_id,
                'extracted_clauses': extracted_clauses,
                'extraction_timestamp': datetime.now().isoformat(),
                'total_clause_types': len(extracted_clauses),
                'summary': self._generate_extraction_summary(extracted_clauses)
            }
            
            return self.create_success_output(result)
            
        except Exception as e:
            return self.create_error_output(f"Failed to extract clauses: {str(e)}")
    
    async def _extract_specific_clauses(self, input_data: AgentInput, state: AgentState, clause_types: List[str]) -> AgentOutput:
        """Extract specific types of clauses"""
        try:
            document_id = input_data.document_id
            if not document_id:
                return self.create_error_output("Document ID is required")
            
            # Get document text
            full_text = await self._get_document_text(document_id)
            
            if not full_text:
                return self.create_error_output("No document text found. Please ensure the document has been processed.")
            
            # Extract specified clause types
            extracted_clauses = {}
            
            for clause_type_str in clause_types:
                try:
                    clause_type = ClauseType(clause_type_str)
                    clauses = await self._extract_clause_type(full_text, clause_type)
                    if clauses:
                        extracted_clauses[clause_type.value] = clauses
                except ValueError:
                    # Invalid clause type, skip
                    continue
            
            result = {
                'document_id': document_id,
                'requested_types': clause_types,
                'extracted_clauses': extracted_clauses,
                'extraction_timestamp': datetime.now().isoformat(),
                'found_types': len(extracted_clauses)
            }
            
            return self.create_success_output(result)
            
        except Exception as e:
            return self.create_error_output(f"Failed to extract specific clauses: {str(e)}")
    
    async def _extract_clause_type(self, text: str, clause_type: ClauseType) -> List[Dict[str, Any]]:
        """Extract a specific type of clause"""
        try:
            # Define extraction prompts for each clause type
            extraction_prompts = {
                ClauseType.OBLIGATION: """
                Extract all obligations from this legal document. For each obligation, provide:
                1. The party responsible (who must do it)
                2. What must be done
                3. When it must be done (deadline/timeline)
                4. Consequences of non-compliance
                5. The exact text of the obligation
                
                Return as JSON array with fields: party, obligation, timeline, consequences, text, importance_score (1-10)
                """,
                
                ClauseType.RIGHT: """
                Extract all rights from this legal document. For each right, provide:
                1. The party who has the right
                2. What the right allows them to do
                3. Any conditions or limitations
                4. Duration of the right
                5. The exact text defining the right
                
                Return as JSON array with fields: party, right, conditions, duration, text, importance_score (1-10)
                """,
                
                ClauseType.PENALTY: """
                Extract all penalties, fines, and consequences from this legal document. For each penalty, provide:
                1. What triggers the penalty
                2. The specific penalty or consequence
                3. Who is responsible for paying/suffering it
                4. How the penalty is calculated
                5. The exact text describing the penalty
                
                Return as JSON array with fields: trigger, penalty, responsible_party, calculation, text, importance_score (1-10)
                """,
                
                ClauseType.TIMELINE: """
                Extract all important dates, deadlines, and timelines from this legal document. For each timeline, provide:
                1. What event or action is tied to the date
                2. The specific date or timeline
                3. Who is responsible
                4. What happens if the deadline is missed
                5. The exact text containing the timeline
                
                Return as JSON array with fields: event, date_timeline, responsible_party, consequences, text, importance_score (1-10)
                """,
                
                ClauseType.TERMINATION: """
                Extract all termination conditions and procedures from this legal document. For each termination clause, provide:
                1. Conditions that allow termination
                2. Notice period required
                3. Procedures to follow
                4. Consequences of termination
                5. The exact text of the termination clause
                
                Return as JSON array with fields: conditions, notice_period, procedures, consequences, text, importance_score (1-10)
                """,
                
                ClauseType.RENEWAL: """
                Extract all renewal conditions and procedures from this legal document. For each renewal clause, provide:
                1. Automatic vs manual renewal
                2. Notice requirements
                3. Terms that may change upon renewal
                4. Renewal period/duration
                5. The exact text of the renewal clause
                
                Return as JSON array with fields: type, notice_requirements, changed_terms, duration, text, importance_score (1-10)
                """,
                
                ClauseType.PAYMENT: """
                Extract all payment terms and conditions from this legal document. For each payment clause, provide:
                1. Amount or calculation method
                2. Payment schedule/due dates
                3. Payment method requirements
                4. Late payment penalties
                5. The exact text of the payment clause
                
                Return as JSON array with fields: amount, schedule, method, penalties, text, importance_score (1-10)
                """,
                
                ClauseType.INDEMNIFICATION: """
                Extract all indemnification clauses from this legal document. For each clause, provide:
                1. Who provides indemnification
                2. Who is protected
                3. What is covered
                4. Limitations or exclusions
                5. The exact text of the indemnification clause
                
                Return as JSON array with fields: indemnifier, protected_party, coverage, limitations, text, importance_score (1-10)
                """,
                
                ClauseType.LIABILITY: """
                Extract all liability clauses from this legal document. For each clause, provide:
                1. Type of liability (limited, unlimited, excluded)
                2. What damages are covered/excluded
                3. Liability caps or limits
                4. Conditions for liability
                5. The exact text of the liability clause
                
                Return as JSON array with fields: type, coverage, limits, conditions, text, importance_score (1-10)
                """,
                
                ClauseType.CONFIDENTIALITY: """
                Extract all confidentiality and non-disclosure clauses from this legal document. For each clause, provide:
                1. What information is considered confidential
                2. Obligations for handling confidential information
                3. Duration of confidentiality obligations
                4. Exceptions to confidentiality
                5. The exact text of the confidentiality clause
                
                Return as JSON array with fields: scope, obligations, duration, exceptions, text, importance_score (1-10)
                """
            }
            
            prompt = extraction_prompts.get(clause_type, "")
            if not prompt:
                return []
            
            # Search for relevant content first
            search_keywords = self._get_search_keywords(clause_type)
            relevant_text = await self._get_relevant_content(text, search_keywords)
            
            if not relevant_text:
                return []
            
            full_prompt = f"""
            {prompt}
            
            Document text:
            {relevant_text[:8000]}  # Limit text length for API
            
            IMPORTANT: Return ONLY a valid JSON array. No explanatory text before or after.
            """
            
            response_text = await self._generate_with_gemini(full_prompt)
            
            # Parse JSON response
            try:
                # Clean the response to extract JSON
                response_text = response_text.strip()
                
                # Remove markdown formatting if present
                if response_text.startswith('```json'):
                    response_text = response_text[7:]
                if response_text.endswith('```'):
                    response_text = response_text[:-3]
                
                clauses = json.loads(response_text)
                
                # Add metadata to each clause
                for clause in clauses:
                    clause['clause_type'] = clause_type.value
                    clause['extracted_at'] = datetime.now().isoformat()
                    # Remove chunk references since we're not using chunks
                
                return clauses
                
            except json.JSONDecodeError:
                # If JSON parsing fails, try to extract information manually
                return self._fallback_extraction(response_text, clause_type)
                
        except Exception as e:
            return []
    
    def _get_search_keywords(self, clause_type: ClauseType) -> List[str]:
        """Get search keywords for each clause type"""
        keywords_map = {
            ClauseType.OBLIGATION: ["shall", "must", "required", "obligation", "duty", "responsible"],
            ClauseType.RIGHT: ["right", "entitled", "may", "authority", "privilege"],
            ClauseType.PENALTY: ["penalty", "fine", "damages", "liquidated", "breach", "default"],
            ClauseType.TIMELINE: ["date", "deadline", "within", "days", "months", "years", "timeline"],
            ClauseType.TERMINATION: ["terminate", "termination", "end", "cancel", "expiry", "dissolution"],
            ClauseType.RENEWAL: ["renew", "renewal", "extend", "extension", "automatic"],
            ClauseType.PAYMENT: ["payment", "pay", "fee", "amount", "invoice", "remuneration"],
            ClauseType.INDEMNIFICATION: ["indemnify", "indemnification", "hold harmless", "defend"],
            ClauseType.LIABILITY: ["liability", "liable", "damages", "loss", "limitation", "exclude"],
            ClauseType.CONFIDENTIALITY: ["confidential", "non-disclosure", "proprietary", "secret"]
        }
        
        return keywords_map.get(clause_type, [])
    
    async def _get_relevant_content(self, full_text: str, keywords: List[str]) -> str:
        """Get content relevant to specific keywords"""
        # Split text into paragraphs
        paragraphs = [p.strip() for p in full_text.split('\n\n') if p.strip()]
        relevant_paragraphs = []
        
        for paragraph in paragraphs:
            paragraph_lower = paragraph.lower()
            if any(keyword.lower() in paragraph_lower for keyword in keywords):
                # Count keyword matches for relevance scoring
                keyword_count = sum(paragraph_lower.count(keyword.lower()) for keyword in keywords)
                relevant_paragraphs.append((paragraph, keyword_count))
        
        # Sort by relevance (number of keyword matches)
        relevant_paragraphs.sort(key=lambda x: x[1], reverse=True)
        
        # Combine top relevant paragraphs
        relevant_text = ""
        for paragraph, _ in relevant_paragraphs[:20]:  # Top 20 relevant paragraphs
            relevant_text += paragraph + "\n\n"
        
        return relevant_text if relevant_text else full_text[:8000]
    
    def _fallback_extraction(self, response_text: str, clause_type: ClauseType) -> List[Dict[str, Any]]:
        """Fallback method when JSON parsing fails"""
        # Simple pattern-based extraction as fallback
        clauses = []
        
        # Split response into lines and look for structured information
        lines = response_text.split('\n')
        current_clause = {}
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Look for key-value patterns
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lower().replace(' ', '_')
                value = value.strip()
                
                if key in ['party', 'obligation', 'right', 'penalty', 'timeline', 'conditions', 'text']:
                    current_clause[key] = value
            
            # If we have enough information, save the clause
            if len(current_clause) >= 2 and 'text' in current_clause:
                current_clause['clause_type'] = clause_type.value
                current_clause['importance_score'] = 5  # Default score
                current_clause['extracted_at'] = datetime.now().isoformat()
                clauses.append(current_clause)
                current_clause = {}
        
        return clauses
    
    async def _extract_general_terms(self, text: str) -> List[Dict[str, Any]]:
        """Extract general terms and conditions"""
        try:
            prompt = """
            Extract the most important general terms and conditions from this legal document that don't fall into specific categories like payments, termination, etc. Focus on:
            1. Governing law and jurisdiction
            2. Dispute resolution methods
            3. Amendment procedures
            4. Force majeure clauses
            5. Assignment restrictions
            6. Entire agreement clauses
            7. Severability provisions
            8. Notice requirements
            
            For each term, provide:
            - term_type: Type of general term
            - description: What the term means
            - implications: Why it's important
            - text: Exact text from document
            - importance_score: 1-10 scale
            
            Return as JSON array.
            """
            
            response_text = await self._generate_with_gemini(f"{prompt}\n\nDocument:\n{text[:8000]}")
            
            try:
                # Clean and parse response
                response_text = response_text.strip()
                if response_text.startswith('```json'):
                    response_text = response_text[7:]
                if response_text.endswith('```'):
                    response_text = response_text[:-3]
                
                terms = json.loads(response_text)
                
                # Add metadata
                for term in terms:
                    term['extracted_at'] = datetime.now().isoformat()
                    # Remove chunk references since we're not using chunks
                
                return terms
                
            except json.JSONDecodeError:
                return []
                
        except Exception as e:
            return []
    
    def _generate_extraction_summary(self, extracted_clauses: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a summary of the extraction results"""
        summary = {
            'total_clause_types_found': len(extracted_clauses),
            'clause_counts': {},
            'high_importance_clauses': 0,
            'coverage_assessment': {}
        }
        
        for clause_type, clauses in extracted_clauses.items():
            summary['clause_counts'][clause_type] = len(clauses)
            
            # Count high importance clauses
            high_importance = sum(1 for clause in clauses if clause.get('importance_score', 0) >= 8)
            summary['high_importance_clauses'] += high_importance
            
            # Assess coverage
            if len(clauses) > 0:
                summary['coverage_assessment'][clause_type] = 'comprehensive' if len(clauses) >= 3 else 'limited'
            else:
                summary['coverage_assessment'][clause_type] = 'none'
        
        return summary