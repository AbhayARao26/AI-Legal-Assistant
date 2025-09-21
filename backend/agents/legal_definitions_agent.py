"""
Legal Definitions Agent
Detects complex legal terms and explains them in simple, layman-friendly language
"""
import re
from typing import List, Dict, Any, Optional
from datetime import datetime
import google.generativeai as genai

from .base_agent import BaseAgent, AgentInput, AgentOutput, AgentState
from .document_retrieval_agent import DocumentRetrievalAgent
from config import settings


class LegalDefinitionsAgent(BaseAgent):
    """Agent responsible for defining and explaining legal terms"""
    
    def __init__(self):
        super().__init__("LegalDefinitionsAgent")
        
        # Initialize Gemini
        genai.configure(api_key=settings.google_api_key)
        self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Initialize document retrieval agent
        self.retrieval_agent = DocumentRetrievalAgent()
        
        # Common legal terms dictionary
        self.legal_terms_dict = self._load_legal_terms_dictionary()
    
    async def run(self, input_data: AgentInput, state: AgentState) -> AgentOutput:
        """Main method to define legal terms"""
        try:
            terms = input_data.parameters.get('terms', [])
            document_id = input_data.document_id
            context = input_data.parameters.get('context', '')
            
            if terms:
                # Define specific terms
                return await self._define_specific_terms(terms, document_id, context, state)
            else:
                # Auto-detect and define terms in document
                return await self._auto_detect_and_define_terms(input_data, state)
                
        except Exception as e:
            return self.create_error_output(f"Error in LegalDefinitionsAgent: {str(e)}")
    
    async def _define_specific_terms(self, terms: List[str], document_id: Optional[str], context: str, state: AgentState) -> AgentOutput:
        """Define specific legal terms"""
        try:
            definitions = []
            
            for term in terms:
                definition = await self._get_term_definition(term, document_id, context, state)
                if definition:
                    definitions.append(definition)
            
            result = {
                'requested_terms': terms,
                'definitions': definitions,
                'context_document_id': document_id,
                'total_defined': len(definitions)
            }
            
            return self.create_success_output(result)
            
        except Exception as e:
            return self.create_error_output(f"Failed to define terms: {str(e)}")
    
    async def _auto_detect_and_define_terms(self, input_data: AgentInput, state: AgentState) -> AgentOutput:
        """Automatically detect and define legal terms in document"""
        try:
            document_id = input_data.document_id
            if not document_id:
                return self.create_error_output("Document ID is required for auto-detection")
            
            # Get document content
            chunks_input = AgentInput(
                query="",
                document_id=document_id,
                parameters={'action': 'get_chunks'}
            )
            chunks_result = await self.retrieval_agent.run(chunks_input, state)
            
            if not chunks_result.success:
                return self.create_error_output(f"Failed to retrieve document: {chunks_result.error_message}")
            
            chunks = chunks_result.result['chunks']
            
            # Detect legal terms
            detected_terms = await self._detect_legal_terms(chunks)
            
            # Define detected terms
            definitions = []
            for term in detected_terms:
                definition = await self._get_term_definition(term, document_id, "", state)
                if definition:
                    definitions.append(definition)
            
            result = {
                'document_id': document_id,
                'detected_terms': detected_terms,
                'definitions': definitions,
                'total_detected': len(detected_terms),
                'total_defined': len(definitions),
                'coverage': len(definitions) / max(len(detected_terms), 1)
            }
            
            return self.create_success_output(result)
            
        except Exception as e:
            return self.create_error_output(f"Failed to auto-detect terms: {str(e)}")
    
    async def _get_term_definition(self, term: str, document_id: Optional[str], context: str, state: AgentState) -> Optional[Dict[str, Any]]:
        """Get definition for a specific legal term"""
        try:
            # First check if term is in our dictionary
            dictionary_definition = self.legal_terms_dict.get(term.lower())
            
            # Get document-specific context if available
            document_context = ""
            if document_id:
                document_context = await self._get_term_context_from_document(term, document_id, state)
            
            # Generate comprehensive definition using Gemini
            ai_definition = await self._generate_ai_definition(term, document_context, context)
            
            # Extract related terms
            related_terms = self._find_related_terms(term)
            
            # Generate examples
            examples = await self._generate_examples(term, document_context)
            
            definition = {
                'term': term,
                'legal_definition': dictionary_definition.get('legal', '') if dictionary_definition else ai_definition.get('legal', ''),
                'simple_definition': ai_definition.get('simple', ''),
                'context_explanation': ai_definition.get('context', ''),
                'examples': examples,
                'related_terms': related_terms,
                'source': 'ai_generated' if not dictionary_definition else 'dictionary_enhanced',
                'confidence_score': self._calculate_confidence_score(term, dictionary_definition, ai_definition),
                'document_context': document_context[:500] if document_context else '',
                'updated_at': datetime.now().isoformat()
            }
            
            return definition
            
        except Exception as e:
            return None
    
    async def _detect_legal_terms(self, chunks: List[Dict[str, Any]]) -> List[str]:
        """Detect legal terms in document chunks"""
        # Combine chunks for analysis
        combined_text = ""
        for chunk in chunks[:10]:  # Analyze first 10 chunks for efficiency
            combined_text += chunk['text'] + " "
        
        # Use pattern matching for common legal terms
        detected_terms = set()
        
        # Add terms from our dictionary that appear in the text
        text_lower = combined_text.lower()
        for term in self.legal_terms_dict.keys():
            if term in text_lower:
                detected_terms.add(term.title())
        
        # Use regex patterns for legal term detection
        legal_patterns = [
            r'\b([A-Za-z\s]+(?:agreement|contract|clause|provision|term))\b',
            r'\b([A-Za-z\s]*(?:liability|indemnification|arbitration|mediation))\b',
            r'\b([A-Za-z\s]*(?:damages|penalty|breach|default))\b',
            r'\b([A-Za-z\s]*(?:jurisdiction|governing law|force majeure))\b'
        ]
        
        for pattern in legal_patterns:
            matches = re.findall(pattern, combined_text, re.IGNORECASE)
            for match in matches:
                if len(match.strip()) > 3 and len(match.strip()) < 50:
                    detected_terms.add(match.strip().title())
        
        # Use AI to detect additional terms
        ai_detected = await self._ai_detect_terms(combined_text[:5000])
        detected_terms.update(ai_detected)
        
        # Filter and clean terms
        filtered_terms = []
        for term in detected_terms:
            if self._is_valid_legal_term(term):
                filtered_terms.append(term)
        
        return sorted(list(set(filtered_terms)))[:20]  # Return top 20 terms
    
    async def _ai_detect_terms(self, text: str) -> List[str]:
        """Use AI to detect legal terms in text"""
        try:
            prompt = f"""
            Identify important legal terms and jargon in this document that would benefit from explanation to a non-lawyer. 
            Focus on complex legal concepts, technical terms, and industry-specific language.
            
            Return only a comma-separated list of terms, no explanations.
            Limit to the 15 most important terms.
            
            Document excerpt:
            {text}
            """
            
            response = await self.gemini_model.generate_content_async(prompt)
            
            # Parse the response
            terms_text = response.text.strip()
            terms = [term.strip() for term in terms_text.split(',')]
            
            # Clean and validate terms
            clean_terms = []
            for term in terms:
                term = term.strip().strip('"').strip("'")
                if self._is_valid_legal_term(term):
                    clean_terms.append(term)
            
            return clean_terms
            
        except Exception as e:
            return []
    
    async def _get_term_context_from_document(self, term: str, document_id: str, state: AgentState) -> str:
        """Get context for a term from the specific document"""
        try:
            # Search for the term in the document
            search_input = AgentInput(
                query=term,
                document_id=document_id,
                parameters={'action': 'search', 'top_k': 3}
            )
            
            search_result = await self.retrieval_agent.run(search_input, state)
            
            if search_result.success and search_result.result['results']:
                context_chunks = []
                for result in search_result.result['results']:
                    context_chunks.append(result['text'])
                
                return " ... ".join(context_chunks)
            
            return ""
            
        except Exception:
            return ""
    
    async def _generate_ai_definition(self, term: str, document_context: str, additional_context: str) -> Dict[str, str]:
        """Generate AI-powered definition for a term"""
        try:
            prompt = f"""
            Provide a comprehensive explanation of the legal term "{term}" with:
            
            1. LEGAL DEFINITION: The formal legal definition
            2. SIMPLE DEFINITION: Explain it in plain English that anyone can understand
            3. CONTEXT EXPLANATION: How this term applies in the specific context provided
            
            Context from document: {document_context}
            Additional context: {additional_context}
            
            Format your response as:
            LEGAL: [formal definition]
            SIMPLE: [plain English explanation]
            CONTEXT: [how it applies in this situation]
            """
            
            response = await self.gemini_model.generate_content_async(prompt)
            response_text = response.text
            
            # Parse the structured response
            definition = {}
            
            legal_match = re.search(r'LEGAL:\s*(.+?)(?=SIMPLE:|$)', response_text, re.DOTALL)
            simple_match = re.search(r'SIMPLE:\s*(.+?)(?=CONTEXT:|$)', response_text, re.DOTALL)
            context_match = re.search(r'CONTEXT:\s*(.+?)$', response_text, re.DOTALL)
            
            definition['legal'] = legal_match.group(1).strip() if legal_match else ""
            definition['simple'] = simple_match.group(1).strip() if simple_match else ""
            definition['context'] = context_match.group(1).strip() if context_match else ""
            
            # If parsing fails, use the whole response as simple definition
            if not any(definition.values()):
                definition['simple'] = response_text.strip()
            
            return definition
            
        except Exception as e:
            return {
                'legal': f"Error generating definition: {str(e)}",
                'simple': f"Unable to define '{term}' at this time.",
                'context': ""
            }
    
    async def _generate_examples(self, term: str, context: str) -> List[str]:
        """Generate examples for a legal term"""
        try:
            prompt = f"""
            Provide 2-3 simple, real-world examples of how the legal term "{term}" might be used or applied.
            Make the examples relatable and easy to understand.
            
            Context: {context[:500]}
            
            Return as a simple list, one example per line starting with "-"
            """
            
            response = await self.gemini_model.generate_content_async(prompt)
            
            # Parse examples from response
            examples = []
            lines = response.text.split('\n')
            
            for line in lines:
                line = line.strip()
                if line.startswith('-') or line.startswith('•'):
                    example = line[1:].strip()
                    if example:
                        examples.append(example)
                elif line and not line.startswith(('Example', 'For example')):
                    # Sometimes AI doesn't use bullet points
                    examples.append(line)
                
                if len(examples) >= 3:
                    break
            
            return examples
            
        except Exception:
            return []
    
    def _find_related_terms(self, term: str) -> List[str]:
        """Find related legal terms"""
        related = []
        term_lower = term.lower()
        
        # Look for related terms in our dictionary
        for dict_term, dict_info in self.legal_terms_dict.items():
            if dict_term != term_lower:
                # Check if terms share words or are conceptually related
                if any(word in dict_term for word in term_lower.split() if len(word) > 3):
                    related.append(dict_term.title())
                elif 'related' in dict_info and term_lower in dict_info['related']:
                    related.append(dict_term.title())
        
        return related[:5]  # Return top 5 related terms
    
    def _calculate_confidence_score(self, term: str, dictionary_def: Optional[Dict], ai_def: Dict) -> float:
        """Calculate confidence score for the definition"""
        score = 0.5  # Base score
        
        # Higher confidence if term is in dictionary
        if dictionary_def:
            score += 0.3
        
        # Higher confidence if AI generated comprehensive definition
        if ai_def.get('legal') and ai_def.get('simple'):
            score += 0.2
        
        # Check term length and complexity
        if len(term.split()) <= 3:  # Simple terms are more likely to be accurate
            score += 0.1
        
        return min(score, 1.0)
    
    def _is_valid_legal_term(self, term: str) -> bool:
        """Validate if a term is a legitimate legal term"""
        if not term or len(term.strip()) < 3:
            return False
        
        # Filter out common words that aren't legal terms
        common_words = ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']
        if term.lower().strip() in common_words:
            return False
        
        # Check if term contains mostly letters
        if not re.match(r'^[A-Za-z\s\-\']+$', term):
            return False
        
        # Avoid very long terms (likely sentences)
        if len(term.split()) > 6:
            return False
        
        return True
    
    def _load_legal_terms_dictionary(self) -> Dict[str, Dict[str, Any]]:
        """Load a dictionary of common legal terms"""
        # This would typically be loaded from a file or database
        # For now, we'll include a few common terms
        return {
            'indemnification': {
                'legal': 'A contractual obligation by which one party agrees to compensate another party for any losses, damages, or liabilities that may arise.',
                'category': 'contract',
                'related': ['liability', 'damages', 'compensation']
            },
            'force majeure': {
                'legal': 'A clause that frees parties from liability or obligation when extraordinary circumstances beyond their control prevent them from fulfilling their duties.',
                'category': 'contract',
                'related': ['unforeseen circumstances', 'act of god']
            },
            'arbitration': {
                'legal': 'A form of alternative dispute resolution where a neutral third party makes a binding decision to resolve a dispute outside of court.',
                'category': 'dispute resolution',
                'related': ['mediation', 'dispute', 'resolution']
            },
            'jurisdiction': {
                'legal': 'The authority of a court to hear and decide a case, or the geographical area over which a court has authority.',
                'category': 'legal authority',
                'related': ['court', 'authority', 'governing law']
            },
            'breach of contract': {
                'legal': 'The failure to perform any duty or obligation specified in a contract without a legal excuse.',
                'category': 'contract',
                'related': ['default', 'violation', 'non-performance']
            },
            'liquidated damages': {
                'legal': 'A specific amount of money agreed upon in advance by the parties to a contract as compensation for breach.',
                'category': 'damages',
                'related': ['penalty', 'damages', 'compensation']
            },
            'confidentiality': {
                'legal': 'The obligation to keep certain information secret and not disclose it to unauthorized parties.',
                'category': 'information protection',
                'related': ['non-disclosure', 'proprietary', 'trade secret']
            }
        }