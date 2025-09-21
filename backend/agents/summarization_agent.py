"""
Summarization Agent
Provides hierarchical document summaries (overall, section-wise, clause-level)
"""
import re
import os
from typing import List, Dict, Any, Optional
try:
    import google.generativeai as genai
except ImportError:
    print("Warning: google.generativeai not installed")
    genai = None

from .base_agent import BaseAgent, AgentInput, AgentOutput, AgentState
from .document_retrieval_agent import DocumentRetrievalAgent
from models.schemas import SummaryType
from models.database import get_db, Document
from config import settings


class SummarizationAgent(BaseAgent):
    """Agent responsible for generating document summaries"""
    
    def __init__(self):
        super().__init__("SummarizationAgent")
        
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
        
        # Initialize document retrieval agent for getting content
        self.retrieval_agent = DocumentRetrievalAgent()
    
    async def _generate_with_gemini(self, prompt: str) -> str:
        """Generate content with Gemini, with fallback if not available"""
        if not self.gemini_model:
            return "Summarization service is not available. Please configure Google AI API key."
        
        try:
            response = await self.gemini_model.generate_content_async(prompt)
            return response.text
        except Exception as e:
            return f"Error generating summary: {str(e)}"
    
    async def run(self, input_data: AgentInput, state: AgentState) -> AgentOutput:
        """Main method to generate summaries"""
        try:
            summary_type = input_data.parameters.get('summary_type', SummaryType.OVERALL)
            section_name = input_data.parameters.get('section_name')
            
            if summary_type == SummaryType.OVERALL:
                return await self._generate_overall_summary(input_data, state)
            elif summary_type == SummaryType.SECTION:
                return await self._generate_section_summary(input_data, state, section_name)
            elif summary_type == SummaryType.CLAUSE:
                return await self._generate_clause_summary(input_data, state, section_name)
            else:
                return self.create_error_output(f"Unknown summary type: {summary_type}")
                
        except Exception as e:
            return self.create_error_output(f"Error in SummarizationAgent: {str(e)}")
    
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
    
    async def _generate_overall_summary(self, input_data: AgentInput, state: AgentState) -> AgentOutput:
        """Generate an overall document summary"""
        try:
            document_id = input_data.document_id
            if not document_id:
                return self.create_error_output("Document ID is required")
            
            # Get document text from database or extracted file
            full_text = await self._get_document_text(document_id)
            
            if not full_text:
                return self.create_error_output("No document text found. Please ensure the document has been processed.")
            
            # Limit text for API (keep first 50000 characters)
            if len(full_text) > 50000:
                full_text = full_text[:50000] + "..."
            
            # Generate summary using Gemini
            summary_prompt = f"""
            Please provide a comprehensive summary of this legal document. Include:
            
            1. Document Type and Purpose
            2. Key Parties Involved
            3. Main Terms and Conditions
            4. Important Rights and Obligations
            5. Key Dates and Timelines
            6. Financial Terms (if any)
            7. Termination/Renewal Conditions
            8. Notable Clauses or Restrictions
            9. Risk Assessment (potential issues or concerns)
            10. Overall Recommendation for Review
            
            Make the summary accessible to non-lawyers while maintaining accuracy.
            
            Document:
            {full_text}
            """
            
            summary_text = await self._generate_with_gemini(summary_prompt)
            
            # Extract key points
            key_points = self._extract_key_points(summary_text)
            
            result = {
                'document_id': document_id,
                'summary_type': SummaryType.OVERALL,
                'summary_text': summary_text,
                'key_points': key_points,
                'word_count': len(summary_text.split()),
                'source_length': len(full_text)
            }
            
            return self.create_success_output(result)
            
        except Exception as e:
            return self.create_error_output(f"Failed to generate overall summary: {str(e)}")
    
    async def _generate_section_summary(self, input_data: AgentInput, state: AgentState, section_name: Optional[str] = None) -> AgentOutput:
        """Generate section-wise summaries"""
        try:
            document_id = input_data.document_id
            if not document_id:
                return self.create_error_output("Document ID is required")
            
            # Get document text
            full_text = await self._get_document_text(document_id)
            
            if not full_text:
                return self.create_error_output("No document text found. Please ensure the document has been processed.")
            
            # Identify sections in the document
            sections = self._identify_sections(full_text)
            
            summaries = []
            
            if section_name:
                # Summarize specific section
                section_content = self._get_section_content(sections, section_name)
                if section_content:
                    summary = await self._summarize_section(section_name, section_content)
                    summaries.append(summary)
                else:
                    return self.create_error_output(f"Section '{section_name}' not found")
            else:
                # Summarize all sections
                for section in sections:
                    summary = await self._summarize_section(section['title'], section['content'])
                    summaries.append(summary)
            
            result = {
                'document_id': document_id,
                'summary_type': SummaryType.SECTION,
                'section_name': section_name,
                'summaries': summaries,
                'total_sections': len(sections)
            }
            
            return self.create_success_output(result)
            
        except Exception as e:
            return self.create_error_output(f"Failed to generate section summary: {str(e)}")
    
    async def _generate_clause_summary(self, input_data: AgentInput, state: AgentState, clause_text: Optional[str] = None) -> AgentOutput:
        """Generate clause-level summaries"""
        try:
            document_id = input_data.document_id
            
            if clause_text:
                # Summarize specific clause
                summary = await self._summarize_clause(clause_text)
                result = {
                    'document_id': document_id,
                    'summary_type': SummaryType.CLAUSE,
                    'clause_summary': summary
                }
            else:
                # Find and summarize important clauses
                important_clauses = await self._find_important_clauses(document_id)
                
                clause_summaries = []
                for clause in important_clauses:
                    summary = await self._summarize_clause(clause['text'])
                    clause_summaries.append({
                        'clause_type': clause.get('type', 'general'),
                        'original_text': clause['text'][:200] + "...",
                        'summary': summary,
                        'importance': clause.get('importance', 5)
                    })
                
                result = {
                    'document_id': document_id,
                    'summary_type': SummaryType.CLAUSE,
                    'clause_summaries': clause_summaries,
                    'total_clauses': len(clause_summaries)
                }
            
            return self.create_success_output(result)
            
        except Exception as e:
            return self.create_error_output(f"Failed to generate clause summary: {str(e)}")
    
    def _combine_chunks(self, chunks: List[Dict[str, Any]], max_length: int = 100000) -> str:
        """Combine chunks into a single text, respecting length limits"""
        combined_text = ""
        current_length = 0
        
        for chunk in sorted(chunks, key=lambda x: x['chunk_index']):
            chunk_text = chunk['text']
            if current_length + len(chunk_text) > max_length:
                break
            combined_text += chunk_text + "\n\n"
            current_length += len(chunk_text)
        
        return combined_text
    
    def _extract_key_points(self, summary_text: str) -> List[str]:
        """Extract key points from summary text"""
        # Look for numbered lists, bullet points, or key phrases
        key_points = []
        
        # Pattern for numbered items
        numbered_pattern = r'^\d+\.\s*(.+)$'
        # Pattern for bullet points
        bullet_pattern = r'^[-•*]\s*(.+)$'
        
        lines = summary_text.split('\n')
        for line in lines:
            line = line.strip()
            if re.match(numbered_pattern, line):
                key_points.append(re.match(numbered_pattern, line).group(1))
            elif re.match(bullet_pattern, line):
                key_points.append(re.match(bullet_pattern, line).group(1))
            elif line and len(line) > 20 and len(line) < 200:
                # Look for sentences that might be key points
                if any(keyword in line.lower() for keyword in ['important', 'key', 'must', 'shall', 'required', 'critical']):
                    key_points.append(line)
        
        return key_points[:10]  # Limit to top 10 key points
    
    def _identify_sections(self, text: str) -> List[Dict[str, Any]]:
        """Identify sections in the document"""
        sections = []
        
        # Common section patterns in legal documents
        section_patterns = [
            r'^(ARTICLE|Article|SECTION|Section)\s+([IVX\d]+)[\.\:]\s*(.+)$',
            r'^(\d+\.\d*)\s+(.+)$',
            r'^([A-Z\s]{3,})\s*$',  # All caps headings
        ]
        
        lines = text.split('\n')
        current_section = None
        current_content = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if line matches section pattern
            is_section_header = False
            for pattern in section_patterns:
                match = re.match(pattern, line)
                if match:
                    # Save previous section
                    if current_section and current_content:
                        sections.append({
                            'title': current_section,
                            'content': '\n'.join(current_content),
                            'word_count': len(' '.join(current_content).split())
                        })
                    
                    # Start new section
                    current_section = line
                    current_content = []
                    is_section_header = True
                    break
            
            if not is_section_header:
                current_content.append(line)
        
        # Add final section
        if current_section and current_content:
            sections.append({
                'title': current_section,
                'content': '\n'.join(current_content),
                'word_count': len(' '.join(current_content).split())
            })
        
        return sections if sections else [{'title': 'Full Document', 'content': text, 'word_count': len(text.split())}]
    
    def _get_section_content(self, sections: List[Dict[str, Any]], section_name: str) -> Optional[str]:
        """Get content for a specific section"""
        for section in sections:
            if section_name.lower() in section['title'].lower():
                return section['content']
        return None
    
    async def _summarize_section(self, section_title: str, section_content: str) -> Dict[str, Any]:
        """Summarize a specific section"""
        try:
            prompt = f"""
            Summarize this section from a legal document. Focus on:
            1. Main purpose of this section
            2. Key obligations or rights
            3. Important terms or conditions
            4. Any deadlines or timeframes
            5. Potential risks or concerns
            
            Section: {section_title}
            Content: {section_content[:3000]}  # Limit content length
            
            Provide a clear, concise summary in plain English.
            """
            
            summary_text = await self._generate_with_gemini(prompt)
            
            return {
                'section_title': section_title,
                'summary': summary_text,
                'word_count': len(section_content.split()),
                'key_terms': self._extract_key_terms(section_content)
            }
            
        except Exception as e:
            return {
                'section_title': section_title,
                'summary': f"Error generating summary: {str(e)}",
                'word_count': len(section_content.split()),
                'key_terms': []
            }
    
    async def _summarize_clause(self, clause_text: str) -> str:
        """Summarize a specific clause"""
        try:
            prompt = f"""
            Explain this legal clause in simple terms:
            1. What does it mean?
            2. What are the obligations?
            3. What are the consequences?
            4. Why is it important?
            
            Clause: {clause_text[:1000]}
            
            Provide a clear explanation that a non-lawyer can understand.
            """
            
            return await self._generate_with_gemini(prompt)
            
        except Exception as e:
            return f"Error summarizing clause: {str(e)}"
    
    async def _find_important_clauses(self, document_id: str) -> List[Dict[str, Any]]:
        """Find important clauses in the document using keyword search"""
        try:
            # Get full document text
            full_text = await self._get_document_text(document_id)
            
            if not full_text:
                return []
                
            important_keywords = [
                "termination", "liability", "indemnification", "confidentiality",
                "payment", "penalty", "breach", "dispute", "arbitration",
                "renewal", "amendment", "force majeure"
            ]
            
            important_clauses = []
            
            # Split text into paragraphs/sections for clause analysis
            paragraphs = [p.strip() for p in full_text.split('\n\n') if p.strip()]
            
            for i, paragraph in enumerate(paragraphs):
                paragraph_lower = paragraph.lower()
                
                for keyword in important_keywords:
                    if keyword in paragraph_lower and len(paragraph) > 100:  # Only meaningful paragraphs
                        # Calculate importance based on keyword frequency and paragraph length
                        keyword_count = paragraph_lower.count(keyword)
                        importance = min(keyword_count * 2 + (len(paragraph) // 100), 10)
                        
                        important_clauses.append({
                            'text': paragraph,
                            'type': keyword,
                            'importance': importance,
                            'chunk_id': f"para_{i}"
                        })
            
            # Remove duplicates and sort by importance
            seen_chunks = set()
            unique_clauses = []
            
            for clause in sorted(important_clauses, key=lambda x: x['importance'], reverse=True):
                if clause['chunk_id'] not in seen_chunks:
                    unique_clauses.append(clause)
                    seen_chunks.add(clause['chunk_id'])
            
            return unique_clauses[:10]  # Return top 10 important clauses
            
        except Exception as e:
            print(f"Error finding important clauses: {str(e)}")
            return []
    
    def _extract_key_terms(self, text: str) -> List[str]:
        """Extract key legal terms from text"""
        legal_terms = [
            "agreement", "contract", "party", "parties", "obligation", "right",
            "liability", "indemnification", "confidentiality", "termination",
            "breach", "remedy", "damages", "penalty", "force majeure",
            "jurisdiction", "arbitration", "mediation", "amendment"
        ]
        
        found_terms = []
        text_lower = text.lower()
        
        for term in legal_terms:
            if term in text_lower:
                found_terms.append(term)
        
        return found_terms