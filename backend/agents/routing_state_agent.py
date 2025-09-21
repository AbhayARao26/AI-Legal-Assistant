"""
Routing & State Management Agent
Central controller for routing requests between agents and managing session state
"""
import uuid
import json
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import google.generativeai as genai

from .base_agent import BaseAgent, AgentInput, AgentOutput, AgentState
from .document_retrieval_agent import DocumentRetrievalAgent
from .summarization_agent import SummarizationAgent
from .clauses_extraction_agent import ClausesExtractionAgent
from .legal_definitions_agent import LegalDefinitionsAgent
from .qa_agent import QAAgent
from config import settings


class RoutingStateAgent(BaseAgent):
    """Central agent for routing requests and managing state"""
    
    def __init__(self):
        super().__init__("RoutingStateAgent")
        
        # Initialize Gemini for intent classification
        genai.configure(api_key=settings.google_api_key)
        self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Initialize all agents
        self.agents = {
            'document_retrieval': DocumentRetrievalAgent(),
            'summarization': SummarizationAgent(),
            'clauses_extraction': ClausesExtractionAgent(),
            'legal_definitions': LegalDefinitionsAgent(),
            'qa': QAAgent()
        }
        
        # Session storage (in production, this would be a database)
        self.sessions = {}
        
        # Intent classification patterns
        self.intent_patterns = self._load_intent_patterns()
    
    async def run(self, input_data: AgentInput, state: AgentState) -> AgentOutput:
        """Main routing method"""
        try:
            action = input_data.parameters.get('action', 'route')
            
            if action == 'route':
                return await self._route_request(input_data, state)
            elif action == 'get_session':
                return await self._get_session_state(input_data, state)
            elif action == 'update_session':
                return await self._update_session_state(input_data, state)
            elif action == 'create_session':
                return await self._create_session(input_data, state)
            else:
                return self.create_error_output(f"Unknown action: {action}")
                
        except Exception as e:
            return self.create_error_output(f"Error in RoutingStateAgent: {str(e)}")
    
    async def _route_request(self, input_data: AgentInput, state: AgentState) -> AgentOutput:
        """Route request to appropriate agent based on intent classification"""
        try:
            query = input_data.query
            
            # Classify intent
            intent_analysis = await self._classify_intent(query, state)
            target_agent = intent_analysis['target_agent']
            confidence = intent_analysis['confidence']
            
            # Update state with routing information
            state.context['last_intent'] = intent_analysis
            state.context['routing_timestamp'] = datetime.now().isoformat()
            
            # Route to appropriate agent
            if target_agent in self.agents:
                # Prepare parameters for the target agent
                agent_input = self._prepare_agent_input(input_data, intent_analysis)
                
                # Execute agent
                agent_result = await self.agents[target_agent].run(agent_input, state)
                
                # Log the interaction
                self.log_interaction(state, input_data, agent_result)
                
                # Generate next suggested actions
                next_actions = await self._suggest_next_actions(target_agent, agent_result, state)
                
                result = {
                    'agent_used': target_agent,
                    'intent_detected': intent_analysis['intent'],
                    'confidence': confidence,
                    'result': agent_result.result,
                    'success': agent_result.success,
                    'error': agent_result.error_message,
                    'citations': agent_result.citations,
                    'next_suggested_actions': next_actions,
                    'processing_time': intent_analysis.get('processing_time', 0)
                }
                
                return self.create_success_output(result, citations=agent_result.citations)
            else:
                return self.create_error_output(f"Unknown target agent: {target_agent}")
                
        except Exception as e:
            return self.create_error_output(f"Failed to route request: {str(e)}")
    
    async def _classify_intent(self, query: str, state: AgentState) -> Dict[str, Any]:
        """Classify user intent to determine which agent to use"""
        start_time = datetime.now()
        
        try:
            # First, try pattern-based classification (fast)
            pattern_result = self._pattern_based_classification(query)
            
            if pattern_result['confidence'] > 0.8:
                processing_time = (datetime.now() - start_time).total_seconds()
                pattern_result['processing_time'] = processing_time
                return pattern_result
            
            # If pattern-based is not confident enough, use AI classification
            ai_result = await self._ai_based_classification(query, state)
            
            # Combine results for final decision
            final_result = self._combine_classification_results(pattern_result, ai_result)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            final_result['processing_time'] = processing_time
            
            return final_result
            
        except Exception as e:
            # Fallback to Q&A if classification fails
            return {
                'intent': 'question',
                'target_agent': 'qa',
                'confidence': 0.3,
                'method': 'fallback',
                'error': str(e)
            }
    
    def _pattern_based_classification(self, query: str) -> Dict[str, Any]:
        """Fast pattern-based intent classification"""
        query_lower = query.lower()
        
        # Check each intent pattern
        best_match = {'intent': 'question', 'target_agent': 'qa', 'confidence': 0.0}
        
        for intent, patterns in self.intent_patterns.items():
            score = 0
            for pattern in patterns['keywords']:
                if pattern.lower() in query_lower:
                    score += patterns['weights'].get(pattern, 1)
            
            # Normalize score
            confidence = min(score / max(len(patterns['keywords']), 1), 1.0)
            
            if confidence > best_match['confidence']:
                best_match = {
                    'intent': intent,
                    'target_agent': patterns['agent'],
                    'confidence': confidence,
                    'method': 'pattern_based'
                }
        
        return best_match
    
    async def _ai_based_classification(self, query: str, state: AgentState) -> Dict[str, Any]:
        """AI-based intent classification using Gemini"""
        try:
            # Get context from conversation history
            context = ""
            if state.conversation_history:
                recent_interactions = state.conversation_history[-3:]
                for interaction in recent_interactions:
                    context += f"Previous: {interaction.get('input', {}).get('query', '')[:100]}...\n"
            
            prompt = f"""
            Classify the user's intent based on their query. Choose the most appropriate category:
            
            1. SUMMARIZE - User wants a summary of the document (overall, section-wise, or clause-level)
               Keywords: summarize, summary, overview, main points, key points, gist, brief
            
            2. EXTRACT_CLAUSES - User wants to extract specific clauses like obligations, rights, penalties
               Keywords: extract, clauses, obligations, rights, penalties, terms, conditions, find clauses
            
            3. DEFINE_TERMS - User wants definitions of legal terms or jargon
               Keywords: define, definition, what does, what is, explain, meaning, legal term
            
            4. QUESTION - User has a specific question about the document content
               Keywords: question words (who, what, when, where, why, how), specific queries about content
            
            5. UPLOAD - User wants to upload or process a document
               Keywords: upload, process, analyze document, new document
            
            Recent conversation context:
            {context}
            
            User query: "{query}"
            
            Respond with only the category name (SUMMARIZE, EXTRACT_CLAUSES, DEFINE_TERMS, QUESTION, or UPLOAD).
            """
            
            response = await self.gemini_model.generate_content_async(prompt)
            classification = response.text.strip().upper()
            
            # Map to agents
            agent_mapping = {
                'SUMMARIZE': 'summarization',
                'EXTRACT_CLAUSES': 'clauses_extraction',
                'DEFINE_TERMS': 'legal_definitions',
                'QUESTION': 'qa',
                'UPLOAD': 'document_retrieval'
            }
            
            if classification in agent_mapping:
                return {
                    'intent': classification.lower(),
                    'target_agent': agent_mapping[classification],
                    'confidence': 0.9,
                    'method': 'ai_based'
                }
            else:
                # Fallback
                return {
                    'intent': 'question',
                    'target_agent': 'qa',
                    'confidence': 0.5,
                    'method': 'ai_fallback'
                }
                
        except Exception as e:
            return {
                'intent': 'question',
                'target_agent': 'qa',
                'confidence': 0.3,
                'method': 'ai_error',
                'error': str(e)
            }
    
    def _combine_classification_results(self, pattern_result: Dict, ai_result: Dict) -> Dict[str, Any]:
        """Combine pattern-based and AI-based classification results"""
        # If both agree and have high confidence, use that
        if (pattern_result['target_agent'] == ai_result['target_agent'] and 
            pattern_result['confidence'] > 0.6 and ai_result['confidence'] > 0.7):
            return {
                'intent': ai_result['intent'],
                'target_agent': ai_result['target_agent'],
                'confidence': min(pattern_result['confidence'] + 0.1, 1.0),
                'method': 'combined_agreement'
            }
        
        # If AI has higher confidence, use AI result
        if ai_result['confidence'] > pattern_result['confidence']:
            return ai_result
        
        # Otherwise use pattern result
        return pattern_result
    
    def _prepare_agent_input(self, original_input: AgentInput, intent_analysis: Dict) -> AgentInput:
        """Prepare input for the target agent based on intent analysis"""
        target_agent = intent_analysis['target_agent']
        intent = intent_analysis['intent']
        
        # Create new parameters based on intent
        new_parameters = original_input.parameters.copy()
        
        if target_agent == 'summarization':
            # Determine summary type from query
            query_lower = original_input.query.lower()
            if 'section' in query_lower:
                new_parameters['summary_type'] = 'section'
            elif 'clause' in query_lower:
                new_parameters['summary_type'] = 'clause'
            else:
                new_parameters['summary_type'] = 'overall'
                
        elif target_agent == 'clauses_extraction':
            # Determine clause types to extract
            query_lower = original_input.query.lower()
            clause_types = []
            
            clause_keywords = {
                'obligation': ['obligation', 'must', 'shall', 'required'],
                'right': ['right', 'can', 'may', 'entitled'],
                'penalty': ['penalty', 'fine', 'damages'],
                'timeline': ['deadline', 'date', 'timeline', 'when'],
                'termination': ['terminate', 'end', 'cancel'],
                'payment': ['payment', 'pay', 'fee', 'cost']
            }
            
            for clause_type, keywords in clause_keywords.items():
                if any(keyword in query_lower for keyword in keywords):
                    clause_types.append(clause_type)
            
            if clause_types:
                new_parameters['clause_types'] = clause_types
                
        elif target_agent == 'legal_definitions':
            # Extract terms to define from query
            query_lower = original_input.query.lower()
            
            # Simple term extraction (in production, this could be more sophisticated)
            import re
            terms = []
            
            # Look for quoted terms
            quoted_terms = re.findall(r'"([^"]+)"', original_input.query)
            terms.extend(quoted_terms)
            
            # Look for "define X" or "what is X" patterns
            define_patterns = [
                r'define\s+([a-zA-Z\s]+)',
                r'what\s+is\s+([a-zA-Z\s]+)',
                r'meaning\s+of\s+([a-zA-Z\s]+)'
            ]
            
            for pattern in define_patterns:
                matches = re.findall(pattern, query_lower)
                terms.extend([match.strip() for match in matches])
            
            if terms:
                new_parameters['terms'] = list(set(terms))  # Remove duplicates
                
        elif target_agent == 'qa':
            # Add conversation history for Q&A
            new_parameters['conversation_history'] = getattr(original_input, 'conversation_history', [])
        
        return AgentInput(
            query=original_input.query,
            document_id=original_input.document_id,
            parameters=new_parameters
        )
    
    async def _suggest_next_actions(self, used_agent: str, agent_result: AgentOutput, state: AgentState) -> List[Dict[str, Any]]:
        """Suggest next actions based on the current result"""
        suggestions = []
        
        if not agent_result.success:
            return suggestions
        
        # Agent-specific suggestions
        if used_agent == 'document_retrieval':
            suggestions.extend([
                {
                    'action': 'summarize',
                    'description': 'Get an overview of the document',
                    'agent': 'summarization',
                    'priority': 'high'
                },
                {
                    'action': 'extract_clauses',
                    'description': 'Extract key clauses and terms',
                    'agent': 'clauses_extraction',
                    'priority': 'medium'
                }
            ])
            
        elif used_agent == 'summarization':
            suggestions.extend([
                {
                    'action': 'extract_clauses',
                    'description': 'Extract specific clauses for detailed analysis',
                    'agent': 'clauses_extraction',
                    'priority': 'high'
                },
                {
                    'action': 'ask_question',
                    'description': 'Ask specific questions about the document',
                    'agent': 'qa',
                    'priority': 'medium'
                }
            ])
            
        elif used_agent == 'clauses_extraction':
            suggestions.extend([
                {
                    'action': 'define_terms',
                    'description': 'Get definitions of legal terms found',
                    'agent': 'legal_definitions',
                    'priority': 'medium'
                },
                {
                    'action': 'ask_question',
                    'description': 'Ask questions about specific clauses',
                    'agent': 'qa',
                    'priority': 'high'
                }
            ])
            
        elif used_agent == 'legal_definitions':
            suggestions.extend([
                {
                    'action': 'ask_question',
                    'description': 'Ask questions using your new understanding',
                    'agent': 'qa',
                    'priority': 'high'
                }
            ])
            
        elif used_agent == 'qa':
            # Suggest related questions if available
            if 'related_questions' in agent_result.result:
                for related_q in agent_result.result['related_questions'][:2]:
                    suggestions.append({
                        'action': 'ask_question',
                        'description': f'Ask: "{related_q}"',
                        'agent': 'qa',
                        'priority': 'medium',
                        'query': related_q
                    })
        
        return suggestions[:4]  # Limit to 4 suggestions
    
    async def _create_session(self, input_data: AgentInput, state: AgentState) -> AgentOutput:
        """Create a new session"""
        try:
            session_id = str(uuid.uuid4())
            user_id = input_data.parameters.get('user_id')
            
            session_data = {
                'session_id': session_id,
                'user_id': user_id,
                'created_at': datetime.now(),
                'updated_at': datetime.now(),
                'is_active': True,
                'context': {},
                'uploaded_documents': [],
                'conversation_count': 0,
                'last_activity': datetime.now()
            }
            
            # Store session (in production, this would be in database)
            self.sessions[session_id] = session_data
            
            # Update state
            state.session_id = session_id
            state.user_id = user_id
            
            result = {
                'session_id': session_id,
                'created_at': session_data['created_at'].isoformat(),
                'user_id': user_id
            }
            
            return self.create_success_output(result)
            
        except Exception as e:
            return self.create_error_output(f"Failed to create session: {str(e)}")
    
    async def _get_session_state(self, input_data: AgentInput, state: AgentState) -> AgentOutput:
        """Get current session state"""
        try:
            session_id = state.session_id or input_data.parameters.get('session_id')
            
            if not session_id or session_id not in self.sessions:
                return self.create_error_output("Session not found")
            
            session_data = self.sessions[session_id]
            
            # Check if session is expired (1 hour timeout)
            if (datetime.now() - session_data['last_activity']).total_seconds() > settings.session_timeout:
                session_data['is_active'] = False
            
            result = {
                'session_id': session_id,
                'user_id': session_data['user_id'],
                'created_at': session_data['created_at'].isoformat(),
                'updated_at': session_data['updated_at'].isoformat(),
                'is_active': session_data['is_active'],
                'context': session_data['context'],
                'uploaded_documents': session_data['uploaded_documents'],
                'conversation_count': session_data['conversation_count']
            }
            
            return self.create_success_output(result)
            
        except Exception as e:
            return self.create_error_output(f"Failed to get session state: {str(e)}")
    
    async def _update_session_state(self, input_data: AgentInput, state: AgentState) -> AgentOutput:
        """Update session state"""
        try:
            session_id = state.session_id or input_data.parameters.get('session_id')
            updates = input_data.parameters.get('updates', {})
            
            if not session_id or session_id not in self.sessions:
                return self.create_error_output("Session not found")
            
            session_data = self.sessions[session_id]
            
            # Apply updates
            for key, value in updates.items():
                if key in ['context', 'uploaded_documents']:
                    session_data[key].update(value) if key == 'context' else session_data[key].extend(value)
                elif key in ['user_id', 'is_active']:
                    session_data[key] = value
            
            session_data['updated_at'] = datetime.now()
            session_data['last_activity'] = datetime.now()
            
            result = {
                'session_id': session_id,
                'updated_at': session_data['updated_at'].isoformat(),
                'applied_updates': list(updates.keys())
            }
            
            return self.create_success_output(result)
            
        except Exception as e:
            return self.create_error_output(f"Failed to update session state: {str(e)}")
    
    def _load_intent_patterns(self) -> Dict[str, Dict]:
        """Load intent classification patterns"""
        return {
            'summarize': {
                'agent': 'summarization',
                'keywords': ['summarize', 'summary', 'overview', 'main points', 'key points', 'brief', 'gist'],
                'weights': {'summarize': 3, 'summary': 3, 'overview': 2, 'main points': 2}
            },
            'extract_clauses': {
                'agent': 'clauses_extraction',
                'keywords': ['extract', 'clauses', 'terms', 'conditions', 'obligations', 'rights', 'penalties', 'find clauses'],
                'weights': {'extract': 2, 'clauses': 3, 'obligations': 3, 'rights': 2}
            },
            'define_terms': {
                'agent': 'legal_definitions',
                'keywords': ['define', 'definition', 'what does', 'what is', 'explain', 'meaning', 'legal term'],
                'weights': {'define': 3, 'definition': 3, 'what does': 2, 'what is': 2}
            },
            'question': {
                'agent': 'qa',
                'keywords': ['who', 'what', 'when', 'where', 'why', 'how', 'can i', 'should i', 'happens if'],
                'weights': {'what': 1, 'how': 1, 'when': 1, 'happens if': 2}
            },
            'upload': {
                'agent': 'document_retrieval',
                'keywords': ['upload', 'process', 'analyze document', 'new document', 'load document'],
                'weights': {'upload': 3, 'process': 2, 'analyze document': 3}
            }
        }