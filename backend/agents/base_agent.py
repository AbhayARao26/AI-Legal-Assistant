"""
Base Agent class that all agents will inherit from
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from pydantic import BaseModel
import uuid
from datetime import datetime


class AgentState(BaseModel):
    """Represents the state passed between agents"""
    session_id: str
    user_id: Optional[str] = None
    document_id: Optional[str] = None
    conversation_history: list = []
    context: Dict[str, Any] = {}
    timestamp: datetime = datetime.now()


class AgentInput(BaseModel):
    """Input structure for agent methods"""
    query: str
    document_id: Optional[str] = None
    parameters: Dict[str, Any] = {}


class AgentOutput(BaseModel):
    """Output structure for agent methods"""
    success: bool
    result: Dict[str, Any]
    error_message: Optional[str] = None
    citations: list = []
    next_agent: Optional[str] = None


class BaseAgent(ABC):
    """Base class for all agents in the system"""
    
    def __init__(self, name: str):
        self.name = name
        self.agent_id = str(uuid.uuid4())
    
    @abstractmethod
    async def run(self, input_data: AgentInput, state: AgentState) -> AgentOutput:
        """
        Main method that each agent must implement
        
        Args:
            input_data: The input data for the agent
            state: The current state of the session
            
        Returns:
            AgentOutput: The result of the agent's processing
        """
        pass
    
    def update_state(self, state: AgentState, updates: Dict[str, Any]) -> AgentState:
        """Update the state with new information"""
        for key, value in updates.items():
            if hasattr(state, key):
                setattr(state, key, value)
            else:
                state.context[key] = value
        state.timestamp = datetime.now()
        return state
    
    def log_interaction(self, state: AgentState, input_data: AgentInput, output: AgentOutput):
        """Log the interaction to conversation history"""
        interaction = {
            "agent": self.name,
            "timestamp": datetime.now().isoformat(),
            "input": input_data.dict(),
            "output": output.dict(),
            "agent_id": self.agent_id
        }
        state.conversation_history.append(interaction)
    
    async def validate_input(self, input_data: AgentInput) -> bool:
        """Validate the input data for the agent"""
        return bool(input_data.query.strip()) if input_data.query else False
    
    def create_error_output(self, error_message: str) -> AgentOutput:
        """Create a standardized error output"""
        return AgentOutput(
            success=False,
            result={},
            error_message=error_message
        )
    
    def create_success_output(self, result: Dict[str, Any], citations: list = None, next_agent: str = None) -> AgentOutput:
        """Create a standardized success output"""
        return AgentOutput(
            success=True,
            result=result,
            citations=citations or [],
            next_agent=next_agent
        )