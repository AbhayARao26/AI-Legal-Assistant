"""
Configuration management for the Legal AI system
"""
from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # Google Gemini API
    google_api_key: str
    
    # Pinecone Configuration
    pinecone_api_key: str
    pinecone_environment: str
    pinecone_index_name: str = "legal-documents"
    
    # Database Configuration
    database_url: str = "sqlite:///./legal_ai.db"
    
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = True
    
    # Upload Configuration
    max_file_size: int = 50000000  # 50MB
    upload_dir: str = "./uploads"
    
    # Session Configuration
    session_timeout: int = 3600  # 1 hour
    
    class Config:
        env_file = "../.env"
        case_sensitive = False


# Global settings instance
settings = Settings()