#!/bin/bash

# Legal AI System Setup Script
echo "🏗️  Setting up Legal AI Document Analysis System..."

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv legal_ai_env
source legal_ai_env/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p uploads
mkdir -p logs
mkdir -p data

# Setup environment file
echo "⚙️  Setting up environment configuration..."
if [ ! -f .env ]; then
    cp .env.example .env
    echo "📝 Created .env file from template. Please update with your API keys."
else
    echo "✅ .env file already exists."
fi

# Initialize database
echo "🗄️  Initializing database..."
python -c "
from backend.models.database import create_tables
create_tables()
print('Database tables created successfully!')
"

echo "✅ Setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Update .env file with your API keys:"
echo "   - GOOGLE_API_KEY (for Gemini)"
echo "   - PINECONE_API_KEY (for vector storage)"
echo "   - PINECONE_ENVIRONMENT (e.g., 'us-west4-gcp')"
echo ""
echo "2. Start the backend server:"
echo "   cd backend && python main.py"
echo ""
echo "3. Start the frontend (in another terminal):"
echo "   streamlit run frontend/app.py"
echo ""
echo "4. Open your browser to: http://localhost:8501"