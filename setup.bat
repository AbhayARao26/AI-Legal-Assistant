@echo off
REM Legal AI System Setup Script for Windows

echo 🏗️  Setting up Legal AI Document Analysis System...

REM Create virtual environment
echo 📦 Creating virtual environment...
python -m venv legal_ai_env
call legal_ai_env\Scripts\activate.bat

REM Install dependencies
echo 📥 Installing dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt

REM Create necessary directories
echo 📁 Creating directories...
if not exist "uploads" mkdir uploads
if not exist "logs" mkdir logs
if not exist "data" mkdir data

REM Setup environment file
echo ⚙️  Setting up environment configuration...
if not exist ".env" (
    copy .env.example .env
    echo 📝 Created .env file from template. Please update with your API keys.
) else (
    echo ✅ .env file already exists.
)

REM Initialize database
echo 🗄️  Initializing database...
python -c "from backend.models.database import create_tables; create_tables(); print('Database tables created successfully!')"

echo ✅ Setup complete!
echo.
echo 📋 Next steps:
echo 1. Update .env file with your API keys:
echo    - GOOGLE_API_KEY (for Gemini)
echo    - PINECONE_API_KEY (for vector storage)
echo    - PINECONE_ENVIRONMENT (e.g., 'us-west4-gcp')
echo.
echo 2. Start the backend server:
echo    cd backend && python main.py
echo.
echo 3. Start the frontend (in another terminal):
echo    streamlit run frontend/app.py
echo.
echo 4. Open your browser to: http://localhost:8501

pause