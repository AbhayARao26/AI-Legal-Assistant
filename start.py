# Legal AI System - Quick Start Script
# Run this after setting up your environment

import os
import sys
import subprocess
import time

def check_python_version():
    """Check if Python version is adequate"""
    if sys.version_info < (3, 10):
        print("❌ Python 3.10+ is required")
        return False
    print("✅ Python version OK")
    return True

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'fastapi', 'uvicorn', 'streamlit', 'requests', 
        'pydantic', 'sqlalchemy', 'pinecone-client'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package}")
    
    if missing_packages:
        print(f"\n📦 Install missing packages: pip install {' '.join(missing_packages)}")
        return False
    return True

def check_env_file():
    """Check if .env file exists and has required keys"""
    if not os.path.exists('.env'):
        print("❌ .env file not found")
        print("📝 Copy .env.example to .env and add your API keys")
        return False
    
    required_keys = ['GOOGLE_API_KEY', 'PINECONE_API_KEY', 'PINECONE_ENVIRONMENT']
    
    with open('.env', 'r') as f:
        env_content = f.read()
    
    missing_keys = []
    for key in required_keys:
        if f"{key}=" not in env_content or f"{key}=your_" in env_content:
            missing_keys.append(key)
    
    if missing_keys:
        print(f"❌ Missing or incomplete API keys: {', '.join(missing_keys)}")
        return False
    
    print("✅ Environment configuration OK")
    return True

def start_backend():
    """Start the FastAPI backend"""
    print("🚀 Starting backend server...")
    backend_process = subprocess.Popen([
        sys.executable, '-m', 'uvicorn', 'backend.main:app',
        '--host', '0.0.0.0', '--port', '8000', '--reload'
    ])
    return backend_process

def start_frontend():
    """Start the Streamlit frontend"""
    print("🚀 Starting frontend...")
    frontend_process = subprocess.Popen([
        'streamlit', 'run', 'frontend/app.py', '--server.port', '8501'
    ])
    return frontend_process

def main():
    """Main startup script"""
    print("🏗️  Legal AI System - Quick Start")
    print("=" * 50)
    
    # Checks
    if not check_python_version():
        return
    
    if not check_dependencies():
        return
    
    if not check_env_file():
        return
    
    # Create directories
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    print("\n🎯 All checks passed! Starting services...")
    
    try:
        # Start backend
        backend = start_backend()
        time.sleep(3)  # Give backend time to start
        
        # Start frontend
        frontend = start_frontend()
        
        print("\n" + "=" * 50)
        print("✅ System is running!")
        print("🌐 Frontend: http://localhost:8501")
        print("🔌 Backend API: http://localhost:8000")
        print("📚 API Docs: http://localhost:8000/docs")
        print("\n💡 Press Ctrl+C to stop both services")
        print("=" * 50)
        
        # Wait for processes
        try:
            backend.wait()
        except KeyboardInterrupt:
            print("\n🛑 Stopping services...")
            backend.terminate()
            frontend.terminate()
            print("✅ Services stopped")
    
    except Exception as e:
        print(f"❌ Error starting services: {e}")

if __name__ == "__main__":
    main()