# Legal AI System - Test Suite
# Run: python test_system.py

import requests
import json
import time
import os
from pathlib import Path

BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test basic server health"""
    try:
        response = requests.get(f"{BASE_URL}/health")
        assert response.status_code == 200
        print("✅ Health check passed")
        return True
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_upload_document():
    """Test document upload"""
    try:
        # Create a test document
        test_content = """
        EMPLOYMENT AGREEMENT
        
        This Employment Agreement is entered into between Company XYZ and John Doe.
        
        TERM: The term of employment shall be for 2 years.
        
        COMPENSATION: Employee shall receive $75,000 annually.
        
        CONFIDENTIALITY: Employee agrees to maintain confidentiality of company information.
        
        TERMINATION: Either party may terminate with 30 days notice.
        """
        
        test_file_path = Path("test_document.txt")
        test_file_path.write_text(test_content)
        
        with open(test_file_path, 'rb') as f:
            files = {'file': ('test_document.txt', f, 'text/plain')}
            response = requests.post(f"{BASE_URL}/upload_document", files=files)
        
        # Cleanup
        test_file_path.unlink()
        
        assert response.status_code == 200
        data = response.json()
        assert 'document_id' in data
        print("✅ Document upload passed")
        return data['document_id']
    
    except Exception as e:
        print(f"❌ Document upload failed: {e}")
        return None

def test_summarization(document_id):
    """Test document summarization"""
    try:
        response = requests.post(f"{BASE_URL}/summarize", json={
            "document_id": document_id,
            "summary_type": "comprehensive"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert 'summary' in data
        print("✅ Summarization passed")
        return True
    
    except Exception as e:
        print(f"❌ Summarization failed: {e}")
        return False

def test_clause_extraction(document_id):
    """Test clause extraction"""
    try:
        response = requests.post(f"{BASE_URL}/extract_clauses", json={
            "document_id": document_id,
            "clause_types": ["termination", "compensation", "confidentiality"]
        })
        
        assert response.status_code == 200
        data = response.json()
        assert 'clauses' in data
        print("✅ Clause extraction passed")
        return True
    
    except Exception as e:
        print(f"❌ Clause extraction failed: {e}")
        return False

def test_legal_definitions(document_id):
    """Test legal definitions"""
    try:
        response = requests.post(f"{BASE_URL}/define_terms", json={
            "document_id": document_id,
            "terms": ["confidentiality", "termination", "employment"]
        })
        
        assert response.status_code == 200
        data = response.json()
        assert 'definitions' in data
        print("✅ Legal definitions passed")
        return True
    
    except Exception as e:
        print(f"❌ Legal definitions failed: {e}")
        return False

def test_qa_system(document_id):
    """Test Q&A system"""
    try:
        response = requests.post(f"{BASE_URL}/ask", json={
            "document_id": document_id,
            "question": "What is the salary mentioned in this contract?"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert 'answer' in data
        print("✅ Q&A system passed")
        return True
    
    except Exception as e:
        print(f"❌ Q&A system failed: {e}")
        return False

def test_session_management():
    """Test session management"""
    try:
        # Create session
        response = requests.post(f"{BASE_URL}/session", json={
            "user_id": "test_user"
        })
        assert response.status_code == 200
        session_data = response.json()
        session_id = session_data['session_id']
        
        # Get session
        response = requests.get(f"{BASE_URL}/session/{session_id}")
        assert response.status_code == 200
        
        print("✅ Session management passed")
        return session_id
    
    except Exception as e:
        print(f"❌ Session management failed: {e}")
        return None

def test_routing_system(document_id):
    """Test routing system"""
    try:
        response = requests.post(f"{BASE_URL}/route", json={
            "user_input": "Can you summarize this document?",
            "document_id": document_id
        })
        
        assert response.status_code == 200
        data = response.json()
        assert 'result' in data
        print("✅ Routing system passed")
        return True
    
    except Exception as e:
        print(f"❌ Routing system failed: {e}")
        return False

def run_all_tests():
    """Run complete test suite"""
    print("🧪 Running Legal AI System Test Suite")
    print("=" * 50)
    
    # Check if server is running
    if not test_health_check():
        print("💡 Make sure the backend server is running: python start.py")
        return
    
    time.sleep(1)
    
    # Test document upload
    document_id = test_upload_document()
    if not document_id:
        print("❌ Cannot continue without document upload")
        return
    
    time.sleep(2)
    
    # Test session management
    session_id = test_session_management()
    
    time.sleep(1)
    
    # Test all agents
    tests = [
        (test_summarization, document_id),
        (test_clause_extraction, document_id),
        (test_legal_definitions, document_id),
        (test_qa_system, document_id),
        (test_routing_system, document_id)
    ]
    
    passed = 0
    total = len(tests) + 3  # +3 for health, upload, session
    
    for test_func, *args in tests:
        time.sleep(1)
        if test_func(*args):
            passed += 1
    
    # Count earlier passed tests
    if document_id:
        passed += 2  # health + upload
    if session_id:
        passed += 1  # session
    
    print("\n" + "=" * 50)
    print(f"🎯 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the logs above.")
    
    print("=" * 50)

if __name__ == "__main__":
    run_all_tests()