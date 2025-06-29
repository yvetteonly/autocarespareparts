#!/usr/bin/env python3
"""
Test script for recommendations endpoint
"""

import requests
import json

def test_recommendations_without_auth():
    """Test recommendations endpoint without authentication"""
    print("Testing recommendations without authentication...")
    response = requests.get('http://127.0.0.1:5000/recommendations')
    print(f"Status Code: {response.status_code}")
    print(f"Content-Type: {response.headers.get('content-type')}")
    print(f"Response length: {len(response.text)}")
    print("First 200 chars of response:")
    print(response.text[:200])
    print("-" * 50)

def test_recommendations_with_auth():
    """Test recommendations endpoint with authentication"""
    print("Testing recommendations with authentication...")
    
    # First, try to login
    login_data = {
        'username': 'admin',
        'password': 'admin123'
    }
    
    session = requests.Session()
    
    # Get the login page first to get any CSRF tokens if needed
    login_response = session.get('http://127.0.0.1:5000/login')
    print(f"Login page status: {login_response.status_code}")
    
    # Try to login
    login_response = session.post('http://127.0.0.1:5000/login', data=login_data, allow_redirects=False)
    print(f"Login response status: {login_response.status_code}")
    print(f"Login response headers: {dict(login_response.headers)}")
    
    if login_response.status_code == 302:
        print("Login successful, redirecting...")
        
        # Now try to access recommendations
        rec_response = session.get('http://127.0.0.1:5000/recommendations')
        print(f"Recommendations status: {rec_response.status_code}")
        print(f"Recommendations content-type: {rec_response.headers.get('content-type')}")
        
        if rec_response.headers.get('content-type', '').startswith('application/json'):
            try:
                data = rec_response.json()
                print("Recommendations response:")
                print(json.dumps(data, indent=2))
            except json.JSONDecodeError:
                print("Failed to parse JSON response")
                print("Response text:", rec_response.text[:500])
        else:
            print("Response is not JSON:")
            print(rec_response.text[:500])
    else:
        print("Login failed")
        print("Response text:", login_response.text[:500])

if __name__ == "__main__":
    print("Testing Recommendations Endpoint")
    print("=" * 50)
    
    test_recommendations_without_auth()
    test_recommendations_with_auth() 