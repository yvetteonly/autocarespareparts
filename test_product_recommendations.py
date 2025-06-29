#!/usr/bin/env python3
"""
Test script for product-specific recommendations endpoint
"""

import requests
import json

def test_product_recommendations():
    """Test product-specific recommendations endpoint"""
    print("Testing Product-Specific Recommendations")
    print("=" * 50)
    
    # Test with a few different product IDs
    product_ids = [1, 2, 3, 6, 12]
    
    for product_id in product_ids:
        print(f"\nTesting recommendations for product ID: {product_id}")
        print("-" * 40)
        
        try:
            response = requests.get(f'http://127.0.0.1:5000/recommendations/product/{product_id}')
            print(f"Status Code: {response.status_code}")
            print(f"Content-Type: {response.headers.get('content-type')}")
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    if data['success']:
                        print(f"Found {len(data['recommendations'])} recommendations")
                        for i, rec in enumerate(data['recommendations'][:3], 1):  # Show first 3
                            print(f"  {i}. {rec['name']} - {rec['price']} RWF (Score: {rec.get('similarity_score', 'N/A')})")
                    else:
                        print(f"Error: {data.get('message', 'Unknown error')}")
                except json.JSONDecodeError:
                    print("Response is not valid JSON")
                    print("Response text:", response.text[:200])
            else:
                print(f"HTTP Error: {response.status_code}")
                print("Response text:", response.text[:200])
                
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")

if __name__ == "__main__":
    test_product_recommendations() 