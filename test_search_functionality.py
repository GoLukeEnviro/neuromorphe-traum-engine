#!/usr/bin/env python3
"""
Test script for search functionality
"""

import requests
import json

def test_search_health():
    """Test the search health endpoint"""
    url = 'http://localhost:8003/api/v1/search/health'
    
    try:
        response = requests.get(url, timeout=5)
        
        if response.status_code == 200:
            health_data = response.json()
            print('✅ Search Health Check: OK')
            print(f'   Status: {health_data.get("status", "Unknown")}')
            return True
        else:
            print(f'❌ Search Health Check failed: {response.status_code}')
            return False
            
    except requests.exceptions.ConnectionError:
        print('❌ Search Health Check: Connection refused (endpoint may not exist)')
        return False
    except Exception as e:
        print(f'❌ Search Health Check error: {e}')
        return False

def test_search_query():
    """Test the search query endpoint"""
    url = 'http://localhost:8003/api/v1/search/'
    
    search_data = {
        'query': 'techno beat',
        'limit': 5
    }
    
    try:
        response = requests.post(url, json=search_data, timeout=10)
        
        if response.status_code == 200:
            results = response.json()
            result_count = len(results.get('results', []))
            print(f'✅ Search Query: Found {result_count} results')
            
            if result_count > 0:
                print('   Sample results:')
                for i, result in enumerate(results.get('results', [])[:3]):
                    print(f'     {i+1}. {result.get("filename", "Unknown")} (Score: {result.get("score", "N/A")})')
            
            return True
        else:
            print(f'❌ Search Query failed: {response.status_code}')
            print(f'   Response: {response.text}')
            return False
            
    except requests.exceptions.ConnectionError:
        print('❌ Search Query: Connection refused (endpoint may not exist)')
        return False
    except Exception as e:
        print(f'❌ Search Query error: {e}')
        return False

def test_api_docs():
    """Test if API documentation includes search endpoints"""
    url = 'http://localhost:8003/openapi.json'
    
    try:
        response = requests.get(url, timeout=5)
        
        if response.status_code == 200:
            api_spec = response.json()
            paths = api_spec.get('paths', {})
            
            search_endpoints = [path for path in paths.keys() if 'search' in path]
            
            if search_endpoints:
                print(f'✅ Search endpoints found in API: {len(search_endpoints)}')
                for endpoint in search_endpoints:
                    print(f'   - {endpoint}')
                return True
            else:
                print('⚠️  No search endpoints found in API documentation')
                return False
        else:
            print(f'❌ API documentation not accessible: {response.status_code}')
            return False
            
    except Exception as e:
        print(f'❌ API documentation error: {e}')
        return False

if __name__ == '__main__':
    print('🔍 Testing Search Functionality')
    print('=' * 50)
    
    # Test API documentation first
    docs_success = test_api_docs()
    
    # Test search health
    health_success = test_search_health()
    
    # Test search query
    query_success = test_search_query()
    
    print('\n' + '=' * 50)
    
    if docs_success and health_success and query_success:
        print('🎯 All search tests passed! Search system is working.')
    elif docs_success:
        print('⚠️  Search endpoints exist but may not be fully functional.')
    else:
        print('❌ Search system appears to be missing or not working.')
        print('   This is expected if search functionality is not yet implemented.')