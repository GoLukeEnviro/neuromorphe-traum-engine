import requests
import streamlit as st
from typing import List, Dict, Any, Optional
import json

def get_backend_url() -> str:
    """Get the backend URL from session state"""
    return st.session_state.get('backend_url', 'http://localhost:8000')

def check_api_health() -> bool:
    """Check if the backend API is running"""
    try:
        backend_url = get_backend_url()
        response = requests.get(f"{backend_url}/system/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def get_api_info() -> Dict[str, Any]:
    """Get API information and health status"""
    try:
        backend_url = get_backend_url()
        response = requests.get(f"{backend_url}/system/health", timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            return {'status': 'error', 'message': f'HTTP {response.status_code}'}
    except Exception as e:
        return {'status': 'error', 'message': str(e)}

def search_stems(
    query: str, 
    top_k: int = 5, 
    category: Optional[str] = None,
    min_similarity: float = 0.0,
    bpm_range: Optional[tuple] = None
) -> List[Dict[str, Any]]:
    """Search stems using the backend API"""
    try:
        backend_url = get_backend_url()
        params = {"prompt": query, "top_k": top_k}
        
        if category:
            params["category"] = category
        if min_similarity > 0:
            params["min_similarity"] = min_similarity
        if bmp_range:
            params["min_bpm"] = bpm_range[0]
            params["max_bpm"] = bpm_range[1]
        
        response = requests.get(f"{backend_url}/api/v1/stems/search/", params=params)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return []
    except Exception as e:
        st.error(f"Verbindungsfehler: {str(e)}")
        return []

def get_all_stems(skip: int = 0, limit: int = 20) -> List[Dict[str, Any]]:
    """Get all stems from the backend API"""
    try:
        backend_url = get_backend_url()
        params = {"skip": skip, "limit": limit}
        response = requests.get(f"{backend_url}/api/v1/stems/", params=params)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return []
    except Exception as e:
        st.error(f"Verbindungsfehler: {str(e)}")
        return []

def get_stem_by_id(stem_id: str) -> Optional[Dict[str, Any]]:
    """Get a specific stem by ID"""
    try:
        backend_url = get_backend_url()
        response = requests.get(f"{backend_url}/api/v1/stems/{stem_id}")
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except Exception as e:
        st.error(f"Verbindungsfehler: {str(e)}")
        return None

def upload_audio_file(file_data: bytes, filename: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
    """Upload an audio file to the backend"""
    try:
        backend_url = get_backend_url()
        files = {'file': (filename, file_data)}
        data = {'metadata': json.dumps(metadata or {})}
        
        response = requests.post(f"{backend_url}/api/v1/stems/upload/", files=files, data=data)
        if response.status_code == 200:
            return response.json()
        else:
            return {'error': f'HTTP {response.status_code}: {response.text}'}
    except Exception as e:
        return {'error': str(e)}

def delete_stem(stem_id: str) -> bool:
    """Delete a stem from the backend"""
    try:
        backend_url = get_backend_url()
        response = requests.delete(f"{backend_url}/api/v1/stems/{stem_id}")
        return response.status_code == 200
    except Exception as e:
        st.error(f"Verbindungsfehler: {str(e)}")
        return False

def get_categories() -> List[str]:
    """Get all available categories"""
    try:
        backend_url = get_backend_url()
        response = requests.get(f"{backend_url}/api/v1/stems/categories/")
        if response.status_code == 200:
            return response.json()
        else:
            return []
    except Exception as e:
        return []

def get_backend_stats() -> Dict[str, Any]:
    """Get backend statistics"""
    try:
        backend_url = get_backend_url()
        response = requests.get(f"{backend_url}/api/v1/stats/")
        if response.status_code == 200:
            return response.json()
        else:
            return {}
    except Exception as e:
        return {}

def test_backend_connection(url: str) -> Dict[str, Any]:
    """Test connection to a specific backend URL"""
    try:
        response = requests.get(f"{url}/system/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            return {
                'success': True,
                'status': 'connected',
                'data': data,
                'response_time': response.elapsed.total_seconds()
            }
        else:
            return {
                'success': False,
                'status': 'error',
                'message': f'HTTP {response.status_code}'
            }
    except requests.exceptions.Timeout:
        return {
            'success': False,
            'status': 'timeout',
            'message': 'Verbindung zeitüberschritten'
        }
    except requests.exceptions.ConnectionError:
        return {
            'success': False,
            'status': 'connection_error',
            'message': 'Verbindung fehlgeschlagen'
        }
    except Exception as e:
        return {
            'success': False,
            'status': 'error',
            'message': str(e)
        }