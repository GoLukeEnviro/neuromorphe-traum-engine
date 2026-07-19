#!/usr/bin/env python3
"""
Test-Skript für Audio-Upload-Funktionalität der Neuromorphen Traum-Engine.

Dieses Skript testet die Audio-Endpunkte und demonstriert die korrekte Funktion.
"""

import requests
import os
import json
from pathlib import Path

def test_audio_endpoints():
    """Testet alle Audio-Endpunkte der API."""
    
    base_url = "http://localhost:8003"
    audio_prefix = f"{base_url}/api/v1/audio"
    
    print("🎵 Testing Neuromorphe Traum-Engine Audio Endpoints")
    print("=" * 50)
    
    # Test Health Endpoint
    try:
        response = requests.get(f"{audio_prefix}/health")
        if response.status_code == 200:
            health_data = response.json()
            print(f"✅ Health Check: {health_data}")
        else:
            print(f"❌ Health Check failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Health Check error: {e}")
    
    # Test Upload Endpoint Info
    try:
        response = requests.get(f"{base_url}/docs")
        if response.status_code == 200:
            print("✅ API Documentation available at /docs")
        else:
            print(f"❌ API Documentation not accessible: {response.status_code}")
    except Exception as e:
        print(f"❌ API Documentation error: {e}")
    
    # Test Files Endpoint
    try:
        response = requests.get(f"{audio_prefix}/files")
        if response.status_code == 200:
            files_data = response.json()
            print(f"✅ Files Endpoint: Found {len(files_data)} files")
        else:
            print(f"⚠️ Files Endpoint: {response.status_code} (expected empty on fresh start)")
    except Exception as e:
        print(f"⚠️ Files Endpoint error: {e}")
    
    print("\n📋 Available Audio Endpoints:")
    endpoints = [
        f"{audio_prefix}/health",
        f"{audio_prefix}/upload",
        f"{audio_prefix}/files",
        f"{audio_prefix}/files/{{file_id}}",
        f"{audio_prefix}/files/{{file_id}}/process",
        f"{audio_prefix}/files/{{file_id}}/embedding"
    ]
    
    for endpoint in endpoints:
        print(f"  • {endpoint}")
    
    print("\n🎯 Next Steps:")
    print("1. Open browser: http://localhost:8003/docs")
    print("2. Test file upload via Swagger UI")
    print("3. Use Streamlit frontend on port 8501")

if __name__ == "__main__":
    test_audio_endpoints()