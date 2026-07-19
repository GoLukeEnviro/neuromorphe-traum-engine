#!/usr/bin/env python3
"""
Test script for audio upload functionality
"""

import requests
import json
from pathlib import Path

def test_audio_upload():
    """Test the audio upload endpoint with a minimal WAV file"""
    url = 'http://localhost:8003/api/v1/audio/upload'
    
    # Create a minimal WAV file content
    # RIFF header + WAV format + minimal data
    wav_header = (
        b'RIFF'  # ChunkID
        b'\x24\x00\x00\x00'  # ChunkSize (36 bytes)
        b'WAVE'  # Format
        b'fmt '  # Subchunk1ID
        b'\x10\x00\x00\x00'  # Subchunk1Size (16)
        b'\x01\x00'  # AudioFormat (PCM)
        b'\x01\x00'  # NumChannels (1)
        b'\x44\xac\x00\x00'  # SampleRate (44100)
        b'\x88X\x01\x00'  # ByteRate
        b'\x02\x00'  # BlockAlign
        b'\x10\x00'  # BitsPerSample (16)
        b'data'  # Subchunk2ID
        b'\x00\x00\x00\x00'  # Subchunk2Size (0)
    )
    
    try:
        files = {'file': ('test_upload.wav', wav_header, 'audio/wav')}
        data = {'description': 'Test upload from Python script'}
        
        print('🎵 Testing Audio Upload Functionality')
        print('=' * 50)
        
        response = requests.post(url, files=files, data=data, timeout=10)
        
        print(f'Status Code: {response.status_code}')
        
        if response.status_code == 200:
            result = response.json()
            print('✅ Upload successful!')
            print(f'File ID: {result.get("file_id", "N/A")}')
            print(f'Filename: {result.get("filename", "N/A")}')
            print(f'Size: {result.get("size", "N/A")} bytes')
            return True
        else:
            print(f'❌ Upload failed: {response.text}')
            return False
            
    except requests.exceptions.ConnectionError as e:
        print(f'❌ Connection error: Server not reachable on {url}')
        print(f'Details: {e}')
        return False
    except Exception as e:
        print(f'❌ Unexpected error: {e}')
        return False

def test_file_listing():
    """Test the file listing endpoint"""
    url = 'http://localhost:8003/api/v1/audio/files'
    
    try:
        response = requests.get(url, timeout=5)
        
        if response.status_code == 200:
            file_ids = response.json()
            print(f'\n📁 Files in database: {len(file_ids)} files')
            for file_id in file_ids:
                print(f'  - File ID: {file_id}')
            return True
        else:
            print(f'❌ File listing failed: {response.text}')
            return False
            
    except Exception as e:
        print(f'❌ File listing error: {e}')
        return False

if __name__ == '__main__':
    # Test upload functionality
    upload_success = test_audio_upload()
    
    # Test file listing
    listing_success = test_file_listing()
    
    if upload_success and listing_success:
        print('\n🎯 All tests passed! Audio upload system is working.')
    else:
        print('\n⚠️  Some tests failed. Check the server logs.')