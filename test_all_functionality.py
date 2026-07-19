#!/usr/bin/env python3
"""
Kompletter Funktionstest für die Neuromorphe Traum-Engine
"""

import requests
import json
from typing import Dict, Any

def test_endpoint(method: str, url: str, data: Dict[Any, Any] = None, files: Dict[str, Any] = None) -> bool:
    """Generische Endpunkt-Test-Funktion"""
    try:
        if method.upper() == 'GET':
            response = requests.get(url, timeout=10)
        elif method.upper() == 'POST':
            if files:
                response = requests.post(url, files=files, data=data, timeout=10)
            else:
                response = requests.post(url, json=data, timeout=10)
        else:
            print(f'❌ Unsupported method: {method}')
            return False
        
        if response.status_code in [200, 201]:
            print(f'✅ {method.upper()} {url} - Success')
            return True
        else:
            print(f'❌ {method.upper()} {url} - Failed ({response.status_code}): {response.text[:100]}')
            return False
            
    except Exception as e:
        print(f'❌ {method.upper()} {url} - Error: {e}')
        return False

def main():
    print('🎵 Neuromorphe Traum-Engine - Vollständiger Funktionstest')
    print('=' * 70)
    
    base_url = 'http://localhost:8003'
    results = []
    
    # 1. System Health Checks
    print('\n🏥 System Health Checks:')
    results.append(test_endpoint('GET', f'{base_url}/system/health'))
    results.append(test_endpoint('GET', f'{base_url}/api/v1/audio/health'))
    
    # 2. Audio Upload Test
    print('\n🎧 Audio Upload Test:')
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
    
    files = {'file': ('test_complete.wav', wav_header, 'audio/wav')}
    data = {'description': 'Complete functionality test'}
    results.append(test_endpoint('POST', f'{base_url}/api/v1/audio/upload', data=data, files=files))
    
    # 3. Audio File Management
    print('\n📁 Audio File Management:')
    results.append(test_endpoint('GET', f'{base_url}/api/v1/audio/files'))
    
    # 4. Stems Functionality
    print('\n🎼 Stems Functionality:')
    results.append(test_endpoint('GET', f'{base_url}/api/v1/stems/'))
    results.append(test_endpoint('GET', f'{base_url}/api/v1/stems/search/?query=techno&limit=5'))
    results.append(test_endpoint('GET', f'{base_url}/api/v1/stems/category/kick'))
    
    # 5. Neuromorphic Engine Tests (diese können fehlschlagen, da sie ML-Modelle benötigen)
    print('\n🧠 Neuromorphic Engine Tests (erwartete Fehler bei fehlenden Modellen):')
    results.append(test_endpoint('GET', f'{base_url}/api/v1/neuromorphic/models/available'))
    results.append(test_endpoint('GET', f'{base_url}/api/v1/neuromorphic/generated/info'))
    
    # 6. API Documentation
    print('\n📚 API Documentation:')
    results.append(test_endpoint('GET', f'{base_url}/docs'))
    results.append(test_endpoint('GET', f'{base_url}/openapi.json'))
    
    # Zusammenfassung
    print('\n' + '=' * 70)
    successful_tests = sum(results)
    total_tests = len(results)
    
    print(f'📊 Test-Ergebnisse: {successful_tests}/{total_tests} Tests erfolgreich')
    
    if successful_tests >= total_tests * 0.8:  # 80% Erfolgsrate
        print('🎯 System funktioniert größtenteils korrekt!')
    elif successful_tests >= total_tests * 0.6:  # 60% Erfolgsrate
        print('⚠️  System funktioniert teilweise - einige Features benötigen Aufmerksamkeit.')
    else:
        print('❌ System hat erhebliche Probleme - Debugging erforderlich.')
    
    print('\n🔗 Wichtige URLs:')
    print(f'   • API Dokumentation: {base_url}/docs')
    print(f'   • System Health: {base_url}/system/health')
    print(f'   • Audio Upload: {base_url}/api/v1/audio/upload')
    print(f'   • Stems Search: {base_url}/api/v1/stems/search/')

if __name__ == '__main__':
    main()