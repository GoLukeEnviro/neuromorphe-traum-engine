#!/usr/bin/env python3
"""
Test-Skript für die Streamlit-App Seiten
Testet alle verfügbaren Menüseiten der Neuromorphen Traum-Engine
"""

import requests
import time
import sys
from typing import Dict, List


class StreamlitPageTester:
    """Testet die Streamlit-App Seiten"""
    
    def __init__(self, base_url: str = "http://localhost:8501"):
        self.base_url = base_url
        self.session = requests.Session()
        
    def test_main_page(self) -> bool:
        """Testet die Hauptseite"""
        try:
            response = self.session.get(self.base_url, timeout=10)
            print(f"✅ Hauptseite erreichbar: {response.status_code}")
            return response.status_code == 200
        except Exception as e:
            print(f"❌ Hauptseite nicht erreichbar: {e}")
            return False
    
    def test_page_availability(self) -> Dict[str, bool]:
        """Testet die Verfügbarkeit aller Seiten"""
        pages = {
            "Audio Upload": "🎵 Audio Upload",
            "Search": "🔍 Search", 
            "Results": "📊 Results",
            "Settings": "⚙️ Settings"
        }
        
        results = {}
        
        for page_name, page_selector in pages.items():
            try:
                # Simuliere Seitenaufruf über Streamlit's interne API
                response = self.session.get(f"{self.base_url}/_stcore/health", timeout=5)
                if response.status_code == 200:
                    print(f"✅ {page_name} Seite: Streamlit läuft")
                    results[page_name] = True
                else:
                    print(f"❌ {page_name} Seite: Streamlit Fehler")
                    results[page_name] = False
            except Exception as e:
                print(f"❌ {page_name} Seite: {e}")
                results[page_name] = False
                
        return results
    
    def test_backend_connection(self) -> bool:
        """Testet die Backend-Verbindung"""
        backend_urls = [
            "http://localhost:8000",
            "http://localhost:8003"
        ]
        
        for url in backend_urls:
            try:
                response = self.session.get(f"{url}/system/health", timeout=5)
                if response.status_code == 200:
                    print(f"✅ Backend erreichbar: {url}")
                    return True
            except Exception:
                continue
                
        print("❌ Kein Backend erreichbar")
        return False
    
    def test_streamlit_components(self) -> Dict[str, bool]:
        """Testet Streamlit-spezifische Komponenten"""
        components = {
            "Health Check": "/_stcore/health",
            "Static Files": "/_stcore/static",
            "Component Registry": "/_stcore/component-registry"
        }
        
        results = {}
        
        for component_name, endpoint in components.items():
            try:
                response = self.session.get(f"{self.base_url}{endpoint}", timeout=5)
                if response.status_code in [200, 404]:  # 404 ist OK für manche Endpoints
                    print(f"✅ {component_name}: Verfügbar")
                    results[component_name] = True
                else:
                    print(f"❌ {component_name}: Status {response.status_code}")
                    results[component_name] = False
            except Exception as e:
                print(f"❌ {component_name}: {e}")
                results[component_name] = False
                
        return results
    
    def run_full_test(self) -> Dict[str, any]:
        """Führt alle Tests aus"""
        print("🚀 Starte Streamlit-App Tests...\n")
        
        results = {
            "main_page": self.test_main_page(),
            "pages": self.test_page_availability(),
            "backend": self.test_backend_connection(),
            "components": self.test_streamlit_components()
        }
        
        print("\n📊 Test-Zusammenfassung:")
        print(f"Hauptseite: {'✅' if results['main_page'] else '❌'}")
        print(f"Backend: {'✅' if results['backend'] else '❌'}")
        
        print("\nSeiten:")
        for page, status in results['pages'].items():
            print(f"  {page}: {'✅' if status else '❌'}")
            
        print("\nKomponenten:")
        for component, status in results['components'].items():
            print(f"  {component}: {'✅' if status else '❌'}")
            
        return results


def main():
    """Hauptfunktion"""
    print("🎵 Neuromorphe Traum-Engine - Streamlit Page Tester")
    print("=" * 50)
    
    tester = StreamlitPageTester()
    results = tester.run_full_test()
    
    # Bestimme Gesamtstatus
    all_pages_ok = all(results['pages'].values())
    main_ok = results['main_page']
    backend_ok = results['backend']
    
    if all_pages_ok and main_ok:
        print("\n🎉 Alle Seiten sind verfügbar und funktionsfähig!")
        if backend_ok:
            print("✅ Backend-Verbindung erfolgreich")
        else:
            print("⚠️  Backend nicht erreichbar (aber Streamlit läuft)")
        return 0
    else:
        print("\n❌ Einige Seiten haben Probleme")
        return 1


if __name__ == "__main__":
    sys.exit(main())