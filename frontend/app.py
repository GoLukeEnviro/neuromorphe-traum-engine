import streamlit as st
import requests
import json
from typing import List, Dict, Any
import os

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
API_STEMS_URL = f"{API_BASE_URL}/api/v1/stems"

def check_api_health() -> bool:
    """Check if the backend API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/system/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def search_stems(query: str, top_k: int = 5, category: str = None) -> List[Dict[str, Any]]:
    """Search stems using the backend API"""
    try:
        params = {"prompt": query, "top_k": top_k}
        if category:
            params["category"] = category
        
        response = requests.get(f"{API_STEMS_URL}/search/", params=params)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return []
    except Exception as e:
        st.error(f"Connection Error: {str(e)}")
        return []

def get_all_stems(skip: int = 0, limit: int = 20) -> List[Dict[str, Any]]:
    """Get all stems from the backend API"""
    try:
        params = {"skip": skip, "limit": limit}
        response = requests.get(f"{API_STEMS_URL}/", params=params)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return []
    except Exception as e:
        st.error(f"Connection Error: {str(e)}")
        return []

def main():
    st.set_page_config(
        page_title="Neuromorphe Traum-Engine v2.0",
        page_icon="🎵",
        layout="wide"
    )
    
    st.title("🎵 Neuromorphe Traum-Engine v2.0")
    st.markdown("### Semantische Audio-Suche und Stem-Verwaltung")
    
    # Check API health
    if not check_api_health():
        st.error("⚠️ Backend API ist nicht erreichbar. Bitte starten Sie den FastAPI-Server.")
        st.code("python -m uvicorn src.main:app --reload")
        return
    
    st.success("✅ Backend API ist verbunden")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Seite auswählen",
        ["🔍 Suche", "📁 Alle Stems", "📤 Upload"]
    )
    
    if page == "🔍 Suche":
        show_search_page()
    elif page == "📁 Alle Stems":
        show_browse_page()
    elif page == "📤 Upload":
        show_upload_page()

def show_search_page():
    """Show the search page"""
    st.header("🔍 Semantische Suche")
    
    # Search form
    with st.form("search_form"):
        col1, col2 = st.columns([3, 1])
        
        with col1:
            query = st.text_input(
                "Suchbegriff",
                placeholder="z.B. 'energetic drum loop', 'melodic bass', 'ambient pad'..."
            )
        
        with col2:
            top_k = st.number_input("Anzahl Ergebnisse", min_value=1, max_value=50, value=5)
        
        category_filter = st.selectbox(
            "Kategorie (optional)",
            ["Alle", "drum", "bass", "melody", "pad", "fx", "vocal"]
        )
        
        search_button = st.form_submit_button("🔍 Suchen")
    
    if search_button and query:
        category = None if category_filter == "Alle" else category_filter
        
        with st.spinner("Suche läuft..."):
            results = search_stems(query, top_k, category)
        
        if results:
            st.success(f"✅ {len(results)} Ergebnisse gefunden")
            
            for i, result in enumerate(results, 1):
                with st.expander(f"#{i} - {os.path.basename(result['path'])} (Ähnlichkeit: {result['similarity']:.3f})"):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.write(f"**Pfad:** {result['path']}")
                        st.write(f"**Kategorie:** {result.get('category', 'N/A')}")
                        st.write(f"**BPM:** {result.get('bpm', 'N/A')}")
                        st.write(f"**Key:** {result.get('key', 'N/A')}")
                        if result.get('tags'):
                            st.write(f"**Tags:** {result['tags']}")
                    
                    with col2:
                        st.metric("Ähnlichkeit", f"{result['similarity']:.3f}")
                        if result.get('quality_ok') is not None:
                            quality = "✅ Gut" if result['quality_ok'] else "⚠️ Prüfen"
                            st.write(f"**Qualität:** {quality}")
        else:
            st.warning("Keine Ergebnisse gefunden.")

def show_browse_page():
    """Show the browse all stems page"""
    st.header("📁 Alle Stems durchsuchen")
    
    # Pagination controls
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        page_size = st.selectbox("Stems pro Seite", [10, 20, 50], index=1)
    
    with col2:
        page_number = st.number_input("Seite", min_value=1, value=1)
    
    skip = (page_number - 1) * page_size
    
    with st.spinner("Lade Stems..."):
        stems = get_all_stems(skip=skip, limit=page_size)
    
    if stems:
        st.success(f"✅ {len(stems)} Stems geladen")
        
        for i, stem in enumerate(stems, skip + 1):
            with st.expander(f"#{i} - {os.path.basename(stem['path'])}"):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write(f"**Pfad:** {stem['path']}")
                    st.write(f"**Kategorie:** {stem.get('category', 'N/A')}")
                    st.write(f"**BPM:** {stem.get('bpm', 'N/A')}")
                    st.write(f"**Key:** {stem.get('key', 'N/A')}")
                    if stem.get('tags'):
                        st.write(f"**Tags:** {stem['tags']}")
                    st.write(f"**Importiert:** {stem.get('imported_at', 'N/A')}")
                
                with col2:
                    if stem.get('quality_ok') is not None:
                        quality = "✅ Gut" if stem['quality_ok'] else "⚠️ Prüfen"
                        st.write(f"**Qualität:** {quality}")
                    
                    if stem.get('clap_embedding'):
                        st.write("**CLAP:** ✅ Verfügbar")
                    else:
                        st.write("**CLAP:** ❌ Nicht verfügbar")
    else:
        st.warning("Keine Stems gefunden.")

def show_upload_page():
    """Show the upload page"""
    st.header("📤 Audio-Dateien hochladen")
    
    st.info("🚧 Upload-Funktionalität wird in einer zukünftigen Version implementiert.")
    
    st.markdown("""
    **Geplante Features:**
    - Drag & Drop Upload für Audio-Dateien
    - Automatische Metadaten-Extraktion
    - CLAP-Embedding-Generierung
    - Batch-Upload für mehrere Dateien
    - Fortschrittsanzeige
    """)
    
    # Placeholder for future upload functionality
    uploaded_files = st.file_uploader(
        "Audio-Dateien auswählen",
        type=['wav', 'mp3', 'flac', 'aiff'],
        accept_multiple_files=True,
        disabled=True
    )
    
    if uploaded_files:
        st.warning("Upload-Funktionalität ist noch nicht implementiert.")

if __name__ == "__main__":
    main()