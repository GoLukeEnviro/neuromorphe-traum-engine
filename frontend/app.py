import streamlit as st
import os
from typing import Dict, Any

# Import page modules
from pages.search import SearchPage
from pages.results import ResultsPage
from pages.upload import UploadPage
from pages.settings import SettingsPage

def main():
    """Main application entry point"""
    # Page configuration
    st.set_page_config(
        page_title="Neuromorphe Traum-Engine v2.0",
        page_icon="🎵",
        layout="wide" if st.session_state.get('wide_mode', False) else "centered",
        initial_sidebar_state=st.session_state.get('sidebar_state', 'auto')
    )
    
    # Initialize session state
    _initialize_session_state()
    
    # Main header
    st.title("🎵 Neuromorphe Traum-Engine v2.0")
    st.markdown("### KI-gestützte Audio-Suche und Stem-Verwaltung")
    
    # Sidebar navigation
    with st.sidebar:
        st.title("🧠 NeuroMorph")
        st.markdown("*Ihr kreativer Co-Pilot*")
        
        # Navigation
        page = st.selectbox(
            "Navigation",
            [
                "🔍 Suche",
                "📊 Ergebnisse", 
                "📤 Upload",
                "⚙️ Einstellungen"
            ],
            index=_get_current_page_index()
        )
        
        st.markdown("---")
        
        # Quick stats
        _render_sidebar_stats()
        
        st.markdown("---")
        
        # Backend status
        _render_backend_status()
    
    # Route to appropriate page
    if page == "🔍 Suche":
        search_page = SearchPage()
        search_page.render()
    elif page == "📊 Ergebnisse":
        results_page = ResultsPage()
        results_page.render()
    elif page == "📤 Upload":
        upload_page = UploadPage()
        upload_page.render()
    elif page == "⚙️ Einstellungen":
        settings_page = SettingsPage()
        settings_page.render()
    
    # Store current page
    st.session_state.current_page = page

def _initialize_session_state():
    """Initialize session state with default values"""
    defaults = {
        'backend_url': 'http://localhost:8000',
        'current_page': '🔍 Suche',
        'search_history': [],
        'upload_history': [],
        'theme': 'auto',
        'wide_mode': False,
        'sidebar_state': 'auto',
        'results_per_page': 10,
        'show_advanced_options': False,
        'autoplay': False,
        'show_waveform': True,
        'show_success_messages': True,
        'show_progress_bars': True,
        'max_file_size_mb': 50,
        'supported_formats': ['wav', 'mp3', 'flac', 'ogg'],
        'auto_process': True,
        'batch_size': 5,
        'default_search_limit': 10,
        'default_similarity_threshold': 0.3,
        'enable_fuzzy_search': True,
        'cache_search_results': True,
        'max_search_history': 20,
        'enable_category_boost': False,
        'enable_bpm_filtering': True
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def _get_current_page_index() -> int:
    """Get the index of the current page for selectbox"""
    pages = ["🔍 Suche", "📊 Ergebnisse", "📤 Upload", "⚙️ Einstellungen"]
    current = st.session_state.get('current_page', '🔍 Suche')
    try:
        return pages.index(current)
    except ValueError:
        return 0

def _render_sidebar_stats():
    """Render quick stats in sidebar"""
    st.subheader("📊 Schnellübersicht")
    
    # Search history count
    search_count = len(st.session_state.get('search_history', []))
    st.metric("Letzte Suchen", search_count)
    
    # Upload history count
    upload_count = len(st.session_state.get('upload_history', []))
    st.metric("Uploads", upload_count)
    
    # Results count
    results = st.session_state.get('search_results', [])
    if results:
        st.metric("Aktuelle Ergebnisse", len(results))

def _render_backend_status():
    """Render backend connection status in sidebar"""
    st.subheader("🔗 Backend-Status")
    
    backend_url = st.session_state.get('backend_url', 'http://localhost:8000')
    
    try:
        import requests
        response = requests.get(f"{backend_url}/system/health", timeout=3)
        if response.status_code == 200:
            st.success("🟢 Verbunden")
            data = response.json()
            if 'version' in data:
                st.caption(f"Version: {data['version']}")
        else:
            st.error("🔴 Fehler")
    except:
        st.error("🔴 Offline")
    
    st.caption(f"URL: {backend_url}")

if __name__ == "__main__":
    main()