import streamlit as st
import requests
import os
from typing import Dict, Any, List, Optional

# Import page modules
from pages.audio_upload import AudioUploadPage
from pages.search import SearchPage
from pages.results import ResultsPage
from pages.settings import SettingsPage

# Configure page - MUST be first Streamlit command
st.set_page_config(
    page_title="Neuromorphe Traum-Engine v2.0",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

class NeuroMorphApp:
    """Main application class for Neuromorphe Traum-Engine v2.0"""
    
    def __init__(self):
        self.backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")
        self.initialize_session_state()
        self.initialize_pages()
    
    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'backend_url' not in st.session_state:
            st.session_state.backend_url = self.backend_url
        
        if 'search_results' not in st.session_state:
            st.session_state.search_results = None
        
        if 'last_query' not in st.session_state:
            st.session_state.last_query = ""
        
        if 'uploaded_files' not in st.session_state:
            st.session_state.uploaded_files = []
        
        if 'backend_status' not in st.session_state:
            st.session_state.backend_status = None
        
        if 'stats' not in st.session_state:
            st.session_state.stats = {}
    
    def initialize_pages(self):
        """Initialize page instances"""
        self.audio_upload_page = AudioUploadPage()
        self.search_page = SearchPage()
        self.results_page = ResultsPage()
        self.settings_page = SettingsPage()
    
    def check_backend_status(self) -> bool:
        """Check if backend is reachable"""
        try:
            response = requests.get(
                f"{st.session_state.backend_url}/system/health", 
                timeout=5
            )
            st.session_state.backend_status = response.status_code == 200
            return st.session_state.backend_status
        except Exception as e:
            st.session_state.backend_status = False
            return False
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics from backend"""
        try:
            response = requests.get(
                f"{st.session_state.backend_url}/api/v1/stems/stats",
                timeout=5
            )
            if response.status_code == 200:
                stats = response.json()
                st.session_state.stats = stats
                return stats
        except Exception:
            pass
        return {}
    
    def render_sidebar(self):
        """Render sidebar with status and navigation"""
        with st.sidebar:
            st.title("🎵 NeuroMorph Engine")
            st.markdown("*Neuromorphe Traum-Engine v2.0*")
            st.markdown("---")
            
            # Backend status
            st.subheader("🔗 System Status")
            
            col1, col2 = st.columns([3, 1])
            with col1:
                if st.button("🔄 Status prüfen", use_container_width=True):
                    with st.spinner("Prüfe Backend..."):
                        self.check_backend_status()
                        self.get_system_stats()
            
            with col2:
                if st.session_state.backend_status:
                    st.success("✅")
                elif st.session_state.backend_status is False:
                    st.error("❌")
                else:
                    st.warning("❓")
            
            # Status details
            if st.session_state.backend_status:
                st.success("Backend verbunden")
            elif st.session_state.backend_status is False:
                st.error("Backend nicht erreichbar")
                st.code("python -m uvicorn src.main:app --reload")
            
            st.markdown("---")
            
            # Quick stats
            if st.session_state.stats:
                st.subheader("📊 Quick Stats")
                stats = st.session_state.stats
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Stems", stats.get('total_stems', 0))
                with col2:
                    st.metric("Embeddings", stats.get('total_embeddings', 0))
                
                if stats.get('categories'):
                    st.metric("Kategorien", len(stats['categories']))
            
            st.markdown("---")
            
            # Settings shortcut
            st.subheader("⚙️ Einstellungen")
            backend_url = st.text_input(
                "Backend URL",
                value=st.session_state.backend_url,
                key="sidebar_backend_url"
            )
            
            if backend_url != st.session_state.backend_url:
                st.session_state.backend_url = backend_url
                st.rerun()
    
    def create_navigation(self):
        """Create navigation using st.navigation"""
        # Define pages with proper structure
        pages = {
            "🎵 Audio Upload": st.Page(
                self.audio_upload_page.render,
                title="Audio Upload",
                icon="🎵"
            ),
            "🔍 Search": st.Page(
                self.search_page.render,
                title="Semantic Search",
                icon="🔍"
            ),
            "📊 Results": st.Page(
                self.results_page.render,
                title="Search Results",
                icon="📊"
            ),
            "⚙️ Settings": st.Page(
                self.settings_page.render,
                title="Settings",
                icon="⚙️"
            )
        }
        
        # Create navigation
        nav = st.navigation(pages)
        return nav
    
    def run(self):
        """Main application entry point"""
        # Render sidebar
        self.render_sidebar()
        
        # Create and run navigation
        nav = self.create_navigation()
        nav.run()
        
        # Footer
        st.markdown("---")
        st.markdown(
            "<div style='text-align: center; color: #666; font-size: 0.8em;'>"  
            "Neuromorphe Traum-Engine v2.0 - Semantic Audio Search with CLAP Embeddings<br>"  
            "Powered by NeuroMorph AI Architecture"
            "</div>", 
            unsafe_allow_html=True
        )

def main():
    """Application entry point"""
    app = NeuroMorphApp()
    app.run()

if __name__ == "__main__":
    main()