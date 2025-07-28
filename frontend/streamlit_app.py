import streamlit as st
import requests
import pandas as pd
from pages.audio_upload import AudioUploadPage
from pages.search import SearchPage
from pages.results import ResultsPage
from pages.settings import SettingsPage

# Configure page
st.set_page_config(
    page_title="Neuromorphe Traum-Engine v2.0",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'backend_url' not in st.session_state:
    st.session_state.backend_url = "http://localhost:8000"

if 'search_results' not in st.session_state:
    st.session_state.search_results = None

if 'last_query' not in st.session_state:
    st.session_state.last_query = ""

if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []

# Create page instances
audio_upload_page = AudioUploadPage()
search_page = SearchPage()
results_page = ResultsPage()
settings_page = SettingsPage()

# Simple navigation using selectbox
page_names = ["🎵 Audio Upload", "🔍 Search", "📊 Results", "⚙️ Settings"]
selected_page = st.sidebar.selectbox("Navigate to:", page_names)

# Main app header
st.sidebar.title("🎵 Neuromorphe Traum-Engine v2.0")
st.sidebar.markdown("---")

# Backend status check
with st.sidebar:
    if st.button("🔄 Check Backend Status"):
        try:
            import requests
            response = requests.get(f"{st.session_state.backend_url}/system/health", timeout=5)
            if response.status_code == 200:
                st.success("✅ Backend is running")
            else:
                st.error("❌ Backend error")
        except Exception as e:
            st.error(f"❌ Backend not reachable: {str(e)}")
    
    st.markdown("---")
    st.markdown("**Quick Stats**")
    
    # Display quick stats if available
    if hasattr(st.session_state, 'stats') and st.session_state.stats:
        stats = st.session_state.stats
        st.metric("Total Files", stats.get('total_files', 0))
        st.metric("Total Embeddings", stats.get('total_embeddings', 0))
        if stats.get('categories'):
            st.metric("Categories", len(stats['categories']))

def main():
    """Main application function"""
    # Render the selected page
    if selected_page == "🎵 Audio Upload":
        audio_upload_page.render()
    elif selected_page == "🔍 Search":
        search_page.render()
    elif selected_page == "📊 Results":
        results_page.render()
    elif selected_page == "⚙️ Settings":
        settings_page.render()

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"  
        "Neuromorphe Traum-Engine v2.0 - Semantic Audio Search with CLAP Embeddings"
        "</div>", 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()