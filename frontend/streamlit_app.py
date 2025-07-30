import streamlit as st
import requests
import pandas as pd

# Configure page
st.set_page_config(
    page_title="Neuromorphe Traum-Engine v2.0",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'backend_url' not in st.session_state:
    st.session_state.backend_url = "http://localhost:8508"

if 'search_results' not in st.session_state:
    st.session_state.search_results = None

if 'last_query' not in st.session_state:
    st.session_state.last_query = ""

if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []

# Define pages using st.Page for proper URL routing
pages = {
    "Neuromorphe Traum-Engine": [
        st.Page("pages/home.py", title="🏠 Home", url_path="home", default=True),
        st.Page("pages/audio_upload.py", title="🎵 Audio Upload", url_path="audio_upload"),
        st.Page("pages/search.py", title="🔍 Search", url_path="search"),
        st.Page("pages/results.py", title="📊 Results", url_path="results"),
        st.Page("pages/settings.py", title="⚙️ Settings", url_path="settings"),
    ]
}

# Create navigation
pg = st.navigation(pages)

# Sidebar content
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
    
    # Display stats if available
    if hasattr(st.session_state, 'stats') and st.session_state.stats:
        stats = st.session_state.stats
        st.metric("Total Files", stats.get('total_files', 0))
        st.metric("Total Embeddings", stats.get('total_embeddings', 0))
        if stats.get('categories'):
            st.metric("Categories", len(stats['categories']))

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; font-size: 0.8em;'>"  
    "Neuromorphe Traum-Engine v2.0 | Powered by Collective Unconscious AI"  
    "</div>", 
    unsafe_allow_html=True
)

# Run the selected page
pg.run()