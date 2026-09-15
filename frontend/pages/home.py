import streamlit as st
import requests

st.title("🎵 Neuromorphe Traum-Engine v2.0")
st.markdown("---")

# Hero Section
st.markdown("""
## Willkommen im Kollektiven Unbewussten der Elektronischen Musik

Die **Neuromorphe Traum-Engine** ist dein kreativer Partner für die Generierung von neuartigem Raw-Techno. 
Unser intelligenter Dirigent analysiert deine musikalischen Ideen und verwandelt sie in lebendige Klanglandschaften.
""")

# Quick Stats
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("🎵 Audio Stems", "Loading...", delta=None)

with col2:
    st.metric("🔍 Searches", "Loading...", delta=None)

with col3:
    st.metric("⚡ Engine Status", "Checking...", delta=None)

with col4:
    st.metric("🧠 Neural Patterns", "Analyzing...", delta=None)

# Backend Status Check
st.markdown("---")
st.subheader("🔧 System Status")

try:
    response = requests.get(f"{st.session_state.backend_url}/system/health", timeout=5)
    if response.status_code == 200:
        st.success("✅ Backend Engine läuft optimal")
        
        # Try to get some stats
        try:
            stats_response = requests.get(f"{st.session_state.backend_url}/system/stats", timeout=5)
            if stats_response.status_code == 200:
                stats = stats_response.json()
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.info(f"📁 Stems in Database: {stats.get('total_stems', 'N/A')}")
                with col2:
                    st.info(f"🔍 Total Searches: {stats.get('total_searches', 'N/A')}")
                with col3:
                    st.info(f"💾 Database Size: {stats.get('db_size', 'N/A')}")
        except:
            st.info("📊 Detaillierte Statistiken werden geladen...")
    else:
        st.error("❌ Backend nicht erreichbar")
except requests.exceptions.RequestException:
    st.warning("⚠️ Backend-Verbindung wird aufgebaut...")

# Quick Actions
st.markdown("---")
st.subheader("🚀 Schnellstart")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🎵 Audio hochladen", use_container_width=True):
        st.switch_page("pages/audio_upload.py")

with col2:
    if st.button("🔍 Stems durchsuchen", use_container_width=True):
        st.switch_page("pages/search.py")

with col3:
    if st.button("⚙️ Einstellungen", use_container_width=True):
        st.switch_page("pages/settings.py")

# About Section
st.markdown("---")
with st.expander("ℹ️ Über die Neuromorphe Traum-Engine"):
    st.markdown("""
    ### Das Konzept
    
    Die Neuromorphe Traum-Engine basiert auf dem **Zwei-Framework-Prinzip**:
    
    - **🏭 Produktions-Fabrik**: Rechenintensive Analyse und Training von Audio-Embeddings
    - **🎭 Intelligenter Dirigent**: Interaktive Echtzeit-Komposition und kreative Partnerschaften
    
    ### Technologie
    
    - **🧠 CLAP-Embeddings**: Semantische Audio-Analyse
    - **🔍 Vektor-Suche**: Intelligente Stem-Retrieval
    - **⚡ FastAPI Backend**: Hochperformante Audio-Verarbeitung
    - **🎨 Streamlit Frontend**: Intuitive Benutzeroberfläche
    
    ### Vision
    
    Wir erschaffen einen kreativen Partner, der das **Kollektive Unbewusste** der elektronischen Musik 
    anzapft und dir hilft, einzigartige Raw-Techno-Arrangements zu entwickeln.
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.8em;'>
    Neuromorphe Traum-Engine v2.0 | Powered by Collective Unconscious Technology
</div>
""", unsafe_allow_html=True)