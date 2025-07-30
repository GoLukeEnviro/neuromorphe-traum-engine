import streamlit as st

# Minimale Streamlit App für Debugging
st.set_page_config(
    page_title="Neuromorphe Traum-Engine v2.0",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🎵 Neuromorphe Traum-Engine v2.0")
st.subheader("Semantic Audio Search with CLAP Embeddings")

# Einfache Navigation
page = st.selectbox(
    "Navigation",
    ["Audio Upload", "Search", "Results", "Settings"]
)

st.write(f"Aktuelle Seite: {page}")

if page == "Audio Upload":
    st.header("🎵 Audio Upload")
    st.write("Hier können Sie Audio-Dateien hochladen.")
    
elif page == "Search":
    st.header("🔍 Search")
    st.write("Hier können Sie nach Audio-Inhalten suchen.")
    
elif page == "Results":
    st.header("📊 Results")
    st.write("Hier werden die Suchergebnisse angezeigt.")
    
elif page == "Settings":
    st.header("⚙️ Settings")
    st.write("Hier können Sie Einstellungen vornehmen.")

# Status-Anzeige
st.sidebar.header("System Status")
st.sidebar.success("✅ Frontend läuft")
st.sidebar.info("ℹ️ Backend: Nicht verbunden")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>Neuromorphe Traum-Engine v2.0 - Minimal Test App</div>",
    unsafe_allow_html=True
)