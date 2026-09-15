# Streamlit Frontend - Technische Spezifikation
## Neuromorphe Traum-Engine v2.0

### Frontend-Architektur

Das Frontend basiert auf Streamlit mit einer modularen Multi-Page-Architektur und modernem, futuristischem Design:

```
frontend/
├── streamlit_app.py            # Hauptanwendung & Navigation
├── config/
│   ├── __init__.py
│   ├── settings.py             # Frontend-Konfiguration
│   └── theme.py                # UI-Theme & Styling
├── pages/
│   ├── __init__.py
│   ├── upload.py               # Audio-Upload Interface
│   ├── search.py               # Such-Interface
│   ├── results.py              # Ergebnis-Darstellung
│   ├── analytics.py            # System-Analytics
│   └── settings.py             # Anwendungseinstellungen
├── components/
│   ├── __init__.py
│   ├── audio_player.py         # Audio-Player Komponente
│   ├── file_uploader.py        # Erweiterte Upload-Komponente
│   ├── search_filters.py       # Such-Filter Komponente
│   ├── progress_tracker.py     # Progress-Anzeige
│   └── data_visualizer.py      # Datenvisualisierung
├── utils/
│   ├── __init__.py
│   ├── api_client.py           # Backend-API Client
│   ├── state_manager.py        # Session State Management
│   ├── ui_helpers.py           # UI-Hilfsfunktionen
│   └── validators.py           # Input-Validierung
├── assets/
│   ├── styles.css              # Custom CSS
│   ├── icons/                  # SVG Icons
│   └── images/                 # UI Images
└── requirements.txt            # Frontend Dependencies
```

## Core Configuration

### config/settings.py
```python
from pydantic import BaseSettings
from typing import List, Dict, Any
import os

class FrontendSettings(BaseSettings):
    # App Configuration
    app_title: str = "🧠 Neuromorphe Traum-Engine"
    app_subtitle: str = "Semantische Audio-Suche mit KI"
    app_version: str = "2.0.0"
    
    # Backend API
    backend_url: str = "http://localhost:8000"
    api_timeout: int = 30
    
    # UI Configuration
    page_title: str = "Neuromorphe Traum-Engine"
    page_icon: str = "🧠"
    layout: str = "wide"
    initial_sidebar_state: str = "expanded"
    
    # Upload Settings
    max_file_size_mb: int = 100
    allowed_extensions: List[str] = [".wav", ".mp3", ".flac", ".m4a"]
    
    # Search Settings
    default_search_limit: int = 20
    max_search_limit: int = 100
    
    # Theme
    primary_color: str = "#00f5ff"
    background_color: str = "#0e1117"
    secondary_background_color: str = "#262730"
    text_color: str = "#fafafa"
    
    class Config:
        env_file = ".env"
        env_prefix = "FRONTEND_"

settings = FrontendSettings()
```

### config/theme.py
```python
import streamlit as st
from .settings import settings

def apply_custom_theme():
    """Wendet das benutzerdefinierte Theme an"""
    
    # Custom CSS für futuristisches Design
    custom_css = f"""
    <style>
    /* Hauptcontainer */
    .main .block-container {{
        padding-top: 2rem;
        padding-bottom: 2rem;
        background: linear-gradient(135deg, #0e1117 0%, #1a1d29 100%);
    }}
    
    /* Sidebar Styling */
    .css-1d391kg {{
        background: linear-gradient(180deg, #262730 0%, #1a1d29 100%);
        border-right: 2px solid {settings.primary_color};
    }}
    
    /* Buttons */
    .stButton > button {{
        background: linear-gradient(45deg, {settings.primary_color}, #0099cc);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 245, 255, 0.3);
    }}
    
    .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 245, 255, 0.5);
    }}
    
    /* File Uploader */
    .stFileUploader > div {{
        border: 2px dashed {settings.primary_color};
        border-radius: 15px;
        background: rgba(0, 245, 255, 0.05);
        transition: all 0.3s ease;
    }}
    
    .stFileUploader > div:hover {{
        background: rgba(0, 245, 255, 0.1);
        border-color: #66ffff;
    }}
    
    /* Progress Bars */
    .stProgress > div > div > div {{
        background: linear-gradient(90deg, {settings.primary_color}, #66ffff);
        border-radius: 10px;
    }}
    
    /* Metrics */
    .metric-container {{
        background: linear-gradient(135deg, rgba(0, 245, 255, 0.1), rgba(0, 153, 204, 0.1));
        border: 1px solid {settings.primary_color};
        border-radius: 15px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 4px 15px rgba(0, 245, 255, 0.2);
    }}
    
    /* Audio Player */
    .audio-player {{
        background: linear-gradient(135deg, #262730, #1a1d29);
        border: 1px solid {settings.primary_color};
        border-radius: 15px;
        padding: 1rem;
        margin: 1rem 0;
    }}
    
    /* Search Results */
    .search-result {{
        background: linear-gradient(135deg, rgba(38, 39, 48, 0.8), rgba(26, 29, 41, 0.8));
        border: 1px solid rgba(0, 245, 255, 0.3);
        border-radius: 15px;
        padding: 1rem;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
    }}
    
    .search-result:hover {{
        border-color: {settings.primary_color};
        box-shadow: 0 4px 15px rgba(0, 245, 255, 0.3);
        transform: translateY(-2px);
    }}
    
    /* Headers */
    h1, h2, h3 {{
        color: {settings.primary_color};
        text-shadow: 0 0 10px rgba(0, 245, 255, 0.5);
    }}
    
    /* Selectbox */
    .stSelectbox > div > div {{
        background: linear-gradient(135deg, #262730, #1a1d29);
        border: 1px solid {settings.primary_color};
        border-radius: 10px;
    }}
    
    /* Text Input */
    .stTextInput > div > div > input {{
        background: linear-gradient(135deg, #262730, #1a1d29);
        border: 1px solid {settings.primary_color};
        border-radius: 10px;
        color: {settings.text_color};
    }}
    
    /* Slider */
    .stSlider > div > div > div {{
        background: {settings.primary_color};
    }}
    
    /* Expander */
    .streamlit-expanderHeader {{
        background: linear-gradient(135deg, rgba(0, 245, 255, 0.1), rgba(0, 153, 204, 0.1));
        border: 1px solid {settings.primary_color};
        border-radius: 10px;
    }}
    
    /* Loading Animation */
    @keyframes pulse {{
        0% {{ opacity: 0.6; }}
        50% {{ opacity: 1; }}
        100% {{ opacity: 0.6; }}
    }}
    
    .loading {{
        animation: pulse 2s infinite;
    }}
    
    /* Glow Effect */
    .glow {{
        box-shadow: 0 0 20px {settings.primary_color};
    }}
    </style>
    """
    
    st.markdown(custom_css, unsafe_allow_html=True)

def create_metric_card(title: str, value: str, delta: str = None, help_text: str = None):
    """Erstellt eine stilisierte Metrik-Karte"""
    
    delta_html = f"<div style='color: #00ff00; font-size: 0.8rem;'>↗ {delta}</div>" if delta else ""
    help_html = f"<div style='color: #888; font-size: 0.7rem; margin-top: 0.5rem;'>{help_text}</div>" if help_text else ""
    
    card_html = f"""
    <div class="metric-container">
        <div style="color: {settings.primary_color}; font-size: 0.9rem; font-weight: bold;">{title}</div>
        <div style="color: {settings.text_color}; font-size: 1.5rem; font-weight: bold; margin: 0.5rem 0;">{value}</div>
        {delta_html}
        {help_html}
    </div>
    """
    
    st.markdown(card_html, unsafe_allow_html=True)
```

## Main Application

### streamlit_app.py
```python
import streamlit as st
from config.settings import settings
from config.theme import apply_custom_theme
from utils.state_manager import initialize_session_state
from utils.api_client import APIClient

# Page Configuration
st.set_page_config(
    page_title=settings.page_title,
    page_icon=settings.page_icon,
    layout=settings.layout,
    initial_sidebar_state=settings.initial_sidebar_state
)

# Apply Custom Theme
apply_custom_theme()

# Initialize Session State
initialize_session_state()

# Initialize API Client
if 'api_client' not in st.session_state:
    st.session_state.api_client = APIClient(settings.backend_url)

def main():
    """Hauptanwendung mit Navigation"""
    
    # Header
    st.markdown(f"""
    <div style="text-align: center; padding: 2rem 0;">
        <h1 style="font-size: 3rem; margin-bottom: 0.5rem;">{settings.app_title}</h1>
        <p style="font-size: 1.2rem; color: #888; margin-bottom: 2rem;">{settings.app_subtitle}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar Navigation
    with st.sidebar:
        st.markdown(f"""
        <div style="text-align: center; padding: 1rem 0; border-bottom: 1px solid {settings.primary_color}; margin-bottom: 2rem;">
            <h2 style="margin: 0;">Navigation</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Navigation Menu
        page = st.selectbox(
            "Seite auswählen",
            ["🎵 Audio Upload", "🔍 Suche", "📊 Ergebnisse", "📈 Analytics", "⚙️ Einstellungen"],
            key="navigation"
        )
        
        # System Status
        st.markdown("---")
        st.markdown("### System Status")
        
        # Backend Connection Check
        try:
            health = st.session_state.api_client.check_health()
            if health:
                st.success("✅ Backend verbunden")
            else:
                st.error("❌ Backend nicht erreichbar")
        except Exception:
            st.error("❌ Backend nicht erreichbar")
        
        # Quick Stats
        try:
            stats = st.session_state.api_client.get_stats()
            if stats:
                st.metric("Audio-Dateien", stats.get('total_files', 0))
                st.metric("Verarbeitete Dateien", stats.get('processed_files', 0))
        except Exception:
            pass
    
    # Page Routing
    if page == "🎵 Audio Upload":
        from pages.upload import show_upload_page
        show_upload_page()
    elif page == "🔍 Suche":
        from pages.search import show_search_page
        show_search_page()
    elif page == "📊 Ergebnisse":
        from pages.results import show_results_page
        show_results_page()
    elif page == "📈 Analytics":
        from pages.analytics import show_analytics_page
        show_analytics_page()
    elif page == "⚙️ Einstellungen":
        from pages.settings import show_settings_page
        show_settings_page()

if __name__ == "__main__":
    main()
```

## Page Components

### pages/upload.py
```python
import streamlit as st
import asyncio
from typing import List
from pathlib import Path
from config.settings import settings
from config.theme import create_metric_card
from components.file_uploader import enhanced_file_uploader
from components.progress_tracker import show_progress_tracker
from utils.validators import validate_audio_file

def show_upload_page():
    """Audio-Upload Seite"""
    
    st.markdown("## 🎵 Audio-Dateien hochladen")
    st.markdown("Laden Sie Ihre Audio-Dateien hoch, um sie in die Traum-Engine zu integrieren.")
    
    # Upload Statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        create_metric_card("Erlaubte Formate", ", ".join(settings.allowed_extensions))
    
    with col2:
        create_metric_card("Max. Dateigröße", f"{settings.max_file_size_mb} MB")
    
    with col3:
        try:
            stats = st.session_state.api_client.get_upload_stats()
            create_metric_card("Heute hochgeladen", str(stats.get('today_uploads', 0)))
        except:
            create_metric_card("Heute hochgeladen", "0")
    
    with col4:
        try:
            stats = st.session_state.api_client.get_upload_stats()
            create_metric_card("Warteschlange", str(stats.get('queue_length', 0)))
        except:
            create_metric_card("Warteschlange", "0")
    
    st.markdown("---")
    
    # File Upload Section
    st.markdown("### Datei-Upload")
    
    # Enhanced File Uploader
    uploaded_files = enhanced_file_uploader(
        label="Audio-Dateien hier ablegen oder auswählen",
        type=settings.allowed_extensions,
        accept_multiple_files=True,
        max_size_mb=settings.max_file_size_mb
    )
    
    if uploaded_files:
        st.markdown("### Upload-Vorschau")
        
        # Validate Files
        valid_files = []
        invalid_files = []
        
        for file in uploaded_files:
            if validate_audio_file(file, settings.allowed_extensions, settings.max_file_size_mb):
                valid_files.append(file)
            else:
                invalid_files.append(file)
        
        # Show validation results
        if invalid_files:
            st.error(f"❌ {len(invalid_files)} Datei(en) sind ungültig:")
            for file in invalid_files:
                st.write(f"- {file.name}")
        
        if valid_files:
            st.success(f"✅ {len(valid_files)} Datei(en) bereit zum Upload:")
            
            # File Preview Table
            file_data = []
            for file in valid_files:
                file_data.append({
                    "Dateiname": file.name,
                    "Größe": f"{file.size / (1024*1024):.2f} MB",
                    "Typ": file.type or "Unbekannt"
                })
            
            st.dataframe(file_data, use_container_width=True)
            
            # Upload Controls
            col1, col2, col3 = st.columns([1, 1, 2])
            
            with col1:
                if st.button("🚀 Upload starten", type="primary", use_container_width=True):
                    start_upload_process(valid_files)
            
            with col2:
                if st.button("🗑️ Auswahl löschen", use_container_width=True):
                    st.rerun()
            
            with col3:
                # Upload Options
                with st.expander("⚙️ Upload-Optionen"):
                    auto_process = st.checkbox("Automatische Verarbeitung", value=True)
                    priority = st.selectbox("Priorität", ["Normal", "Hoch", "Niedrig"])
                    notify_completion = st.checkbox("Benachrichtigung bei Fertigstellung", value=True)
    
    # Active Uploads Section
    if 'active_uploads' in st.session_state and st.session_state.active_uploads:
        st.markdown("---")
        st.markdown("### Aktive Uploads")
        show_active_uploads()
    
    # Recent Uploads Section
    st.markdown("---")
    st.markdown("### Kürzlich hochgeladene Dateien")
    show_recent_uploads()

def start_upload_process(files: List):
    """Startet den Upload-Prozess für die ausgewählten Dateien"""
    
    if 'active_uploads' not in st.session_state:
        st.session_state.active_uploads = {}
    
    progress_placeholder = st.empty()
    
    for i, file in enumerate(files):
        try:
            # Upload starten
            with progress_placeholder.container():
                st.info(f"📤 Uploading {file.name}...")
                progress_bar = st.progress(0)
            
            # API Call
            response = st.session_state.api_client.upload_audio_file(file)
            
            if response and 'job_id' in response:
                # Job tracking hinzufügen
                st.session_state.active_uploads[response['job_id']] = {
                    'filename': file.name,
                    'status': 'uploading',
                    'progress': 0
                }
                
                progress_bar.progress(100)
                st.success(f"✅ {file.name} erfolgreich hochgeladen!")
            else:
                st.error(f"❌ Fehler beim Upload von {file.name}")
        
        except Exception as e:
            st.error(f"❌ Fehler beim Upload von {file.name}: {str(e)}")
    
    # Refresh page to show active uploads
    st.rerun()

def show_active_uploads():
    """Zeigt aktive Upload-Jobs an"""
    
    for job_id, job_data in st.session_state.active_uploads.items():
        try:
            # Status vom Backend abrufen
            status = st.session_state.api_client.get_processing_status(job_id)
            
            if status:
                # Update local state
                st.session_state.active_uploads[job_id].update({
                    'status': status.get('status', 'unknown'),
                    'progress': status.get('progress', 0),
                    'message': status.get('message', '')
                })
                
                # Show progress
                show_progress_tracker(
                    filename=job_data['filename'],
                    status=status.get('status', 'unknown'),
                    progress=status.get('progress', 0),
                    message=status.get('message', '')
                )
                
                # Remove completed jobs
                if status.get('status') in ['completed', 'failed']:
                    del st.session_state.active_uploads[job_id]
                    st.rerun()
        
        except Exception as e:
            st.error(f"Fehler beim Abrufen des Status für {job_data['filename']}: {str(e)}")

def show_recent_uploads():
    """Zeigt kürzlich hochgeladene Dateien an"""
    
    try:
        recent_files = st.session_state.api_client.get_recent_uploads(limit=10)
        
        if recent_files:
            for file_data in recent_files:
                with st.container():
                    col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
                    
                    with col1:
                        st.write(f"🎵 **{file_data['filename']}**")
                    
                    with col2:
                        st.write(f"⏱️ {file_data['duration']:.1f}s")
                    
                    with col3:
                        status_icon = "✅" if file_data['processed'] else "⏳"
                        status_text = "Verarbeitet" if file_data['processed'] else "In Bearbeitung"
                        st.write(f"{status_icon} {status_text}")
                    
                    with col4:
                        if st.button("🗑️", key=f"delete_{file_data['id']}", help="Datei löschen"):
                            delete_audio_file(file_data['id'])
                    
                    st.markdown("---")
        else:
            st.info("Noch keine Dateien hochgeladen.")
    
    except Exception as e:
        st.error(f"Fehler beim Laden der kürzlichen Uploads: {str(e)}")

def delete_audio_file(file_id: int):
    """Löscht eine Audio-Datei"""
    
    try:
        response = st.session_state.api_client.delete_audio_file(file_id)
        if response:
            st.success("Datei erfolgreich gelöscht!")
            st.rerun()
        else:
            st.error("Fehler beim Löschen der Datei")
    except Exception as e:
        st.error(f"Fehler beim Löschen: {str(e)}")
```

### pages/search.py
```python
import streamlit as st
from typing import Optional, Dict, Any
from config.settings import settings
from config.theme import create_metric_card
from components.search_filters import show_search_filters
from components.audio_player import show_audio_player
from utils.state_manager import get_search_history, add_to_search_history

def show_search_page():
    """Such-Interface Seite"""
    
    st.markdown("## 🔍 Semantische Audio-Suche")
    st.markdown("Finden Sie ähnliche Audio-Dateien durch Text-Beschreibungen oder Audio-Beispiele.")
    
    # Search Statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        try:
            stats = st.session_state.api_client.get_search_stats()
            create_metric_card("Verfügbare Dateien", str(stats.get('total_files', 0)))
        except:
            create_metric_card("Verfügbare Dateien", "0")
    
    with col2:
        try:
            stats = st.session_state.api_client.get_search_stats()
            create_metric_card("Heute gesucht", str(stats.get('today_searches', 0)))
        except:
            create_metric_card("Heute gesucht", "0")
    
    with col3:
        history = get_search_history()
        create_metric_card("Such-Verlauf", str(len(history)))
    
    with col4:
        create_metric_card("Durchschn. Antwortzeit", "< 500ms")
    
    st.markdown("---")
    
    # Search Interface
    search_tab1, search_tab2 = st.tabs(["📝 Text-Suche", "🎵 Audio-Suche"])
    
    with search_tab1:
        show_text_search_interface()
    
    with search_tab2:
        show_audio_search_interface()
    
    # Search Filters Sidebar
    with st.sidebar:
        st.markdown("### 🎛️ Such-Filter")
        filters = show_search_filters()
        
        # Search History
        st.markdown("---")
        st.markdown("### 📚 Such-Verlauf")
        show_search_history_sidebar()

def show_text_search_interface():
    """Text-basierte Suchschnittstelle"""
    
    st.markdown("### Text-basierte Suche")
    st.markdown("Beschreiben Sie die Art von Audio, die Sie suchen:")
    
    # Search Input
    col1, col2 = st.columns([4, 1])
    
    with col1:
        search_query = st.text_input(
            "Suchbegriff eingeben",
            placeholder="z.B. 'energetic techno beat', 'calm piano melody', 'upbeat drums'...",
            key="text_search_input"
        )
    
    with col2:
        search_button = st.button("🔍 Suchen", type="primary", use_container_width=True)
    
    # Search Suggestions
    if search_query and len(search_query) > 2:
        try:
            suggestions = st.session_state.api_client.get_search_suggestions(search_query)
            if suggestions:
                st.markdown("**Vorschläge:**")
                suggestion_cols = st.columns(min(len(suggestions), 4))
                for i, suggestion in enumerate(suggestions[:4]):
                    with suggestion_cols[i]:
                        if st.button(f"💡 {suggestion}", key=f"suggestion_{i}"):
                            st.session_state.text_search_input = suggestion
                            st.rerun()
        except:
            pass
    
    # Execute Search
    if search_button and search_query:
        execute_text_search(search_query)
    
    # Quick Search Examples
    st.markdown("---")
    st.markdown("**Beispiel-Suchen:**")
    
    example_cols = st.columns(4)
    examples = [
        "energetic techno",
        "calm ambient",
        "heavy bass",
        "melodic piano"
    ]
    
    for i, example in enumerate(examples):
        with example_cols[i]:
            if st.button(f"🎯 {example}", key=f"example_{i}"):
                st.session_state.text_search_input = example
                st.rerun()

def show_audio_search_interface():
    """Audio-basierte Suchschnittstelle"""
    
    st.markdown("### Audio-basierte Suche")
    st.markdown("Laden Sie eine Audio-Datei hoch, um ähnliche Sounds zu finden:")
    
    # Audio Upload for Search
    uploaded_audio = st.file_uploader(
        "Audio-Datei für Suche hochladen",
        type=settings.allowed_extensions,
        key="audio_search_upload"
    )
    
    if uploaded_audio:
        # Audio Preview
        st.markdown("**Audio-Vorschau:**")
        st.audio(uploaded_audio, format='audio/wav')
        
        # Search Options
        col1, col2 = st.columns(2)
        
        with col1:
            similarity_threshold = st.slider(
                "Ähnlichkeits-Schwellenwert",
                min_value=0.1,
                max_value=1.0,
                value=0.7,
                step=0.1,
                help="Höhere Werte = strengere Ähnlichkeit"
            )
        
        with col2:
            search_limit = st.number_input(
                "Max. Ergebnisse",
                min_value=5,
                max_value=settings.max_search_limit,
                value=settings.default_search_limit,
                step=5
            )
        
        # Execute Audio Search
        if st.button("🎵 Audio-Suche starten", type="primary", use_container_width=True):
            execute_audio_search(uploaded_audio, similarity_threshold, search_limit)

def execute_text_search(query: str):
    """Führt Text-Suche aus"""
    
    with st.spinner("🔍 Suche läuft..."):
        try:
            # Get filters from sidebar
            filters = st.session_state.get('search_filters', {})
            
            # Execute search
            results = st.session_state.api_client.text_search(
                query=query,
                filters=filters,
                limit=settings.default_search_limit
            )
            
            if results:
                # Add to search history
                add_to_search_history({
                    'type': 'text',
                    'query': query,
                    'results_count': len(results),
                    'timestamp': 'now'
                })
                
                # Store results in session state
                st.session_state.search_results = results
                st.session_state.search_query = query
                
                # Show results
                show_search_results(results, query)
            else:
                st.warning("Keine Ergebnisse gefunden. Versuchen Sie andere Suchbegriffe.")
        
        except Exception as e:
            st.error(f"Fehler bei der Suche: {str(e)}")

def execute_audio_search(audio_file, similarity_threshold: float, limit: int):
    """Führt Audio-Suche aus"""
    
    with st.spinner("🎵 Audio wird analysiert..."):
        try:
            # Get filters from sidebar
            filters = st.session_state.get('search_filters', {})
            
            # Execute audio search
            results = st.session_state.api_client.audio_search(
                audio_file=audio_file,
                similarity_threshold=similarity_threshold,
                filters=filters,
                limit=limit
            )
            
            if results:
                # Add to search history
                add_to_search_history({
                    'type': 'audio',
                    'query': audio_file.name,
                    'results_count': len(results),
                    'timestamp': 'now'
                })
                
                # Store results in session state
                st.session_state.search_results = results
                st.session_state.search_query = f"Audio: {audio_file.name}"
                
                # Show results
                show_search_results(results, f"Audio: {audio_file.name}")
            else:
                st.warning("Keine ähnlichen Audio-Dateien gefunden.")
        
        except Exception as e:
            st.error(f"Fehler bei der Audio-Suche: {str(e)}")

def show_search_results(results: list, query: str):
    """Zeigt Suchergebnisse an"""
    
    st.markdown("---")
    st.markdown(f"### 📊 Suchergebnisse für: *{query}*")
    st.markdown(f"**{len(results)} Ergebnisse gefunden**")
    
    # Results sorting
    col1, col2 = st.columns(2)
    
    with col1:
        sort_by = st.selectbox(
            "Sortieren nach",
            ["Ähnlichkeit", "Dauer", "Tempo", "Datum"],
            key="results_sort"
        )
    
    with col2:
        sort_order = st.selectbox(
            "Reihenfolge",
            ["Absteigend", "Aufsteigend"],
            key="results_order"
        )
    
    # Sort results
    sorted_results = sort_search_results(results, sort_by, sort_order)
    
    # Display results
    for i, result in enumerate(sorted_results):
        show_search_result_card(result, i)

def show_search_result_card(result: Dict[str, Any], index: int):
    """Zeigt eine einzelne Suchergebnis-Karte an"""
    
    with st.container():
        # Result card HTML
        similarity_color = "#00ff00" if result['similarity_score'] > 0.8 else "#ffff00" if result['similarity_score'] > 0.6 else "#ff6600"
        
        card_html = f"""
        <div class="search-result">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                <h4 style="margin: 0; color: {settings.primary_color};">{result['filename']}</h4>
                <div style="background: {similarity_color}; color: black; padding: 0.2rem 0.5rem; border-radius: 5px; font-weight: bold;">
                    {result['similarity_score']:.1%}
                </div>
            </div>
        </div>
        """
        
        st.markdown(card_html, unsafe_allow_html=True)
        
        # Result details
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Dauer", f"{result['duration']:.1f}s")
        
        with col2:
            st.metric("Tempo", f"{result.get('tempo', 0):.0f} BPM")
        
        with col3:
            st.metric("Spektral-Zentroid", f"{result.get('spectral_centroid', 0):.0f} Hz")
        
        with col4:
            if st.button("▶️ Abspielen", key=f"play_{index}"):
                show_audio_player(result['file_path'], result['filename'])
        
        st.markdown("---")

def sort_search_results(results: list, sort_by: str, sort_order: str) -> list:
    """Sortiert Suchergebnisse"""
    
    reverse = sort_order == "Absteigend"
    
    if sort_by == "Ähnlichkeit":
        return sorted(results, key=lambda x: x['similarity_score'], reverse=reverse)
    elif sort_by == "Dauer":
        return sorted(results, key=lambda x: x['duration'], reverse=reverse)
    elif sort_by == "Tempo":
        return sorted(results, key=lambda x: x.get('tempo', 0), reverse=reverse)
    elif sort_by == "Datum":
        return sorted(results, key=lambda x: x.get('created_at', ''), reverse=reverse)
    
    return results

def show_search_history_sidebar():
    """Zeigt Such-Verlauf in der Sidebar an"""
    
    history = get_search_history()
    
    if history:
        for i, entry in enumerate(history[-5:]):  # Letzte 5 Einträge
            with st.expander(f"{entry['type'].title()}: {entry['query'][:20]}..."):
                st.write(f"**Typ:** {entry['type'].title()}")
                st.write(f"**Ergebnisse:** {entry['results_count']}")
                if st.button("🔄 Wiederholen", key=f"repeat_{i}"):
                    if entry['type'] == 'text':
                        st.session_state.text_search_input = entry['query']
                    st.rerun()
    else:
        st.info("Noch keine Suchen durchgeführt.")
```

Diese Frontend-Spezifikation bietet eine moderne, benutzerfreundliche Oberfläche mit futuristischem Design, umfassender Funktionalität und optimaler User Experience für die Neuromorphe Traum-Engine.