import streamlit as st
import requests
import os
import json
import time
from typing import Dict, Any, Optional

# Page configuration
backend_url = st.session_state.get('backend_url', 'http://localhost:8508')
default_settings = {
    'backend_url': 'http://localhost:8000',
    'connection_timeout': 30,
    'max_results': 50,
    'similarity_threshold': 0.1,
    'audio_format': 'wav',
    'sample_rate': 44100,
    'auto_play': False,
    'theme': 'dark'
}

def test_backend_connection(url: str, timeout: int = 30) -> Dict[str, Any]:
    """Test backend connection"""
    try:
        response = requests.get(f"{url}/health", timeout=timeout)
        if response.status_code == 200:
            return {
                'status': 'success',
                'message': 'Backend connection successful',
                'data': response.json()
            }
        else:
            return {
                'status': 'error',
                'message': f'Backend returned status {response.status_code}',
                'data': None
            }
    except Exception as e:
        return {
            'status': 'error',
            'message': f'Connection failed: {str(e)}',
            'data': None
        }

def get_backend_stats() -> Dict[str, Any]:
    """Get backend statistics"""
    try:
        response = requests.get(f"{backend_url}/api/v1/stats", timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            return {}
    except Exception:
        return {}

def get_api_info() -> Dict[str, Any]:
    """Get API information"""
    try:
        response = requests.get(f"{backend_url}/api/v1/info", timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            return {}
    except Exception:
        return {}

def render_backend_settings():
    """Render backend connection settings"""
    st.header("🔗 Backend-Verbindung")
    
    # Current settings
    current_url = st.session_state.get('backend_url', default_settings['backend_url'])
    current_timeout = st.session_state.get('connection_timeout', default_settings['connection_timeout'])
    current_api_key = st.session_state.get('api_key', '')
    
    # Settings form
    with st.form("backend_settings"):
        st.subheader("Verbindungseinstellungen")
        
        new_url = st.text_input(
            "Backend URL",
            value=current_url,
            help="URL des Backend-Servers (z.B. http://localhost:8000)"
        )
        
        new_timeout = st.number_input(
            "Verbindungs-Timeout (Sekunden)",
            min_value=5,
            max_value=120,
            value=current_timeout,
            help="Maximale Wartezeit für Backend-Anfragen"
        )
        
        new_api_key = st.text_input(
            "API Key (optional)",
            value=current_api_key,
            type="password",
            help="API-Schlüssel für authentifizierte Anfragen"
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.form_submit_button("💾 Einstellungen speichern", type="primary"):
                save_backend_settings(new_url, new_timeout, new_api_key)
        
        with col2:
            if st.form_submit_button("🔍 Verbindung testen"):
                test_connection_ui(new_url, new_timeout)
    
    # Connection status
    show_connection_status()
    
    # Endpoints overview
    render_endpoints_overview()

def show_connection_status():
    """Show current connection status"""
    st.subheader("📊 Verbindungsstatus")
    
    current_url = st.session_state.get('backend_url', default_settings['backend_url'])
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Status prüfen", use_container_width=True):
            check_backend_health()
    
    with col2:
        st.metric("Backend URL", current_url)
    
    with col3:
        last_check = st.session_state.get('last_backend_check', 'Nie')
        st.metric("Letzte Prüfung", last_check)

def save_backend_settings(url: str, timeout: int, api_key: str):
    """Save backend settings"""
    st.session_state.backend_url = url.rstrip('/')
    st.session_state.connection_timeout = timeout
    st.session_state.api_key = api_key
    
    st.success("✅ Einstellungen gespeichert!")
    st.rerun()

def test_connection_ui(url: str, timeout: int):
    """Test backend connection with UI feedback"""
    with st.spinner("Teste Verbindung..."):
        result = test_backend_connection(url, timeout)
        
        if result['status'] == 'success':
            st.success(f"✅ {result['message']}")
            if result['data']:
                st.json(result['data'])
        else:
            st.error(f"❌ {result['message']}")

def render_endpoints_overview():
    """Render available endpoints overview"""
    st.subheader("🛠️ Verfügbare Endpunkte")
    
    endpoints = [
        {"method": "GET", "path": "/health", "description": "Backend-Gesundheitsstatus"},
        {"method": "GET", "path": "/api/v1/audio/files", "description": "Liste aller Audio-Dateien"},
        {"method": "POST", "path": "/api/v1/audio/upload", "description": "Audio-Datei hochladen"},
        {"method": "POST", "path": "/api/v1/search/text", "description": "Textbasierte Suche"},
        {"method": "POST", "path": "/api/v1/search/similar", "description": "Ähnlichkeitssuche"},
        {"method": "GET", "path": "/api/v1/stats", "description": "Backend-Statistiken"}
    ]
    
    for endpoint in endpoints:
        col1, col2, col3 = st.columns([1, 2, 4])
        
        with col1:
            method_color = {
                "GET": "🟢",
                "POST": "🔵",
                "PUT": "🟡",
                "DELETE": "🔴"
            }.get(endpoint["method"], "⚪")
            st.write(f"{method_color} {endpoint['method']}")
        
        with col2:
            st.code(endpoint["path"])
        
        with col3:
            st.write(endpoint["description"])

def render_audio_settings():
    """Render audio processing settings"""
    st.header("🎵 Audio-Einstellungen")
    
    # Current settings
    current_format = st.session_state.get('audio_format', default_settings['audio_format'])
    current_sample_rate = st.session_state.get('sample_rate', default_settings['sample_rate'])
    current_auto_play = st.session_state.get('auto_play', default_settings['auto_play'])
    
    with st.form("audio_settings"):
        st.subheader("Audio-Verarbeitung")
        
        # Audio format
        audio_format = st.selectbox(
            "Bevorzugtes Audio-Format",
            options=['wav', 'mp3', 'flac', 'ogg'],
            index=['wav', 'mp3', 'flac', 'ogg'].index(current_format),
            help="Format für Audio-Export und -verarbeitung"
        )
        
        # Sample rate
        sample_rate = st.selectbox(
            "Sample Rate (Hz)",
            options=[22050, 44100, 48000, 96000],
            index=[22050, 44100, 48000, 96000].index(current_sample_rate),
            help="Abtastrate für Audio-Verarbeitung"
        )
        
        # Auto-play
        auto_play = st.checkbox(
            "Automatische Wiedergabe",
            value=current_auto_play,
            help="Automatisches Abspielen von Suchergebnissen"
        )
        
        # Quality settings
        st.subheader("Qualitätseinstellungen")
        
        bit_depth = st.selectbox(
            "Bit-Tiefe",
            options=[16, 24, 32],
            index=1,
            help="Bit-Tiefe für Audio-Verarbeitung"
        )
        
        # Processing settings
        st.subheader("Verarbeitungseinstellungen")
        
        normalize_audio = st.checkbox(
            "Audio normalisieren",
            value=True,
            help="Automatische Lautstärke-Normalisierung"
        )
        
        remove_silence = st.checkbox(
            "Stille entfernen",
            value=False,
            help="Automatisches Entfernen von Stille am Anfang/Ende"
        )
        
        # Advanced settings
        with st.expander("🔧 Erweiterte Einstellungen"):
            chunk_size = st.number_input(
                "Chunk-Größe (Samples)",
                min_value=512,
                max_value=8192,
                value=2048,
                step=512,
                help="Größe der Audio-Chunks für Verarbeitung"
            )
            
            overlap = st.slider(
                "Overlap (%)",
                min_value=0,
                max_value=50,
                value=25,
                help="Überlappung zwischen Audio-Chunks"
            )
            
            window_function = st.selectbox(
                "Fensterfunktion",
                options=['hann', 'hamming', 'blackman', 'bartlett'],
                help="Fensterfunktion für Spektralanalyse"
            )
        
        if st.form_submit_button("💾 Audio-Einstellungen speichern", type="primary"):
            # Save audio settings
            st.session_state.audio_format = audio_format
            st.session_state.sample_rate = sample_rate
            st.session_state.auto_play = auto_play
            st.session_state.bit_depth = bit_depth
            st.session_state.normalize_audio = normalize_audio
            st.session_state.remove_silence = remove_silence
            st.session_state.chunk_size = chunk_size
            st.session_state.overlap = overlap
            st.session_state.window_function = window_function
            
            st.success("✅ Audio-Einstellungen gespeichert!")
            st.rerun()

def render_search_settings():
    """Render search and similarity settings"""
    st.header("🔍 Such-Einstellungen")
    
    # Current settings
    current_max_results = st.session_state.get('max_results', default_settings['max_results'])
    current_threshold = st.session_state.get('similarity_threshold', default_settings['similarity_threshold'])
    
    with st.form("search_settings"):
        st.subheader("Allgemeine Sucheinstellungen")
        
        max_results = st.number_input(
            "Maximale Anzahl Ergebnisse",
            min_value=10,
            max_value=500,
            value=current_max_results,
            step=10,
            help="Maximale Anzahl der angezeigten Suchergebnisse"
        )
        
        similarity_threshold = st.slider(
            "Ähnlichkeits-Schwellenwert",
            min_value=0.0,
            max_value=1.0,
            value=current_threshold,
            step=0.01,
            help="Minimale Ähnlichkeit für Suchergebnisse (0.0 = alle, 1.0 = nur identische)"
        )
        
        # Search behavior
        st.subheader("Suchverhalten")
        
        fuzzy_search = st.checkbox(
            "Fuzzy-Suche aktivieren",
            value=True,
            help="Toleriert Tippfehler in der Textsuche"
        )
        
        case_sensitive = st.checkbox(
            "Groß-/Kleinschreibung beachten",
            value=False,
            help="Unterscheidet zwischen Groß- und Kleinbuchstaben"
        )
        
        # Advanced search settings
        with st.expander("🔧 Erweiterte Sucheinstellungen"):
            search_timeout = st.number_input(
                "Such-Timeout (Sekunden)",
                min_value=5,
                max_value=120,
                value=30,
                help="Maximale Wartezeit für Suchanfragen"
            )
            
            enable_caching = st.checkbox(
                "Suchergebnis-Caching",
                value=True,
                help="Zwischenspeichern von Suchergebnissen für bessere Performance"
            )
            
            cache_duration = st.number_input(
                "Cache-Dauer (Minuten)",
                min_value=1,
                max_value=60,
                value=10,
                help="Wie lange Suchergebnisse zwischengespeichert werden"
            )
        
        # Embedding settings
        st.subheader("Embedding-Einstellungen")
        
        embedding_model = st.selectbox(
            "Embedding-Modell",
            options=['sentence-transformers', 'openai', 'custom'],
            help="Modell für die Generierung von Text-Embeddings"
        )
        
        embedding_dimensions = st.number_input(
            "Embedding-Dimensionen",
            min_value=128,
            max_value=1536,
            value=384,
            step=64,
            help="Anzahl der Dimensionen für Embeddings"
        )
        
        if st.form_submit_button("💾 Such-Einstellungen speichern", type="primary"):
            # Save search settings
            st.session_state.max_results = max_results
            st.session_state.similarity_threshold = similarity_threshold
            st.session_state.fuzzy_search = fuzzy_search
            st.session_state.case_sensitive = case_sensitive
            st.session_state.search_timeout = search_timeout
            st.session_state.enable_caching = enable_caching
            st.session_state.cache_duration = cache_duration
            st.session_state.embedding_model = embedding_model
            st.session_state.embedding_dimensions = embedding_dimensions
            
            st.success("✅ Such-Einstellungen gespeichert!")
            st.rerun()

def render_system_info():
    """Render system information and diagnostics"""
    st.header("📊 System-Information")
    
    # System status
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 System-Status aktualisieren", use_container_width=True):
            check_backend_health()
    
    with col2:
        if st.button("📊 Statistiken laden", use_container_width=True):
            get_system_stats()
    
    with col3:
        if st.button("🧹 Cache leeren", use_container_width=True):
            clear_cache()
    
    # Backend health
    st.subheader("🏥 Backend-Gesundheit")
    
    health_status = st.session_state.get('backend_health', {})
    
    if health_status:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            status = health_status.get('status', 'unknown')
            color = "🟢" if status == 'healthy' else "🔴" if status == 'unhealthy' else "🟡"
            st.metric("Status", f"{color} {status.title()}")
        
        with col2:
            uptime = health_status.get('uptime', 'N/A')
            st.metric("Uptime", uptime)
        
        with col3:
            version = health_status.get('version', 'N/A')
            st.metric("Version", version)
        
        with col4:
            memory_usage = health_status.get('memory_usage', 'N/A')
            st.metric("Memory", memory_usage)
    
    # System statistics
    st.subheader("📈 Statistiken")
    
    stats = st.session_state.get('system_stats', {})
    
    if stats:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_files = stats.get('total_files', 0)
            st.metric("Gesamt Dateien", total_files)
        
        with col2:
            total_embeddings = stats.get('total_embeddings', 0)
            st.metric("Embeddings", total_embeddings)
        
        with col3:
            avg_processing_time = stats.get('avg_processing_time', 0)
            st.metric("Ø Verarbeitungszeit", f"{avg_processing_time:.2f}s")
        
        with col4:
            storage_used = stats.get('storage_used', 'N/A')
            st.metric("Speicher verwendet", storage_used)
        
        # Detailed stats
        with st.expander("📋 Detaillierte Statistiken"):
            st.json(stats)
    
    # Environment info
    st.subheader("🌍 Umgebungsinformationen")
    
    env_info = {
        "Python Version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
        "Streamlit Version": st.__version__,
        "Working Directory": os.getcwd(),
        "Session State Keys": len(st.session_state.keys()),
        "Backend URL": st.session_state.get('backend_url', 'Not set')
    }
    
    for key, value in env_info.items():
        col1, col2 = st.columns([1, 2])
        with col1:
            st.write(f"**{key}:**")
        with col2:
            st.write(value)

def check_backend_health():
    """Check backend health status"""
    try:
        with st.spinner("Prüfe Backend-Gesundheit..."):
            response = requests.get(f"{backend_url}/health", timeout=10)
            
            if response.status_code == 200:
                health_data = response.json()
                st.session_state.backend_health = health_data
                st.session_state.last_backend_check = time.strftime("%H:%M:%S")
                st.success("✅ Backend ist gesund!")
            else:
                st.error(f"❌ Backend-Gesundheitsprüfung fehlgeschlagen: {response.status_code}")
                
    except Exception as e:
        st.error(f"❌ Verbindungsfehler: {str(e)}")
        st.session_state.backend_health = {'status': 'error', 'message': str(e)}

def get_system_stats():
    """Get system statistics from backend"""
    try:
        with st.spinner("Lade Statistiken..."):
            response = requests.get(f"{backend_url}/api/v1/stats", timeout=10)
            
            if response.status_code == 200:
                stats_data = response.json()
                st.session_state.system_stats = stats_data
                st.success("✅ Statistiken geladen!")
            else:
                st.error(f"❌ Statistiken konnten nicht geladen werden: {response.status_code}")
                
    except Exception as e:
        st.error(f"❌ Fehler beim Laden der Statistiken: {str(e)}")

def clear_cache():
    """Clear application cache"""
    # Clear session state cache items
    cache_keys = ['search_results', 'filtered_results', 'backend_health', 'system_stats']
    
    for key in cache_keys:
        if key in st.session_state:
            del st.session_state[key]
    
    st.success("✅ Cache geleert!")
    st.rerun()

def render_appearance_settings():
    """Render appearance and UI settings"""
    st.header("🎨 Erscheinungsbild")
    
    # Current settings
    current_theme = st.session_state.get('theme', default_settings['theme'])
    
    with st.form("appearance_settings"):
        st.subheader("Design-Einstellungen")
        
        # Theme selection
        theme = st.selectbox(
            "Farbschema",
            options=['light', 'dark', 'auto'],
            index=['light', 'dark', 'auto'].index(current_theme),
            help="Wähle das Farbschema der Anwendung"
        )
        
        # Layout settings
        st.subheader("Layout-Einstellungen")
        
        sidebar_state = st.selectbox(
            "Seitenleiste",
            options=['expanded', 'collapsed', 'auto'],
            help="Standard-Zustand der Seitenleiste"
        )
        
        page_width = st.selectbox(
            "Seitenbreite",
            options=['wide', 'centered'],
            help="Layout der Hauptseite"
        )
        
        # Display settings
        st.subheader("Anzeige-Einstellungen")
        
        show_progress_bars = st.checkbox(
            "Fortschrittsbalken anzeigen",
            value=True,
            help="Zeige Fortschrittsbalken bei längeren Operationen"
        )
        
        show_tooltips = st.checkbox(
            "Tooltips anzeigen",
            value=True,
            help="Zeige Hilfe-Tooltips bei UI-Elementen"
        )
        
        animations_enabled = st.checkbox(
            "Animationen aktivieren",
            value=True,
            help="Aktiviere UI-Animationen und Übergänge"
        )
        
        # Advanced appearance settings
        with st.expander("🔧 Erweiterte Einstellungen"):
            custom_css = st.text_area(
                "Benutzerdefiniertes CSS",
                height=100,
                help="Füge benutzerdefiniertes CSS hinzu"
            )
            
            font_size = st.selectbox(
                "Schriftgröße",
                options=['small', 'medium', 'large'],
                index=1,
                help="Grundschriftgröße der Anwendung"
            )
        
        if st.form_submit_button("💾 Erscheinungsbild speichern", type="primary"):
            # Save appearance settings
            st.session_state.theme = theme
            st.session_state.sidebar_state = sidebar_state
            st.session_state.page_width = page_width
            st.session_state.show_progress_bars = show_progress_bars
            st.session_state.show_tooltips = show_tooltips
            st.session_state.animations_enabled = animations_enabled
            st.session_state.custom_css = custom_css
            st.session_state.font_size = font_size
            
            st.success("✅ Erscheinungsbild-Einstellungen gespeichert!")
            st.rerun()

# Main page content
st.title("⚙️ Settings & Configuration")
st.markdown("Configure the Neuromorphic Dream Engine application settings.")

# Settings tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔗 Backend", 
    "🎵 Audio", 
    "🔍 Search", 
    "📊 System",
    "🎨 Appearance"
])

with tab1:
    render_backend_settings()

with tab2:
    render_audio_settings()

with tab3:
    render_search_settings()

with tab4:
    render_system_info()

with tab5:
    render_appearance_settings()