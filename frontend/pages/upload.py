import streamlit as st
import requests
import os
import time
from typing import List, Optional, Dict, Any
import pandas as pd
from pathlib import Path

# Page configuration
backend_url = st.session_state.get('backend_url', 'http://localhost:8000')
max_file_size = st.session_state.get('max_file_size_mb', 50) * 1024 * 1024  # Convert to bytes
supported_formats = st.session_state.get('supported_formats', ['wav', 'mp3', 'flac', 'ogg'])

def check_backend_connection() -> bool:
    """Check if backend is reachable"""
    try:
        response = requests.get(f"{backend_url}/health", timeout=5)
        return response.status_code == 200
    except Exception:
        return False

def upload_audio_file(file_data: bytes, filename: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
    """Upload audio file to backend"""
    try:
        files = {'file': (filename, file_data, 'audio/wav')}
        data = metadata or {}
        
        response = requests.post(
            f"{backend_url}/api/v1/audio/upload",
            files=files,
            data=data,
            timeout=60
        )
        
        if response.status_code == 200:
            return {
                'success': True,
                'data': response.json(),
                'message': 'Upload successful'
            }
        else:
            return {
                'success': False,
                'data': None,
                'message': f'Upload failed: {response.status_code}'
            }
    except Exception as e:
        return {
            'success': False,
            'data': None,
            'message': f'Upload error: {str(e)}'
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

def render_upload_interface():
    """Render the main upload interface"""
    st.header("🎵 Dateien hochladen")
    
    # Upload method selection
    upload_method = st.radio(
        "Upload-Methode wählen:",
        ["📁 Einzelne Dateien", "📂 Ordner hochladen", "🔗 URL-Import"],
        horizontal=True
    )
    
    if upload_method == "📁 Einzelne Dateien":
        render_file_upload()
    elif upload_method == "📂 Ordner hochladen":
        render_folder_upload()
    else:
        render_url_import()

def render_file_upload():
    """Render single file upload interface"""
    st.subheader("📁 Einzelne Dateien hochladen")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Audio-Dateien auswählen",
        type=supported_formats,
        accept_multiple_files=True,
        help=f"Unterstützte Formate: {', '.join(supported_formats)}. Max. Größe: {max_file_size // (1024*1024)}MB pro Datei"
    )
    
    if uploaded_files:
        st.write(f"📊 {len(uploaded_files)} Datei(en) ausgewählt")
        
        # Show file details
        with st.expander("📋 Datei-Details", expanded=True):
            file_data = []
            total_size = 0
            
            for file in uploaded_files:
                file_size = len(file.getvalue())
                total_size += file_size
                
                file_data.append({
                    "Dateiname": file.name,
                    "Größe": f"{file_size / (1024*1024):.2f} MB",
                    "Format": file.name.split('.')[-1].upper(),
                    "Status": "✅ Bereit" if file_size <= max_file_size else "❌ Zu groß"
                })
            
            df = pd.DataFrame(file_data)
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            st.info(f"📊 Gesamtgröße: {total_size / (1024*1024):.2f} MB")
        
        # Metadata input
        with st.expander("🏷️ Metadaten (optional)"):
            col1, col2 = st.columns(2)
            
            with col1:
                category = st.selectbox(
                    "Kategorie",
                    ["Kick", "Snare", "Hi-Hat", "Percussion", "Bass", "Lead", "Pad", "FX", "Vocal", "Other"],
                    index=9
                )
                
                bpm = st.number_input(
                    "BPM (optional)",
                    min_value=60,
                    max_value=200,
                    value=120,
                    step=1
                )
            
            with col2:
                key = st.selectbox(
                    "Tonart (optional)",
                    ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B", "Unknown"],
                    index=12
                )
                
                tags = st.text_input(
                    "Tags (kommagetrennt)",
                    placeholder="dark, techno, industrial, ..."
                )
            
            description = st.text_area(
                "Beschreibung (optional)",
                placeholder="Beschreiben Sie den Sound, die Stimmung oder den Kontext..."
            )
        
        # Upload settings
        with st.expander("⚙️ Upload-Einstellungen"):
            col1, col2 = st.columns(2)
            
            with col1:
                auto_process = st.checkbox(
                    "Automatische Verarbeitung",
                    value=True,
                    help="Generiere automatisch Embeddings nach dem Upload"
                )
                
                overwrite_existing = st.checkbox(
                    "Bestehende Dateien überschreiben",
                    value=False,
                    help="Überschreibe Dateien mit gleichem Namen"
                )
            
            with col2:
                normalize_audio = st.checkbox(
                    "Audio normalisieren",
                    value=True,
                    help="Automatische Lautstärke-Normalisierung"
                )
                
                generate_preview = st.checkbox(
                    "Vorschau generieren",
                    value=True,
                    help="Erstelle Vorschau-Snippets für schnelle Wiedergabe"
                )
        
        # Upload button
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            if st.button("🚀 Upload starten", type="primary", use_container_width=True):
                process_file_uploads(
                    uploaded_files,
                    {
                        'category': category,
                        'bpm': bpm if bpm != 120 else None,
                        'key': key if key != "Unknown" else None,
                        'tags': tags,
                        'description': description,
                        'auto_process': auto_process,
                        'overwrite_existing': overwrite_existing,
                        'normalize_audio': normalize_audio,
                        'generate_preview': generate_preview
                    }
                )

def render_folder_upload():
    """Render folder upload interface"""
    st.subheader("📂 Ordner hochladen")
    
    st.info("💡 **Hinweis:** Aufgrund von Browser-Beschränkungen ist der direkte Ordner-Upload nicht verfügbar. Bitte verwenden Sie die Einzeldatei-Upload-Funktion oder die URL-Import-Option.")
    
    # Alternative: Batch file upload
    st.write("**Alternative: Batch-Upload**")
    
    uploaded_files = st.file_uploader(
        "Mehrere Dateien aus einem Ordner auswählen",
        type=supported_formats,
        accept_multiple_files=True,
        help="Wählen Sie alle Dateien aus Ihrem Ordner aus (Strg+A oder Cmd+A)"
    )
    
    if uploaded_files:
        st.success(f"📁 {len(uploaded_files)} Dateien aus Ordner ausgewählt")
        
        # Batch metadata
        with st.expander("🏷️ Batch-Metadaten"):
            col1, col2 = st.columns(2)
            
            with col1:
                batch_category = st.selectbox(
                    "Standard-Kategorie für alle Dateien",
                    ["Kick", "Snare", "Hi-Hat", "Percussion", "Bass", "Lead", "Pad", "FX", "Vocal", "Other"],
                    index=9
                )
                
                batch_bpm = st.number_input(
                    "Standard-BPM für alle Dateien",
                    min_value=60,
                    max_value=200,
                    value=120
                )
            
            with col2:
                batch_key = st.selectbox(
                    "Standard-Tonart für alle Dateien",
                    ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B", "Unknown"],
                    index=12
                )
                
                batch_tags = st.text_input(
                    "Standard-Tags für alle Dateien",
                    placeholder="batch, folder, collection, ..."
                )
        
        if st.button("🚀 Batch-Upload starten", type="primary"):
            process_batch_upload(
                uploaded_files,
                {
                    'category': batch_category,
                    'bpm': batch_bpm if batch_bpm != 120 else None,
                    'key': batch_key if batch_key != "Unknown" else None,
                    'tags': batch_tags
                }
            )

def render_url_import():
    """Render URL import interface"""
    st.subheader("🔗 URL-Import")
    
    st.info("💡 Importieren Sie Audio-Dateien direkt von URLs (z.B. SoundCloud, Bandcamp, etc.)")
    
    # URL input
    url_input = st.text_area(
        "URLs eingeben (eine pro Zeile)",
        placeholder="https://example.com/audio1.wav\nhttps://example.com/audio2.mp3\n...",
        height=100
    )
    
    if url_input:
        urls = [url.strip() for url in url_input.split('\n') if url.strip()]
        st.write(f"📊 {len(urls)} URL(s) eingegeben")
        
        # URL validation
        valid_urls = []
        invalid_urls = []
        
        for url in urls:
            if url.startswith(('http://', 'https://')) and any(url.endswith(f'.{fmt}') for fmt in supported_formats):
                valid_urls.append(url)
            else:
                invalid_urls.append(url)
        
        if valid_urls:
            st.success(f"✅ {len(valid_urls)} gültige URL(s)")
        
        if invalid_urls:
            st.error(f"❌ {len(invalid_urls)} ungültige URL(s)")
            with st.expander("Ungültige URLs anzeigen"):
                for url in invalid_urls:
                    st.write(f"- {url}")
        
        # Import settings
        if valid_urls:
            with st.expander("⚙️ Import-Einstellungen"):
                col1, col2 = st.columns(2)
                
                with col1:
                    download_timeout = st.number_input(
                        "Download-Timeout (Sekunden)",
                        min_value=10,
                        max_value=300,
                        value=60
                    )
                    
                    verify_ssl = st.checkbox(
                        "SSL-Zertifikate überprüfen",
                        value=True
                    )
                
                with col2:
                    max_file_size_url = st.number_input(
                        "Max. Dateigröße (MB)",
                        min_value=1,
                        max_value=100,
                        value=50
                    )
                    
                    auto_rename = st.checkbox(
                        "Automatische Umbenennung bei Konflikten",
                        value=True
                    )
            
            if st.button("🌐 URL-Import starten", type="primary"):
                process_url_import(
                    valid_urls,
                    {
                        'timeout': download_timeout,
                        'verify_ssl': verify_ssl,
                        'max_size': max_file_size_url * 1024 * 1024,
                        'auto_rename': auto_rename
                    }
                )

def process_file_uploads(files: List, metadata: Dict[str, Any]):
    """Process uploaded files"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    successful_uploads = []
    failed_uploads = []
    
    for i, file in enumerate(files):
        status_text.text(f"Uploading {file.name}...")
        
        # Check file size
        if len(file.getvalue()) > max_file_size:
            failed_uploads.append({
                'filename': file.name,
                'error': 'File too large'
            })
            continue
        
        # Prepare metadata for this file
        file_metadata = metadata.copy()
        file_metadata['filename'] = file.name
        file_metadata['size'] = len(file.getvalue())
        
        # Upload file
        result = upload_audio_file(file.getvalue(), file.name, file_metadata)
        
        if result['success']:
            successful_uploads.append({
                'filename': file.name,
                'data': result['data']
            })
        else:
            failed_uploads.append({
                'filename': file.name,
                'error': result['message']
            })
        
        # Update progress
        progress_bar.progress((i + 1) / len(files))
    
    # Show results
    status_text.empty()
    progress_bar.empty()
    
    if successful_uploads:
        st.success(f"✅ {len(successful_uploads)} Datei(en) erfolgreich hochgeladen!")
        
        # Add to upload history
        if 'upload_history' not in st.session_state:
            st.session_state.upload_history = []
        
        for upload in successful_uploads:
            st.session_state.upload_history.append({
                'filename': upload['filename'],
                'timestamp': time.time(),
                'status': 'success',
                'data': upload['data']
            })
    
    if failed_uploads:
        st.error(f"❌ {len(failed_uploads)} Datei(en) konnten nicht hochgeladen werden")
        
        with st.expander("Fehler-Details"):
            for failure in failed_uploads:
                st.write(f"**{failure['filename']}**: {failure['error']}")

def process_batch_upload(files: List, metadata: Dict[str, Any]):
    """Process batch upload"""
    st.info("🔄 Batch-Upload wird verarbeitet...")
    
    # Use the same logic as single file upload but with batch metadata
    process_file_uploads(files, metadata)

def process_url_import(urls: List[str], settings: Dict[str, Any]):
    """Process URL import"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    successful_imports = []
    failed_imports = []
    
    for i, url in enumerate(urls):
        status_text.text(f"Importing from {url}...")
        
        try:
            # Download file
            response = requests.get(
                url,
                timeout=settings['timeout'],
                verify=settings['verify_ssl'],
                stream=True
            )
            
            if response.status_code == 200:
                # Check file size
                content_length = response.headers.get('content-length')
                if content_length and int(content_length) > settings['max_size']:
                    failed_imports.append({
                        'url': url,
                        'error': 'File too large'
                    })
                    continue
                
                # Get filename from URL
                filename = url.split('/')[-1]
                if not filename or '.' not in filename:
                    filename = f"imported_audio_{int(time.time())}.wav"
                
                # Upload to backend
                file_data = response.content
                result = upload_audio_file(file_data, filename, {'source': 'url_import', 'original_url': url})
                
                if result['success']:
                    successful_imports.append({
                        'url': url,
                        'filename': filename,
                        'data': result['data']
                    })
                else:
                    failed_imports.append({
                        'url': url,
                        'error': result['message']
                    })
            else:
                failed_imports.append({
                    'url': url,
                    'error': f'HTTP {response.status_code}'
                })
        
        except Exception as e:
            failed_imports.append({
                'url': url,
                'error': str(e)
            })
        
        # Update progress
        progress_bar.progress((i + 1) / len(urls))
    
    # Show results
    status_text.empty()
    progress_bar.empty()
    
    if successful_imports:
        st.success(f"✅ {len(successful_imports)} Datei(en) erfolgreich importiert!")
    
    if failed_imports:
        st.error(f"❌ {len(failed_imports)} Import(s) fehlgeschlagen")
        
        with st.expander("Import-Fehler"):
            for failure in failed_imports:
                st.write(f"**{failure['url']}**: {failure['error']}")

def render_upload_history():
    """Render upload history"""
    st.header("📚 Upload-Verlauf")
    
    if 'upload_history' not in st.session_state or not st.session_state.upload_history:
        st.info("📭 Noch keine Uploads vorhanden")
        return
    
    # Filter and sort history
    history = st.session_state.upload_history.copy()
    history.sort(key=lambda x: x['timestamp'], reverse=True)
    
    # Show recent uploads
    recent_uploads = history[:10]  # Show last 10 uploads
    
    for upload in recent_uploads:
        with st.container():
            col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
            
            with col1:
                st.write(f"**{upload['filename']}**")
            
            with col2:
                timestamp = time.strftime("%Y-%m-%d %H:%M", time.localtime(upload['timestamp']))
                st.write(timestamp)
            
            with col3:
                if upload['status'] == 'success':
                    st.success("✅ Erfolgreich")
                else:
                    st.error("❌ Fehlgeschlagen")
            
            with col4:
                if st.button("🗑️", key=f"delete_{upload['timestamp']}", help="Aus Verlauf entfernen"):
                    st.session_state.upload_history.remove(upload)
                    st.rerun()
    
    # Clear history button
    if st.button("🗑️ Verlauf löschen"):
        st.session_state.upload_history = []
        st.rerun()

def render_processing_status():
    """Render processing status"""
    st.header("⚡ Verarbeitungsstatus")
    
    # Get backend stats
    stats = get_backend_stats()
    
    if stats:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_files = stats.get('total_files', 0)
            st.metric("Gesamt Dateien", total_files)
        
        with col2:
            processing_queue = stats.get('processing_queue', 0)
            st.metric("In Warteschlange", processing_queue)
        
        with col3:
            processed_today = stats.get('processed_today', 0)
            st.metric("Heute verarbeitet", processed_today)
        
        with col4:
            avg_processing_time = stats.get('avg_processing_time', 0)
            st.metric("Ø Verarbeitungszeit", f"{avg_processing_time:.1f}s")
        
        # Processing queue details
        if processing_queue > 0:
            st.subheader("🔄 Aktuelle Warteschlange")
            
            queue_items = stats.get('queue_items', [])
            if queue_items:
                for item in queue_items[:5]:  # Show first 5 items
                    col1, col2, col3 = st.columns([3, 2, 2])
                    
                    with col1:
                        st.write(item.get('filename', 'Unknown'))
                    
                    with col2:
                        status = item.get('status', 'pending')
                        if status == 'processing':
                            st.info("🔄 Verarbeitung...")
                        elif status == 'pending':
                            st.warning("⏳ Wartend")
                        else:
                            st.write(status)
                    
                    with col3:
                        progress = item.get('progress', 0)
                        st.progress(progress / 100 if progress else 0)
    else:
        st.info("📊 Keine Verarbeitungsstatistiken verfügbar")

# Main page content
st.title("📤 Audio Upload")
st.markdown("Upload your audio stems to integrate them into the Neuromorphic Dream Engine.")

# Check backend connection
if not check_backend_connection():
    st.error("❌ Backend nicht erreichbar. Bitte überprüfen Sie die Verbindung in den Einstellungen.")
    st.stop()

# Upload interface
render_upload_interface()

# Upload history
render_upload_history()

# Processing status
render_processing_status()