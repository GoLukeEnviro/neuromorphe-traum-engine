import streamlit as st
import requests
import time
from typing import Optional


class AudioUploadPage:
    """Audio upload and processing page"""
    
    def __init__(self):
        self.backend_url = None
    
    def render(self):
        """Render the audio upload page"""
        self.backend_url = st.session_state.get('backend_url', 'http://localhost:8000')
        
        st.title("🎵 Audio Upload & Processing")
        st.markdown("Upload audio files to generate CLAP embeddings for semantic search.")
        
        # Upload section
        self._render_upload_section()
        
        # Processing status
        self._render_processing_status()
        
        # Recent uploads
        self._render_recent_uploads()
    
    def _render_upload_section(self):
        """Render the file upload section"""
        st.header("📁 Upload Audio Files")
        
        with st.form("audio_upload_form", clear_on_submit=True):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                uploaded_files = st.file_uploader(
                    "Choose audio files",
                    type=['wav', 'mp3', 'flac', 'ogg'],
                    accept_multiple_files=True,
                    help="Supported formats: WAV, MP3, FLAC, OGG (Max 50MB per file)"
                )
            
            with col2:
                category = st.selectbox(
                    "Category (Optional)",
                    options=["", "techno", "house", "ambient", "drum_and_bass", "trance", "other"],
                    help="Categorize your audio for better organization"
                )
                
                bpm = st.number_input(
                    "BPM (Optional)",
                    min_value=60,
                    max_value=200,
                    value=None,
                    help="Beats per minute for rhythm-based filtering"
                )
            
            submitted = st.form_submit_button(
                "🚀 Upload & Process",
                type="primary",
                use_container_width=True
            )
            
            if submitted and uploaded_files:
                self._process_uploads(uploaded_files, category, bpm)
    
    def _process_uploads(self, uploaded_files, category: str, bpm: Optional[int]):
        """Process uploaded files"""
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total_files = len(uploaded_files)
        successful_uploads = []
        failed_uploads = []
        
        for i, uploaded_file in enumerate(uploaded_files):
            try:
                status_text.text(f"Processing {uploaded_file.name}...")
                progress_bar.progress((i + 1) / total_files)
                
                # Prepare form data
                files = {'file': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                data = {}
                
                if category:
                    data['category'] = category
                if bpm:
                    data['bpm'] = bpm
                
                # Upload to backend
                response = requests.post(
                    f"{self.backend_url}/api/v1/audio/upload",
                    files=files,
                    data=data,
                    timeout=120  # 2 minutes timeout for processing
                )
                
                if response.status_code == 200:
                    result = response.json()
                    successful_uploads.append({
                        'filename': uploaded_file.name,
                        'file_id': result.get('id'),
                        'status': result.get('status'),
                        'message': result.get('message')
                    })
                else:
                    failed_uploads.append({
                        'filename': uploaded_file.name,
                        'error': response.text
                    })
                    
            except Exception as e:
                failed_uploads.append({
                    'filename': uploaded_file.name,
                    'error': str(e)
                })
        
        # Update session state
        if 'uploaded_files' not in st.session_state:
            st.session_state.uploaded_files = []
        st.session_state.uploaded_files.extend(successful_uploads)
        
        # Show results
        progress_bar.empty()
        status_text.empty()
        
        if successful_uploads:
            st.success(f"✅ Successfully processed {len(successful_uploads)} files!")
            
            with st.expander("📋 Processing Details", expanded=True):
                for upload in successful_uploads:
                    st.write(f"**{upload['filename']}**")
                    st.write(f"- File ID: `{upload['file_id']}`")
                    st.write(f"- Status: {upload['status']}")
                    st.write(f"- Message: {upload['message']}")
                    st.write("---")
        
        if failed_uploads:
            st.error(f"❌ Failed to process {len(failed_uploads)} files")
            
            with st.expander("⚠️ Error Details"):
                for upload in failed_uploads:
                    st.write(f"**{upload['filename']}**: {upload['error']}")
    
    def _render_processing_status(self):
        """Render processing status section"""
        st.header("⚡ Processing Status")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Get Audio Stats", use_container_width=True):
                self._fetch_audio_stats()
        
        with col2:
            if st.button("📁 List Audio Files", use_container_width=True):
                self._fetch_audio_files()
        
        with col3:
            if st.button("🔄 Refresh Status", use_container_width=True):
                st.rerun()
    
    def _fetch_audio_stats(self):
        """Fetch and display audio statistics"""
        try:
            response = requests.get(f"{self.backend_url}/api/v1/search/stats", timeout=10)
            if response.status_code == 200:
                stats = response.json()
                st.session_state.stats = stats
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Files", stats.get('total_files', 0))
                with col2:
                    st.metric("Total Embeddings", stats.get('total_embeddings', 0))
                with col3:
                    categories = stats.get('categories', [])
                    st.metric("Categories", len(categories))
                
                if categories:
                    st.subheader("📊 Category Breakdown")
                    for cat in categories:
                        st.write(f"**{cat['category']}**: {cat['count']} files")
                        if cat.get('avg_bpm'):
                            st.write(f"  - Avg BPM: {cat['avg_bpm']:.1f}")
                        if cat.get('avg_duration'):
                            st.write(f"  - Avg Duration: {cat['avg_duration']:.1f}s")
            else:
                st.error("Failed to fetch statistics")
        except Exception as e:
            st.error(f"Error fetching statistics: {str(e)}")
    
    def _fetch_audio_files(self):
        """Fetch and display audio files list"""
        try:
            response = requests.get(f"{self.backend_url}/api/v1/audio/files", timeout=10)
            if response.status_code == 200:
                file_ids = response.json()
                
                if file_ids:
                    st.subheader(f"📁 Audio Files ({len(file_ids)} total)")
                    
                    # Show first 10 files with details
                    for file_id in file_ids[:10]:
                        with st.expander(f"File: {file_id}"):
                            self._show_file_details(file_id)
                    
                    if len(file_ids) > 10:
                        st.info(f"Showing first 10 of {len(file_ids)} files")
                else:
                    st.info("No audio files found")
            else:
                st.error("Failed to fetch audio files")
        except Exception as e:
            st.error(f"Error fetching audio files: {str(e)}")
    
    def _show_file_details(self, file_id: str):
        """Show details for a specific file"""
        try:
            response = requests.get(
                f"{self.backend_url}/api/v1/audio/files/{file_id}", 
                timeout=5
            )
            if response.status_code == 200:
                info = response.json()
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Filename**: {info.get('filename', 'N/A')}")
                    st.write(f"**Category**: {info.get('category', 'N/A')}")
                    st.write(f"**BPM**: {info.get('bpm', 'N/A')}")
                
                with col2:
                    st.write(f"**Duration**: {info.get('duration', 'N/A')}s")
                    st.write(f"**Sample Rate**: {info.get('sample_rate', 'N/A')} Hz")
                    st.write(f"**Has Embedding**: {'✅' if info.get('has_embedding') else '❌'}")
                
                if not info.get('has_embedding'):
                    if st.button(f"🔄 Process {file_id}", key=f"process_{file_id}"):
                        self._process_single_file(file_id)
            else:
                st.error(f"Failed to get file details: {response.text}")
        except Exception as e:
            st.error(f"Error getting file details: {str(e)}")
    
    def _process_single_file(self, file_id: str):
        """Process a single file to generate embedding"""
        try:
            with st.spinner(f"Processing {file_id}..."):
                response = requests.post(
                    f"{self.backend_url}/api/v1/audio/files/{file_id}/process",
                    timeout=120
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if result.get('status') == 'completed':
                        st.success(f"✅ Successfully processed {file_id}")
                    else:
                        st.warning(f"⚠️ Processing status: {result.get('status')}")
                        st.write(f"Message: {result.get('message')}")
                else:
                    st.error(f"Failed to process file: {response.text}")
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    
    def _render_recent_uploads(self):
        """Render recent uploads section"""
        if st.session_state.get('uploaded_files'):
            st.header("📋 Recent Uploads")
            
            recent_files = st.session_state.uploaded_files[-5:]  # Show last 5
            
            for upload in reversed(recent_files):
                with st.container():
                    col1, col2, col3 = st.columns([2, 1, 1])
                    
                    with col1:
                        st.write(f"**{upload['filename']}**")
                    
                    with col2:
                        status_color = "🟢" if upload['status'] == 'completed' else "🟡"
                        st.write(f"{status_color} {upload['status']}")
                    
                    with col3:
                        st.code(upload['file_id'])
                    
                    st.write("---")