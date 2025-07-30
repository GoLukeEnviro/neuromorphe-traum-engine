import streamlit as st
import requests
import time
import os
from typing import Optional, List, Dict, Any
import json

# Page configuration
supported_formats = ['wav', 'mp3', 'flac', 'ogg', 'aiff']
max_file_size_mb = 50
backend_url = st.session_state.get('backend_url', 'http://localhost:8508')

def check_backend_connection() -> bool:
    """Check if backend is reachable"""
    try:
        response = requests.get(f"{backend_url}/system/health", timeout=5)
        return response.status_code == 200
    except Exception:
        return False

def process_uploads(uploaded_files, category: str, bpm: Optional[int]):
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
                f"{backend_url}/api/v1/audio/upload",
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
                st.write(f"**{upload['filename']}**")
                st.write(f"Error: {upload['error']}")
                st.write("---")

# Main page content
st.title("🎵 Audio Upload & Processing")
st.markdown(
    "Upload audio files to generate CLAP embeddings for semantic search. "
    "Die Engine analysiert automatisch die Audio-Charakteristika und erstellt semantische Embeddings."
)

# Check backend status first
if not check_backend_connection():
    st.error("❌ Backend nicht erreichbar. Bitte starten Sie den FastAPI-Server.")
    st.code("python -m uvicorn src.main:app --reload")
    st.stop()

# Upload section
st.header("📁 File Upload")

with st.form("upload_form", clear_on_submit=True):
    uploaded_files = st.file_uploader(
        "Choose audio files",
        type=supported_formats,
        accept_multiple_files=True,
        help=f"Supported formats: {', '.join(supported_formats)}. Max size: {max_file_size_mb}MB per file"
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        category_options = ["", "Kick", "Snare", "Hi-Hat", "Percussion", "Bass", "Lead", "Pad", "FX", "Vocal"]
        category_default_index = 0
        if 'category_selected' in st.session_state and st.session_state.category_selected in category_options:
            category_default_index = category_options.index(st.session_state.category_selected)

        category = st.selectbox(
            "Category",
            options=category_options,
            index=category_default_index,
            key="audio_upload_category_selectbox",
            help="Kategorisierung für bessere Organisation"
        )
        st.session_state.category_selected = category
        
        auto_metadata = st.checkbox(
            "Auto-Metadaten",
            value=True,
            help="Automatische Extraktion von BPM, Key, etc."
        )
    
    with col2:
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
        process_uploads(uploaded_files, category, bpm)

# Recent uploads section
st.header("📋 Recent Uploads")

if 'uploaded_files' in st.session_state and st.session_state.uploaded_files:
    st.write(f"Total uploaded files: {len(st.session_state.uploaded_files)}")
    
    # Show recent uploads in a table
    recent_uploads = st.session_state.uploaded_files[-10:]  # Last 10 uploads
    
    for upload in reversed(recent_uploads):
        with st.expander(f"🎵 {upload['filename']}"):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**File ID:** `{upload['file_id']}`")
                st.write(f"**Status:** {upload['status']}")
            with col2:
                st.write(f"**Message:** {upload['message']}")
                if st.button(f"🔍 Search Similar", key=f"search_{upload['file_id']}"):
                    # Set search parameters and switch to search page
                    st.session_state.search_file_id = upload['file_id']
                    st.switch_page("search")
else:
    st.info("No files uploaded yet. Upload some audio files to get started!")

# Batch processing section
st.header("⚡ Batch Processing")

with st.expander("🔧 Advanced Options"):
    st.markdown("**Batch Processing Settings**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        batch_size = st.number_input(
            "Batch Size",
            min_value=1,
            max_value=10,
            value=5,
            help="Number of files to process simultaneously"
        )
        
        enable_gpu = st.checkbox(
            "Enable GPU Processing",
            value=False,
            help="Use GPU for faster embedding generation (if available)"
        )
    
    with col2:
        quality_preset = st.selectbox(
            "Quality Preset",
            options=["Fast", "Balanced", "High Quality"],
            index=1,
            help="Processing quality vs speed trade-off"
        )
        
        auto_normalize = st.checkbox(
            "Auto Normalize",
            value=True,
            help="Automatically normalize audio levels"
        )
    
    if st.button("🔄 Apply Batch Settings"):
        st.success("Batch settings updated!")
        # Here you would typically send these settings to the backend

# Processing status
if st.button("📊 Check Processing Status"):
    try:
        response = requests.get(f"{backend_url}/api/v1/audio/status")
        if response.status_code == 200:
            status_data = response.json()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Files", status_data.get('total_files', 0))
            
            with col2:
                st.metric("Processing Queue", status_data.get('queue_size', 0))
            
            with col3:
                st.metric("Completed Today", status_data.get('completed_today', 0))
            
            if status_data.get('recent_activity'):
                st.subheader("Recent Activity")
                for activity in status_data['recent_activity']:
                    st.write(f"• {activity}")
        else:
            st.error("Failed to fetch processing status")
    except Exception as e:
        st.error(f"Error fetching status: {str(e)}")