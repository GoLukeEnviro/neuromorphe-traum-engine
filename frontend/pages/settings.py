import streamlit as st
import requests
import os
from typing import Dict, Any


class SettingsPage:
    """Application settings and configuration page"""
    
    def __init__(self):
        self.backend_url = None
    
    def render(self):
        """Render the settings page"""
        self.backend_url = st.session_state.get('backend_url', 'http://localhost:8000')
        
        st.title("⚙️ Settings & Configuration")
        st.markdown("Configure the Neuromorphe Traum-Engine application settings.")
        
        # Settings tabs
        tab1, tab2, tab3, tab4 = st.tabs(["🔗 Backend", "🎵 Audio", "🔍 Search", "📊 System"])
        
        with tab1:
            self._render_backend_settings()
        
        with tab2:
            self._render_audio_settings()
        
        with tab3:
            self._render_search_settings()
        
        with tab4:
            self._render_system_info()
    
    def _render_backend_settings(self):
        """Render backend connection settings"""
        st.header("🔗 Backend Connection")
        
        with st.form("backend_settings"):
            # Backend URL configuration
            current_url = st.session_state.get('backend_url', 'http://localhost:8000')
            
            backend_url = st.text_input(
                "Backend URL",
                value=current_url,
                help="URL of the FastAPI backend server"
            )
            
            # Connection timeout
            timeout = st.number_input(
                "Connection Timeout (seconds)",
                min_value=5,
                max_value=120,
                value=st.session_state.get('connection_timeout', 30),
                help="Timeout for API requests"
            )
            
            # Auto-retry settings
            max_retries = st.number_input(
                "Max Retries",
                min_value=0,
                max_value=5,
                value=st.session_state.get('max_retries', 3),
                help="Maximum number of retry attempts for failed requests"
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.form_submit_button("💾 Save Settings", type="primary"):
                    st.session_state.backend_url = backend_url
                    st.session_state.connection_timeout = timeout
                    st.session_state.max_retries = max_retries
                    st.success("✅ Backend settings saved!")
                    st.rerun()
            
            with col2:
                if st.form_submit_button("🔄 Test Connection"):
                    self._test_backend_connection(backend_url, timeout)
        
        # Connection status
        self._show_connection_status()
    
    def _render_audio_settings(self):
        """Render audio processing settings"""
        st.header("🎵 Audio Processing")
        
        with st.form("audio_settings"):
            # File upload limits
            max_file_size = st.number_input(
                "Max File Size (MB)",
                min_value=1,
                max_value=100,
                value=st.session_state.get('max_file_size_mb', 50),
                help="Maximum size for uploaded audio files"
            )
            
            # Supported formats
            st.subheader("📁 Supported Audio Formats")
            
            supported_formats = st.session_state.get('supported_formats', ['wav', 'mp3', 'flac', 'ogg'])
            
            col1, col2 = st.columns(2)
            
            with col1:
                wav_enabled = st.checkbox("WAV", value='wav' in supported_formats)
                mp3_enabled = st.checkbox("MP3", value='mp3' in supported_formats)
            
            with col2:
                flac_enabled = st.checkbox("FLAC", value='flac' in supported_formats)
                ogg_enabled = st.checkbox("OGG", value='ogg' in supported_formats)
            
            # Processing options
            st.subheader("⚡ Processing Options")
            
            auto_process = st.checkbox(
                "Auto-process uploads",
                value=st.session_state.get('auto_process', True),
                help="Automatically generate embeddings after upload"
            )
            
            batch_size = st.number_input(
                "Batch Processing Size",
                min_value=1,
                max_value=10,
                value=st.session_state.get('batch_size', 5),
                help="Number of files to process simultaneously"
            )
            
            if st.form_submit_button("💾 Save Audio Settings", type="primary"):
                # Update supported formats
                new_formats = []
                if wav_enabled:
                    new_formats.append('wav')
                if mp3_enabled:
                    new_formats.append('mp3')
                if flac_enabled:
                    new_formats.append('flac')
                if ogg_enabled:
                    new_formats.append('ogg')
                
                st.session_state.max_file_size_mb = max_file_size
                st.session_state.supported_formats = new_formats
                st.session_state.auto_process = auto_process
                st.session_state.batch_size = batch_size
                
                st.success("✅ Audio settings saved!")
                st.rerun()
    
    def _render_search_settings(self):
        """Render search configuration settings"""
        st.header("🔍 Search Configuration")
        
        with st.form("search_settings"):
            # Default search parameters
            default_limit = st.number_input(
                "Default Result Limit",
                min_value=5,
                max_value=100,
                value=st.session_state.get('default_search_limit', 10),
                help="Default number of search results to return"
            )
            
            default_threshold = st.slider(
                "Default Similarity Threshold",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.get('default_similarity_threshold', 0.3),
                step=0.05,
                help="Default minimum similarity score for search results"
            )
            
            # Search behavior
            st.subheader("🎯 Search Behavior")
            
            enable_fuzzy_search = st.checkbox(
                "Enable Fuzzy Search",
                value=st.session_state.get('enable_fuzzy_search', True),
                help="Allow approximate matching for text queries"
            )
            
            cache_results = st.checkbox(
                "Cache Search Results",
                value=st.session_state.get('cache_search_results', True),
                help="Cache search results for faster repeated queries"
            )
            
            max_history = st.number_input(
                "Max Search History",
                min_value=5,
                max_value=100,
                value=st.session_state.get('max_search_history', 20),
                help="Maximum number of searches to keep in history"
            )
            
            # Advanced options
            st.subheader("🔧 Advanced Options")
            
            enable_category_boost = st.checkbox(
                "Enable Category Boost",
                value=st.session_state.get('enable_category_boost', False),
                help="Boost results from the same category"
            )
            
            enable_bpm_filtering = st.checkbox(
                "Enable BPM Filtering",
                value=st.session_state.get('enable_bpm_filtering', True),
                help="Allow filtering by BPM range"
            )
            
            if st.form_submit_button("💾 Save Search Settings", type="primary"):
                st.session_state.default_search_limit = default_limit
                st.session_state.default_similarity_threshold = default_threshold
                st.session_state.enable_fuzzy_search = enable_fuzzy_search
                st.session_state.cache_search_results = cache_results
                st.session_state.max_search_history = max_history
                st.session_state.enable_category_boost = enable_category_boost
                st.session_state.enable_bpm_filtering = enable_bpm_filtering
                
                st.success("✅ Search settings saved!")
                st.rerun()
    
    def _render_system_info(self):
        """Render system information and diagnostics"""
        st.header("📊 System Information")
        
        # Backend health check
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🏥 Check Backend Health", use_container_width=True):
                self._check_backend_health()
        
        with col2:
            if st.button("📊 Get System Stats", use_container_width=True):
                self._get_system_stats()
        
        # Session state info
        st.subheader("🔧 Session Information")
        
        with st.expander("📋 Session State", expanded=False):
            # Filter out sensitive or large data
            filtered_state = {}
            for key, value in st.session_state.items():
                if key not in ['search_results', 'uploaded_files'] and not key.startswith('_'):
                    filtered_state[key] = value
            
            st.json(filtered_state)
        
        # Application info
        st.subheader("ℹ️ Application Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Application**: Neuromorphe Traum-Engine v2.0")
            st.write("**Frontend**: Streamlit")
            st.write("**Backend**: FastAPI")
        
        with col2:
            st.write("**AI Model**: CLAP (Contrastive Language-Audio Pre-training)")
            st.write("**Database**: SQLite")
            st.write("**Architecture**: Domain-driven Design")
        
        # Clear data options
        st.subheader("🗑️ Data Management")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🗑️ Clear Search History", use_container_width=True):
                if 'search_history' in st.session_state:
                    del st.session_state.search_history
                st.success("Search history cleared!")
                st.rerun()
        
        with col2:
            if st.button("🗑️ Clear Upload History", use_container_width=True):
                if 'uploaded_files' in st.session_state:
                    del st.session_state.uploaded_files
                st.success("Upload history cleared!")
                st.rerun()
        
        with col3:
            if st.button("🗑️ Clear All Cache", use_container_width=True):
                # Clear various cached data
                keys_to_clear = ['search_results', 'last_query', 'selected_results', 'stats']
                for key in keys_to_clear:
                    if key in st.session_state:
                        del st.session_state[key]
                st.success("All cache cleared!")
                st.rerun()
    
    def _test_backend_connection(self, url: str, timeout: int):
        """Test connection to backend"""
        try:
            with st.spinner("Testing backend connection..."):
                response = requests.get(f"{url}/", timeout=timeout)
                
                if response.status_code == 200:
                    st.success(f"✅ Connection successful! Backend is reachable at {url}")
                    
                    # Try to get API info
                    try:
                        api_response = requests.get(f"{url}/api/v1/", timeout=timeout)
                        if api_response.status_code == 200:
                            api_info = api_response.json()
                            st.info(f"📡 API Version: {api_info.get('version', 'Unknown')}")
                    except:
                        pass
                else:
                    st.error(f"❌ Connection failed! Status code: {response.status_code}")
        except requests.exceptions.Timeout:
            st.error(f"⏰ Connection timeout after {timeout} seconds")
        except requests.exceptions.ConnectionError:
            st.error(f"🔌 Connection error! Cannot reach {url}")
        except Exception as e:
            st.error(f"❌ Unexpected error: {str(e)}")
    
    def _show_connection_status(self):
        """Show current connection status"""
        st.subheader("📡 Connection Status")
        
        try:
            response = requests.get(f"{self.backend_url}/", timeout=5)
            if response.status_code == 200:
                st.success(f"🟢 Connected to {self.backend_url}")
            else:
                st.error(f"🔴 Backend returned status {response.status_code}")
        except:
            st.error(f"🔴 Cannot connect to {self.backend_url}")
    
    def _check_backend_health(self):
        """Check backend health status"""
        try:
            with st.spinner("Checking backend health..."):
                # Check main health endpoint
                response = requests.get(f"{self.backend_url}/health", timeout=10)
                
                if response.status_code == 200:
                    st.success("✅ Backend is healthy!")
                    
                    # Check individual service health
                    services = ['audio', 'search']
                    
                    for service in services:
                        try:
                            service_response = requests.get(
                                f"{self.backend_url}/api/v1/{service}/health",
                                timeout=5
                            )
                            if service_response.status_code == 200:
                                st.success(f"✅ {service.title()} service is healthy")
                            else:
                                st.warning(f"⚠️ {service.title()} service returned status {service_response.status_code}")
                        except:
                            st.error(f"❌ {service.title()} service is not responding")
                else:
                    st.error(f"❌ Backend health check failed! Status: {response.status_code}")
        except Exception as e:
            st.error(f"❌ Health check failed: {str(e)}")
    
    def _get_system_stats(self):
        """Get and display system statistics"""
        try:
            with st.spinner("Fetching system statistics..."):
                # Get search stats
                search_response = requests.get(f"{self.backend_url}/api/v1/search/stats", timeout=10)
                
                if search_response.status_code == 200:
                    stats = search_response.json()
                    
                    st.subheader("📊 System Statistics")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Files", stats.get('total_files', 0))
                    
                    with col2:
                        st.metric("Total Embeddings", stats.get('total_embeddings', 0))
                    
                    with col3:
                        categories = stats.get('categories', [])
                        st.metric("Categories", len(categories))
                    
                    with col4:
                        avg_duration = stats.get('avg_duration', 0)
                        st.metric("Avg Duration", f"{avg_duration:.1f}s" if avg_duration else "N/A")
                    
                    # Category breakdown
                    if categories:
                        st.subheader("📂 Category Breakdown")
                        
                        for cat in categories:
                            col1, col2, col3 = st.columns([2, 1, 1])
                            
                            with col1:
                                st.write(f"**{cat['category']}**")
                            
                            with col2:
                                st.write(f"{cat['count']} files")
                            
                            with col3:
                                if cat.get('avg_bpm'):
                                    st.write(f"Avg BPM: {cat['avg_bpm']:.1f}")
                else:
                    st.error("Failed to fetch system statistics")
        except Exception as e:
            st.error(f"Error fetching system stats: {str(e)}")