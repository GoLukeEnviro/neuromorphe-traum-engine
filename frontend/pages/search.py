import streamlit as st
import requests
import json
from typing import List, Dict, Any


class SearchPage:
    """Semantic search page for audio files"""
    
    def __init__(self):
        self.backend_url = None
    
    def render(self):
        """Render the search page"""
        self.backend_url = st.session_state.get('backend_url', 'http://localhost:8000')
        
        st.title("🔍 Semantic Audio Search")
        st.markdown("Search for audio files using natural language descriptions or find similar tracks.")
        
        # Search tabs
        tab1, tab2 = st.tabs(["📝 Text Search", "🎵 Similarity Search"])
        
        with tab1:
            self._render_text_search()
        
        with tab2:
            self._render_similarity_search()
        
        # Search history
        self._render_search_history()
    
    def _render_text_search(self):
        """Render text-based semantic search"""
        st.header("📝 Describe the Audio You're Looking For")
        
        with st.form("text_search_form"):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                query = st.text_input(
                    "Search Query",
                    placeholder="e.g., 'energetic techno with heavy bass', 'ambient nature sounds', 'uplifting trance melody'",
                    help="Describe the audio you're looking for in natural language"
                )
            
            with col2:
                limit = st.selectbox(
                    "Max Results",
                    options=[5, 10, 20, 50],
                    index=1,
                    help="Maximum number of results to return"
                )
            
            # Advanced filters
            with st.expander("🔧 Advanced Filters"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    category_filter = st.selectbox(
                        "Category",
                        options=["", "techno", "house", "ambient", "drum_and_bass", "trance", "other"],
                        help="Filter by audio category"
                    )
                
                with col2:
                    min_bpm = st.number_input(
                        "Min BPM",
                        min_value=60,
                        max_value=200,
                        value=None,
                        help="Minimum beats per minute"
                    )
                
                with col3:
                    max_bpm = st.number_input(
                        "Max BPM",
                        min_value=60,
                        max_value=200,
                        value=None,
                        help="Maximum beats per minute"
                    )
                
                similarity_threshold = st.slider(
                    "Similarity Threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.3,
                    step=0.05,
                    help="Minimum similarity score (higher = more strict)"
                )
            
            submitted = st.form_submit_button(
                "🔍 Search",
                type="primary",
                use_container_width=True
            )
            
            if submitted and query.strip():
                self._perform_text_search(
                    query=query.strip(),
                    limit=limit,
                    category=category_filter if category_filter else None,
                    min_bpm=min_bpm,
                    max_bpm=max_bpm,
                    threshold=similarity_threshold
                )
    
    def _render_similarity_search(self):
        """Render similarity-based search"""
        st.header("🎵 Find Similar Audio Files")
        
        # Get available files for selection
        available_files = self._get_available_files()
        
        if not available_files:
            st.warning("No audio files available. Please upload some files first.")
            return
        
        with st.form("similarity_search_form"):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                selected_file = st.selectbox(
                    "Reference Audio File",
                    options=available_files,
                    format_func=lambda x: f"{x['filename']} ({x['id']})",
                    help="Select an audio file to find similar tracks"
                )
            
            with col2:
                limit = st.selectbox(
                    "Max Results",
                    options=[5, 10, 20, 50],
                    index=1,
                    help="Maximum number of similar files to return"
                )
            
            # Advanced filters
            with st.expander("🔧 Advanced Filters"):
                col1, col2 = st.columns(2)
                
                with col1:
                    category_filter = st.selectbox(
                        "Category",
                        options=["", "techno", "house", "ambient", "drum_and_bass", "trance", "other"],
                        help="Filter by audio category",
                        key="sim_category"
                    )
                
                with col2:
                    similarity_threshold = st.slider(
                        "Similarity Threshold",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.5,
                        step=0.05,
                        help="Minimum similarity score",
                        key="sim_threshold"
                    )
            
            submitted = st.form_submit_button(
                "🔍 Find Similar",
                type="primary",
                use_container_width=True
            )
            
            if submitted and selected_file:
                self._perform_similarity_search(
                    file_id=selected_file['id'],
                    limit=limit,
                    category=category_filter if category_filter else None,
                    threshold=similarity_threshold
                )
    
    def _perform_text_search(self, query: str, limit: int, category: str = None, 
                           min_bpm: int = None, max_bpm: int = None, threshold: float = 0.3):
        """Perform text-based semantic search"""
        try:
            with st.spinner("🔍 Searching for matching audio files..."):
                # Prepare request data
                search_data = {
                    "query": query,
                    "limit": limit,
                    "threshold": threshold
                }
                
                if category:
                    search_data["category"] = category
                if min_bpm is not None:
                    search_data["min_bpm"] = min_bpm
                if max_bpm is not None:
                    search_data["max_bpm"] = max_bpm
                
                response = requests.post(
                    f"{self.backend_url}/api/v1/search/text",
                    json=search_data,
                    timeout=30
                )
                
                if response.status_code == 200:
                    results = response.json()
                    self._display_search_results(results, query, "text")
                    
                    # Store in session state
                    st.session_state.search_results = results
                    st.session_state.last_query = query
                    
                    # Add to search history
                    self._add_to_search_history(query, "text", len(results.get('results', [])))
                    
                else:
                    st.error(f"Search failed: {response.text}")
                    
        except Exception as e:
            st.error(f"Error performing search: {str(e)}")
    
    def _perform_similarity_search(self, file_id: str, limit: int, 
                                 category: str = None, threshold: float = 0.5):
        """Perform similarity-based search"""
        try:
            with st.spinner("🎵 Finding similar audio files..."):
                # Prepare request data
                search_data = {
                    "file_id": file_id,
                    "limit": limit,
                    "threshold": threshold
                }
                
                if category:
                    search_data["category"] = category
                
                response = requests.post(
                    f"{self.backend_url}/api/v1/search/similar",
                    json=search_data,
                    timeout=30
                )
                
                if response.status_code == 200:
                    results = response.json()
                    self._display_search_results(results, f"Similar to {file_id}", "similarity")
                    
                    # Store in session state
                    st.session_state.search_results = results
                    st.session_state.last_query = f"Similar to {file_id}"
                    
                    # Add to search history
                    self._add_to_search_history(f"Similar to {file_id}", "similarity", len(results.get('results', [])))
                    
                else:
                    st.error(f"Similarity search failed: {response.text}")
                    
        except Exception as e:
            st.error(f"Error performing similarity search: {str(e)}")
    
    def _display_search_results(self, results: Dict[str, Any], query: str, search_type: str):
        """Display search results"""
        search_results = results.get('results', [])
        total_found = results.get('total_found', 0)
        search_time = results.get('search_time_ms', 0)
        
        if not search_results:
            st.warning("🔍 No matching audio files found. Try adjusting your search criteria.")
            return
        
        # Results header
        st.success(f"✅ Found {total_found} matching files in {search_time:.1f}ms")
        
        # Results display
        st.subheader(f"🎯 Search Results for: '{query}'")
        
        for i, result in enumerate(search_results, 1):
            with st.container():
                # Result header
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    st.write(f"**{i}. {result.get('filename', 'Unknown')}**")
                
                with col2:
                    similarity = result.get('similarity_score', 0)
                    color = "🟢" if similarity > 0.7 else "🟡" if similarity > 0.5 else "🔴"
                    st.write(f"{color} {similarity:.3f}")
                
                with col3:
                    st.code(result.get('id', 'N/A'))
                
                # Result details
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    category = result.get('category', 'N/A')
                    st.write(f"📂 **Category**: {category}")
                
                with col2:
                    bpm = result.get('bpm', 'N/A')
                    st.write(f"🥁 **BPM**: {bpm}")
                
                with col3:
                    duration = result.get('duration', 'N/A')
                    if duration != 'N/A':
                        duration = f"{duration:.1f}s"
                    st.write(f"⏱️ **Duration**: {duration}")
                
                with col4:
                    sample_rate = result.get('sample_rate', 'N/A')
                    if sample_rate != 'N/A':
                        sample_rate = f"{sample_rate} Hz"
                    st.write(f"🎚️ **Sample Rate**: {sample_rate}")
                
                # Action buttons
                col1, col2, col3 = st.columns([1, 1, 2])
                
                with col1:
                    if st.button(f"🔍 Details", key=f"details_{result.get('id')}_{i}"):
                        self._show_file_details(result.get('id'))
                
                with col2:
                    if st.button(f"🎵 Find Similar", key=f"similar_{result.get('id')}_{i}"):
                        self._perform_similarity_search(result.get('id'), 10)
                
                st.write("---")
    
    def _get_available_files(self) -> List[Dict[str, str]]:
        """Get list of available audio files"""
        try:
            response = requests.get(f"{self.backend_url}/api/v1/audio/files", timeout=10)
            if response.status_code == 200:
                file_ids = response.json()
                
                # Get details for each file
                files = []
                for file_id in file_ids[:50]:  # Limit to first 50 files
                    try:
                        detail_response = requests.get(
                            f"{self.backend_url}/api/v1/audio/files/{file_id}",
                            timeout=5
                        )
                        if detail_response.status_code == 200:
                            info = detail_response.json()
                            if info.get('has_embedding'):  # Only include files with embeddings
                                files.append({
                                    'id': file_id,
                                    'filename': info.get('filename', file_id)
                                })
                    except:
                        continue
                
                return files
            else:
                return []
        except Exception as e:
            st.error(f"Error fetching available files: {str(e)}")
            return []
    
    def _show_file_details(self, file_id: str):
        """Show detailed information about a file"""
        try:
            response = requests.get(
                f"{self.backend_url}/api/v1/audio/files/{file_id}",
                timeout=10
            )
            if response.status_code == 200:
                info = response.json()
                
                with st.expander(f"📋 Details for {info.get('filename', file_id)}", expanded=True):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**File ID**: `{file_id}`")
                        st.write(f"**Filename**: {info.get('filename', 'N/A')}")
                        st.write(f"**Category**: {info.get('category', 'N/A')}")
                        st.write(f"**BPM**: {info.get('bpm', 'N/A')}")
                    
                    with col2:
                        st.write(f"**Duration**: {info.get('duration', 'N/A')}s")
                        st.write(f"**Sample Rate**: {info.get('sample_rate', 'N/A')} Hz")
                        st.write(f"**File Size**: {info.get('file_size', 'N/A')} bytes")
                        st.write(f"**Has Embedding**: {'✅' if info.get('has_embedding') else '❌'}")
                    
                    if info.get('created_at'):
                        st.write(f"**Created**: {info['created_at']}")
                    
                    if info.get('processed_at'):
                        st.write(f"**Processed**: {info['processed_at']}")
            else:
                st.error(f"Failed to get file details: {response.text}")
        except Exception as e:
            st.error(f"Error getting file details: {str(e)}")
    
    def _add_to_search_history(self, query: str, search_type: str, result_count: int):
        """Add search to history"""
        if 'search_history' not in st.session_state:
            st.session_state.search_history = []
        
        history_entry = {
            'query': query,
            'type': search_type,
            'result_count': result_count,
            'timestamp': str(pd.Timestamp.now())
        }
        
        st.session_state.search_history.insert(0, history_entry)
        
        # Keep only last 20 searches
        if len(st.session_state.search_history) > 20:
            st.session_state.search_history = st.session_state.search_history[:20]
    
    def _render_search_history(self):
        """Render search history section"""
        if st.session_state.get('search_history'):
            st.header("📚 Recent Searches")
            
            with st.expander("🕒 Search History", expanded=False):
                for i, entry in enumerate(st.session_state.search_history[:10]):
                    col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                    
                    with col1:
                        st.write(f"**{entry['query']}**")
                    
                    with col2:
                        type_icon = "📝" if entry['type'] == 'text' else "🎵"
                        st.write(f"{type_icon} {entry['type']}")
                    
                    with col3:
                        st.write(f"📊 {entry['result_count']} results")
                    
                    with col4:
                        if st.button("🔄", key=f"repeat_{i}", help="Repeat this search"):
                            if entry['type'] == 'text':
                                # Trigger text search
                                st.session_state.repeat_query = entry['query']
                            else:
                                # For similarity searches, we'd need the file_id
                                st.info("Similarity search repeat not implemented")
                    
                    st.write("---")