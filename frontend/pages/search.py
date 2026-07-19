import streamlit as st
import requests
import json
import time
from typing import List, Dict, Any, Optional

# Page configuration
backend_url = st.session_state.get('backend_url', 'http://localhost:8508')

search_examples = [
    "energetic techno with heavy bass",
    "soft ambient texture with nature sounds",
    "melodic trance with uplifting progression",
    "dark industrial drums",
    "warm analog pad sounds",
    "crisp hi-hats and percussion"
]

def check_backend_connection() -> bool:
    """Check if backend is reachable"""
    try:
        response = requests.get(f"{backend_url}/system/health", timeout=5)
        return response.status_code == 200
    except Exception:
        return False

def perform_text_search(query: str, limit: int, category: str = None, 
                       min_bpm: int = None, max_bpm: int = None, threshold: float = 0.3):
    """Perform text-based semantic search"""
    try:
        params = {
            'query': query,
            'limit': limit,
            'threshold': threshold
        }
        
        if category:
            params['category'] = category
        if min_bpm:
            params['min_bpm'] = min_bpm
        if max_bpm:
            params['max_bpm'] = max_bpm
        
        response = requests.get(f"{backend_url}/api/v1/search/text", params=params, timeout=30)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Search failed: {response.text}")
            return None
    except Exception as e:
        st.error(f"Search error: {str(e)}")
        return None

def perform_similarity_search(file_id: str, limit: int, 
                             category: str = None, threshold: float = 0.5):
    """Perform similarity search based on a reference file"""
    try:
        params = {
            'file_id': file_id,
            'limit': limit,
            'threshold': threshold
        }
        
        if category:
            params['category'] = category
        
        response = requests.get(f"{backend_url}/api/v1/search/similar", params=params, timeout=30)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Similarity search failed: {response.text}")
            return None
    except Exception as e:
        st.error(f"Similarity search error: {str(e)}")
        return None

def get_available_files() -> List[Dict[str, str]]:
    """Get list of available audio files"""
    try:
        response = requests.get(f"{backend_url}/api/v1/audio/files", timeout=10)
        if response.status_code == 200:
            file_ids = response.json()
            files = []
            
            for file_id in file_ids[:50]:  # Limit to first 50 files
                try:
                    file_response = requests.get(
                        f"{backend_url}/api/v1/audio/files/{file_id}", 
                        timeout=5
                    )
                    if file_response.status_code == 200:
                        file_info = file_response.json()
                        files.append({
                            'id': file_id,
                            'filename': file_info.get('filename', file_id),
                            'category': file_info.get('category', 'Unknown')
                        })
                except Exception:
                    continue
            
            return files
        else:
            return []
    except Exception:
        return []

def display_search_results(results: Dict[str, Any], query: str, search_type: str):
    """Display search results"""
    if not results or 'results' not in results:
        st.warning("No results found")
        return
    
    matches = results['results']
    
    if not matches:
        st.info("No matches found. Try adjusting your search terms or threshold.")
        return
    
    # Store results in session state
    st.session_state.search_results = results
    st.session_state.last_query = query
    
    st.success(f"Found {len(matches)} matches for '{query}'")
    
    # Results summary
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Results", len(matches))
    with col2:
        avg_score = sum(match.get('score', 0) for match in matches) / len(matches)
        st.metric("Avg Similarity", f"{avg_score:.3f}")
    with col3:
        categories = set(match.get('metadata', {}).get('category', 'Unknown') for match in matches)
        st.metric("Categories", len(categories))
    
    # Results display
    for i, match in enumerate(matches):
        with st.expander(f"🎵 {match.get('filename', 'Unknown')} (Score: {match.get('score', 0):.3f})"):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write(f"**File ID:** `{match.get('id', 'N/A')}`")
                st.write(f"**Filename:** {match.get('filename', 'N/A')}")
                
                metadata = match.get('metadata', {})
                if metadata:
                    st.write(f"**Category:** {metadata.get('category', 'N/A')}")
                    if metadata.get('bpm'):
                        st.write(f"**BPM:** {metadata['bpm']}")
                    if metadata.get('duration'):
                        st.write(f"**Duration:** {metadata['duration']:.1f}s")
                    if metadata.get('key'):
                        st.write(f"**Key:** {metadata['key']}")
            
            with col2:
                st.write(f"**Similarity Score:** {match.get('score', 0):.3f}")
                
                # Action buttons
                if st.button(f"🔍 Find Similar", key=f"similar_{i}"):
                    st.session_state.similarity_file_id = match.get('id')
                    st.rerun()
                
                if st.button(f"📥 Download", key=f"download_{i}"):
                    st.info("Download functionality coming soon!")
    
    # Add to search history
    add_to_search_history(query, search_type, len(matches))

def add_to_search_history(query: str, search_type: str, result_count: int):
    """Add search to history"""
    if 'search_history' not in st.session_state:
        st.session_state.search_history = []
    
    history_entry = {
        'query': query,
        'type': search_type,
        'result_count': result_count,
        'timestamp': time.time()
    }
    
    st.session_state.search_history.insert(0, history_entry)
    # Keep only last 20 searches
    st.session_state.search_history = st.session_state.search_history[:20]

# Main page content
st.title("🔍 Semantic Audio Search")
st.markdown(
    "Suche nach Audio-Dateien mit natürlicher Sprache oder finde ähnliche Tracks. "
    "Die NeuroMorph-Engine versteht semantische Beschreibungen und findet passende Stems."
)

# Check backend connection
if not check_backend_connection():
    st.error("❌ Backend nicht erreichbar. Bitte starten Sie den FastAPI-Server.")
    st.code("python -m uvicorn src.main:app --reload")
    st.stop()

# Search tabs
tab1, tab2, tab3 = st.tabs(["📝 Text Search", "🎵 Similarity Search", "📊 Advanced Search"])

with tab1:
    st.header("📝 Text-Based Search")
    st.markdown("Beschreibe den gewünschten Sound in natürlicher Sprache:")
    
    # Search examples
    with st.expander("💡 Search Examples", expanded=False):
        st.markdown("**Try these example searches:**")
        for example in search_examples:
            if st.button(f"'{example}'", key=f"example_{example}"):
                st.session_state.search_query = example
                st.rerun()
    
    # Search form
    with st.form("text_search_form"):
        query = st.text_input(
            "Search Query",
            value=st.session_state.get('search_query', ''),
            placeholder="e.g., 'dark techno kick with heavy distortion'",
            help="Describe the sound you're looking for in natural language"
        )
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            limit = st.slider("Max Results", 1, 50, 10)
            threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.3, 0.05)
        
        with col2:
            category = st.selectbox(
                "Category Filter",
                options=["", "Kick", "Snare", "Hi-Hat", "Percussion", "Bass", "Lead", "Pad", "FX", "Vocal"]
            )
        
        with col3:
            bpm_filter = st.checkbox("BPM Filter")
            if bpm_filter:
                min_bpm = st.number_input("Min BPM", 60, 200, 120)
                max_bpm = st.number_input("Max BPM", 60, 200, 140)
            else:
                min_bpm = max_bpm = None
        
        submitted = st.form_submit_button("🔍 Search", type="primary")
        
        if submitted and query:
            with st.spinner("Searching..."):
                results = perform_text_search(query, limit, category, min_bpm, max_bpm, threshold)
                if results:
                    display_search_results(results, query, "text")

with tab2:
    st.header("🎵 Similarity Search")
    st.markdown("Find audio files similar to a reference track:")
    
    # File selection
    available_files = get_available_files()
    
    if not available_files:
        st.warning("No audio files available. Upload some files first!")
    else:
        with st.form("similarity_search_form"):
            # File selector
            file_options = {f"{file['filename']} ({file['category']})": file['id'] for file in available_files}
            
            selected_file_display = st.selectbox(
                "Reference File",
                options=list(file_options.keys()),
                help="Select a file to find similar audio"
            )
            
            selected_file_id = file_options[selected_file_display]
            
            col1, col2 = st.columns(2)
            
            with col1:
                limit = st.slider("Max Results", 1, 50, 10, key="sim_limit")
                threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.5, 0.05, key="sim_threshold")
            
            with col2:
                category = st.selectbox(
                    "Category Filter",
                    options=["", "Kick", "Snare", "Hi-Hat", "Percussion", "Bass", "Lead", "Pad", "FX", "Vocal"],
                    key="sim_category"
                )
            
            submitted = st.form_submit_button("🔍 Find Similar", type="primary")
            
            if submitted:
                with st.spinner("Finding similar files..."):
                    results = perform_similarity_search(selected_file_id, limit, category, threshold)
                    if results:
                        display_search_results(results, f"Similar to {selected_file_display}", "similarity")

with tab3:
    st.header("📊 Advanced Search")
    st.markdown("Combine multiple search criteria for precise results:")
    
    with st.form("advanced_search_form"):
        query = st.text_input(
            "Base Query",
            placeholder="e.g., 'techno drums'",
            help="Base semantic search query"
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Filters")
            categories = st.multiselect(
                "Categories",
                options=["Kick", "Snare", "Hi-Hat", "Percussion", "Bass", "Lead", "Pad", "FX", "Vocal"]
            )
            
            bpm_range = st.slider(
                "BPM Range",
                60, 200, (120, 140),
                help="Filter by tempo range"
            )
            
            keys = st.multiselect(
                "Musical Keys",
                options=["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
            )
        
        with col2:
            st.subheader("Search Parameters")
            limit = st.slider("Max Results", 1, 100, 20, key="adv_limit")
            threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.3, 0.05, key="adv_threshold")
            
            quality = st.selectbox(
                "Quality Filter",
                options=["", "High", "Medium", "Low"],
                help="Filter by audio quality"
            )
            
            sort_by = st.selectbox(
                "Sort By",
                options=["Similarity", "BPM", "Duration", "Upload Date"],
                help="Sort results by criteria"
            )
        
        submitted = st.form_submit_button("🔍 Advanced Search", type="primary")
        
        if submitted and query:
            with st.spinner("Performing advanced search..."):
                # For now, use basic text search with filters
                # In a real implementation, you'd have a dedicated advanced search endpoint
                results = perform_text_search(
                    query, limit, 
                    categories[0] if categories else None,
                    bpm_range[0], bpm_range[1], threshold
                )
                if results:
                    display_search_results(results, f"Advanced: {query}", "advanced")

# Search History
st.header("📚 Search History")

if 'search_history' in st.session_state and st.session_state.search_history:
    with st.expander("Recent Searches", expanded=False):
        for i, entry in enumerate(st.session_state.search_history[:10]):
            col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
            
            with col1:
                st.write(f"**{entry['query']}**")
            
            with col2:
                st.write(entry['type'].title())
            
            with col3:
                st.write(f"{entry['result_count']} results")
            
            with col4:
                if st.button("🔄 Repeat", key=f"repeat_{i}"):
                    st.session_state.search_query = entry['query']
                    st.rerun()
else:
    st.info("No search history yet. Perform some searches to see them here!")

# Quick actions
st.header("⚡ Quick Actions")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🎲 Random Search", use_container_width=True):
        import random
        random_query = random.choice(search_examples)
        st.session_state.search_query = random_query
        st.rerun()

with col2:
    if st.button("📊 Search Stats", use_container_width=True):
        try:
            response = requests.get(f"{backend_url}/api/v1/search/stats")
            if response.status_code == 200:
                stats = response.json()
                st.json(stats)
            else:
                st.error("Failed to fetch search stats")
        except Exception as e:
            st.error(f"Error: {str(e)}")

with col3:
    if st.button("🔄 Clear History", use_container_width=True):
        st.session_state.search_history = []
        st.success("Search history cleared!")
        st.rerun()