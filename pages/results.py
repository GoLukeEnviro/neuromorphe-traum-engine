import streamlit as st
import requests
import pandas as pd
from typing import List, Dict, Any


class ResultsPage:
    """Search results display page"""
    
    def __init__(self):
        self.backend_url = None
    
    def render(self):
        """Render the results page"""
        self.backend_url = st.session_state.get('backend_url', 'http://localhost:8000')
        
        st.title("🎯 Search Results")
        
        # Check if we have results to display
        if not st.session_state.get('search_results'):
            st.info("🔍 No search results to display. Please perform a search first.")
            if st.button("🔍 Go to Search", type="primary"):
                st.switch_page("pages/search.py")
            return
        
        # Display results
        self._render_results_header()
        self._render_results_table()
        self._render_results_actions()
        self._render_export_options()
    
    def _render_results_header(self):
        """Render results header with summary"""
        results = st.session_state.search_results
        last_query = st.session_state.get('last_query', 'Unknown')
        
        search_results = results.get('results', [])
        total_found = results.get('total_found', 0)
        search_time = results.get('search_time_ms', 0)
        
        # Header info
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.subheader(f"Query: '{last_query}'")
        
        with col2:
            st.metric("Results Found", total_found)
        
        with col3:
            st.metric("Search Time", f"{search_time:.1f}ms")
        
        # Quick stats
        if search_results:
            avg_similarity = sum(r.get('similarity_score', 0) for r in search_results) / len(search_results)
            categories = list(set(r.get('category', 'Unknown') for r in search_results if r.get('category')))
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Avg Similarity", f"{avg_similarity:.3f}")
            
            with col2:
                st.metric("Categories", len(categories))
            
            with col3:
                bpm_values = [r.get('bpm') for r in search_results if r.get('bpm')]
                avg_bpm = sum(bpm_values) / len(bpm_values) if bpm_values else 0
                st.metric("Avg BPM", f"{avg_bpm:.0f}" if avg_bpm > 0 else "N/A")
        
        st.write("---")
    
    def _render_results_table(self):
        """Render results in a table format"""
        results = st.session_state.search_results
        search_results = results.get('results', [])
        
        if not search_results:
            return
        
        st.header("📊 Results Table")
        
        # Create DataFrame for better display
        df_data = []
        for i, result in enumerate(search_results, 1):
            df_data.append({
                'Rank': i,
                'Filename': result.get('filename', 'Unknown'),
                'Similarity': f"{result.get('similarity_score', 0):.3f}",
                'Category': result.get('category', 'N/A'),
                'BPM': result.get('bpm', 'N/A'),
                'Duration': f"{result.get('duration', 0):.1f}s" if result.get('duration') else 'N/A',
                'Sample Rate': f"{result.get('sample_rate', 0)} Hz" if result.get('sample_rate') else 'N/A',
                'File ID': result.get('id', 'N/A')
            })
        
        df = pd.DataFrame(df_data)
        
        # Display table with selection
        selected_indices = st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            selection_mode="multi-row",
            on_select="rerun"
        )
        
        # Handle selection
        if hasattr(selected_indices, 'selection') and selected_indices.selection.rows:
            selected_rows = selected_indices.selection.rows
            st.session_state.selected_results = [search_results[i] for i in selected_rows]
            st.success(f"✅ Selected {len(selected_rows)} results")
        
        st.write("---")
    
    def _render_results_actions(self):
        """Render action buttons for results"""
        st.header("🎬 Actions")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("🔍 New Search", use_container_width=True):
                st.switch_page("pages/search.py")
        
        with col2:
            if st.button("📋 Show Details", use_container_width=True):
                self._show_detailed_view()
        
        with col3:
            if st.button("🎵 Find Similar", use_container_width=True):
                self._find_similar_to_selected()
        
        with col4:
            if st.button("🔄 Refresh Results", use_container_width=True):
                st.rerun()
    
    def _show_detailed_view(self):
        """Show detailed view of results"""
        results = st.session_state.search_results.get('results', [])
        
        st.subheader("🔍 Detailed View")
        
        # Show selected results or all if none selected
        display_results = st.session_state.get('selected_results', results)
        
        if not display_results:
            st.info("No results to display in detail view.")
            return
        
        for i, result in enumerate(display_results, 1):
            with st.expander(f"📁 {i}. {result.get('filename', 'Unknown')}", expanded=i <= 3):
                self._render_result_details(result)
    
    def _render_result_details(self, result: Dict[str, Any]):
        """Render detailed information for a single result"""
        file_id = result.get('id')
        
        # Basic info
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**File ID**: `{file_id}`")
            st.write(f"**Filename**: {result.get('filename', 'N/A')}")
            st.write(f"**Category**: {result.get('category', 'N/A')}")
            st.write(f"**BPM**: {result.get('bpm', 'N/A')}")
            st.write(f"**Similarity Score**: {result.get('similarity_score', 0):.3f}")
        
        with col2:
            st.write(f"**Duration**: {result.get('duration', 'N/A')}s")
            st.write(f"**Sample Rate**: {result.get('sample_rate', 'N/A')} Hz")
            st.write(f"**File Size**: {result.get('file_size', 'N/A')} bytes")
            st.write(f"**Has Embedding**: {'✅' if result.get('has_embedding') else '❌'}")
        
        # Action buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button(f"🔍 Get Full Details", key=f"details_{file_id}"):
                self._fetch_full_details(file_id)
        
        with col2:
            if st.button(f"🎵 Find Similar", key=f"similar_{file_id}"):
                self._find_similar_to_file(file_id)
        
        with col3:
            if st.button(f"📊 Get Embedding", key=f"embedding_{file_id}"):
                self._show_embedding_info(file_id)
    
    def _fetch_full_details(self, file_id: str):
        """Fetch and display full file details"""
        try:
            response = requests.get(
                f"{self.backend_url}/api/v1/audio/files/{file_id}",
                timeout=10
            )
            if response.status_code == 200:
                info = response.json()
                
                st.json(info)
                
                if info.get('created_at'):
                    st.write(f"**Created**: {info['created_at']}")
                if info.get('processed_at'):
                    st.write(f"**Processed**: {info['processed_at']}")
            else:
                st.error(f"Failed to get file details: {response.text}")
        except Exception as e:
            st.error(f"Error fetching file details: {str(e)}")
    
    def _find_similar_to_file(self, file_id: str):
        """Find similar files to the selected one"""
        try:
            with st.spinner(f"Finding files similar to {file_id}..."):
                response = requests.post(
                    f"{self.backend_url}/api/v1/search/similar",
                    json={
                        "file_id": file_id,
                        "limit": 10,
                        "threshold": 0.5
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    results = response.json()
                    
                    # Update session state with new results
                    st.session_state.search_results = results
                    st.session_state.last_query = f"Similar to {file_id}"
                    
                    st.success(f"✅ Found {len(results.get('results', []))} similar files")
                    st.rerun()
                else:
                    st.error(f"Similarity search failed: {response.text}")
        except Exception as e:
            st.error(f"Error finding similar files: {str(e)}")
    
    def _find_similar_to_selected(self):
        """Find similar files to selected results"""
        selected_results = st.session_state.get('selected_results', [])
        
        if not selected_results:
            st.warning("Please select at least one result first.")
            return
        
        if len(selected_results) > 1:
            st.warning("Please select only one result for similarity search.")
            return
        
        file_id = selected_results[0].get('id')
        self._find_similar_to_file(file_id)
    
    def _show_embedding_info(self, file_id: str):
        """Show embedding information for a file"""
        try:
            response = requests.get(
                f"{self.backend_url}/api/v1/audio/files/{file_id}/embedding",
                timeout=10
            )
            if response.status_code == 200:
                embedding_data = response.json()
                
                st.subheader(f"🧠 Embedding Info for {file_id}")
                
                if 'embedding' in embedding_data:
                    embedding = embedding_data['embedding']
                    st.write(f"**Embedding Dimensions**: {len(embedding)}")
                    st.write(f"**Embedding Norm**: {sum(x*x for x in embedding)**0.5:.3f}")
                    st.write(f"**Min Value**: {min(embedding):.3f}")
                    st.write(f"**Max Value**: {max(embedding):.3f}")
                    st.write(f"**Mean Value**: {sum(embedding)/len(embedding):.3f}")
                    
                    # Show first few dimensions
                    st.write("**First 10 dimensions**:")
                    st.code(str(embedding[:10]))
                else:
                    st.warning("No embedding data found")
            else:
                st.error(f"Failed to get embedding: {response.text}")
        except Exception as e:
            st.error(f"Error getting embedding: {str(e)}")
    
    def _render_export_options(self):
        """Render export options for results"""
        st.header("📤 Export Options")
        
        results = st.session_state.search_results.get('results', [])
        
        if not results:
            st.info("No results to export.")
            return
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Export as CSV", use_container_width=True):
                self._export_as_csv(results)
        
        with col2:
            if st.button("📋 Export as JSON", use_container_width=True):
                self._export_as_json(results)
        
        with col3:
            if st.button("📝 Export File IDs", use_container_width=True):
                self._export_file_ids(results)
    
    def _export_as_csv(self, results: List[Dict[str, Any]]):
        """Export results as CSV"""
        try:
            df_data = []
            for result in results:
                df_data.append({
                    'file_id': result.get('id', ''),
                    'filename': result.get('filename', ''),
                    'similarity_score': result.get('similarity_score', 0),
                    'category': result.get('category', ''),
                    'bpm': result.get('bpm', ''),
                    'duration': result.get('duration', ''),
                    'sample_rate': result.get('sample_rate', ''),
                    'file_size': result.get('file_size', '')
                })
            
            df = pd.DataFrame(df_data)
            csv = df.to_csv(index=False)
            
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"search_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        except Exception as e:
            st.error(f"Error exporting CSV: {str(e)}")
    
    def _export_as_json(self, results: List[Dict[str, Any]]):
        """Export results as JSON"""
        try:
            import json
            
            export_data = {
                'query': st.session_state.get('last_query', ''),
                'timestamp': str(pd.Timestamp.now()),
                'total_results': len(results),
                'results': results
            }
            
            json_str = json.dumps(export_data, indent=2)
            
            st.download_button(
                label="📥 Download JSON",
                data=json_str,
                file_name=f"search_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
        except Exception as e:
            st.error(f"Error exporting JSON: {str(e)}")
    
    def _export_file_ids(self, results: List[Dict[str, Any]]):
        """Export just the file IDs"""
        try:
            file_ids = [result.get('id', '') for result in results if result.get('id')]
            file_ids_text = '\n'.join(file_ids)
            
            st.download_button(
                label="📥 Download File IDs",
                data=file_ids_text,
                file_name=f"file_ids_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )
        except Exception as e:
            st.error(f"Error exporting file IDs: {str(e)}")