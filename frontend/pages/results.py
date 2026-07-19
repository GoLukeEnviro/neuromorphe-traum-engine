import streamlit as st
import requests
import json
import os
import pandas as pd
from typing import List, Dict, Any, Optional

# Page configuration
backend_url = st.session_state.get('backend_url', 'http://localhost:8508')
results_per_page = 10

def load_all_stems():
    """Load all available stems"""
    try:
        response = requests.get(f"{backend_url}/api/v1/audio/files", timeout=10)
        if response.status_code == 200:
            file_ids = response.json()
            
            # Get details for each file
            all_stems = []
            for file_id in file_ids[:100]:  # Limit to first 100
                try:
                    file_response = requests.get(
                        f"{backend_url}/api/v1/audio/files/{file_id}", 
                        timeout=5
                    )
                    if file_response.status_code == 200:
                        file_info = file_response.json()
                        all_stems.append({
                            'id': file_id,
                            'filename': file_info.get('filename', file_id),
                            'category': file_info.get('category', 'Unknown'),
                            'bpm': file_info.get('bpm'),
                            'duration': file_info.get('duration'),
                            'similarity_score': 1.0  # Default for all stems view
                        })
                except Exception:
                    continue
            
            # Store as search results
            st.session_state.search_results = {
                'results': all_stems,
                'total_found': len(all_stems),
                'search_time_ms': 0
            }
            st.session_state.last_query = "All Stems"
            st.session_state.last_search_type = "all"
            st.success(f"Loaded {len(all_stems)} stems")
            st.rerun()
            
    except Exception as e:
        st.error(f"Error loading stems: {str(e)}")

def render_results_header():
    """Render results header with summary"""
    results = st.session_state.search_results
    last_query = st.session_state.get('last_query', 'Unknown')
    search_type = st.session_state.get('last_search_type', 'text')
    
    search_results = results.get('results', [])
    total_found = results.get('total_found', len(search_results))
    search_time = results.get('search_time_ms', 0)
    
    # Header info
    col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
    
    with col1:
        st.subheader(f"🔍 Query: '{last_query}'")
        search_type_label = {
            'text': '📝 Text Search',
            'similarity': '🎵 Similarity Search', 
            'advanced': '⚙️ Advanced Search',
            'all': '📊 All Stems'
        }.get(search_type, '🔍 Search')
        st.caption(f"Search Type: {search_type_label}")
    
    with col2:
        st.metric("Results", total_found)
    
    with col3:
        st.metric("Search Time", f"{search_time:.0f}ms")
    
    with col4:
        if st.button("🔄 New Search", use_container_width=True):
            # Clear results and go to search page
            if 'search_results' in st.session_state:
                del st.session_state.search_results
            st.info("Go to Search page to perform a new search")
    
    # Quick stats
    if search_results:
        st.write("---")
        st.subheader("📊 Statistics")
        
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

def render_filter_controls():
    """Render filter controls for results"""
    results = st.session_state.search_results.get('results', [])
    
    if not results:
        return
    
    st.subheader("🔧 Filter Results")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Category filter
        categories = sorted(set(r.get('category', 'Unknown') for r in results))
        selected_categories = st.multiselect(
            "Categories",
            options=categories,
            default=categories,
            key="filter_categories"
        )
    
    with col2:
        # BPM filter
        bpm_values = [r.get('bpm') for r in results if r.get('bpm')]
        if bpm_values:
            min_bpm, max_bpm = min(bpm_values), max(bpm_values)
            bpm_range = st.slider(
                "BPM Range",
                min_value=int(min_bpm),
                max_value=int(max_bpm),
                value=(int(min_bpm), int(max_bpm)),
                key="filter_bpm"
            )
        else:
            bpm_range = None
    
    with col3:
        # Similarity filter
        similarity_values = [r.get('similarity_score', 0) for r in results]
        min_sim, max_sim = min(similarity_values), max(similarity_values)
        similarity_range = st.slider(
            "Similarity Range",
            min_value=float(min_sim),
            max_value=float(max_sim),
            value=(float(min_sim), float(max_sim)),
            step=0.01,
            key="filter_similarity"
        )
    
    with col4:
        # Sort options
        sort_by = st.selectbox(
            "Sort By",
            options=["Similarity", "BPM", "Category", "Filename"],
            key="sort_by"
        )
        
        sort_order = st.radio(
            "Order",
            options=["Descending", "Ascending"],
            key="sort_order"
        )
    
    # Apply filters
    filtered_results = results
    
    # Category filter
    if selected_categories:
        filtered_results = [r for r in filtered_results if r.get('category', 'Unknown') in selected_categories]
    
    # BPM filter
    if bpm_range:
        filtered_results = [r for r in filtered_results 
                          if r.get('bpm') and bpm_range[0] <= r.get('bpm') <= bpm_range[1]]
    
    # Similarity filter
    filtered_results = [r for r in filtered_results 
                       if similarity_range[0] <= r.get('similarity_score', 0) <= similarity_range[1]]
    
    # Sort results
    sort_key_map = {
        "Similarity": lambda x: x.get('similarity_score', 0),
        "BPM": lambda x: x.get('bpm', 0),
        "Category": lambda x: x.get('category', 'Unknown'),
        "Filename": lambda x: x.get('filename', '')
    }
    
    if sort_by in sort_key_map:
        reverse = sort_order == "Descending"
        filtered_results = sorted(filtered_results, key=sort_key_map[sort_by], reverse=reverse)
    
    # Store filtered results
    st.session_state.filtered_results = filtered_results
    
    st.info(f"Showing {len(filtered_results)} of {len(results)} results")

def render_results_grid():
    """Render results in grid format"""
    filtered_results = st.session_state.get('filtered_results', 
                                           st.session_state.search_results.get('results', []))
    
    if not filtered_results:
        st.warning("No results match the current filters.")
        return
    
    # Pagination
    total_results = len(filtered_results)
    total_pages = (total_results - 1) // results_per_page + 1
    
    if total_pages > 1:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            current_page = st.selectbox(
                "Page",
                options=list(range(1, total_pages + 1)),
                index=0,
                key="current_page"
            )
    else:
        current_page = 1
    
    # Calculate slice
    start_idx = (current_page - 1) * results_per_page
    end_idx = start_idx + results_per_page
    page_results = filtered_results[start_idx:end_idx]
    
    # Display format toggle
    col1, col2 = st.columns([1, 4])
    with col1:
        display_format = st.radio(
            "Display",
            options=["Cards", "Table"],
            key="display_format"
        )
    
    if display_format == "Cards":
        # Grid layout
        for i in range(0, len(page_results), 2):
            col1, col2 = st.columns(2)
            
            with col1:
                if i < len(page_results):
                    render_result_card(page_results[i], start_idx + i + 1)
            
            with col2:
                if i + 1 < len(page_results):
                    render_result_card(page_results[i + 1], start_idx + i + 2)
    else:
        render_results_table(page_results, start_idx)

def render_result_card(result: Dict[str, Any], rank: int):
    """Render individual result card"""
    with st.container():
        # Card header
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.write(f"**#{rank}. {result.get('filename', 'Unknown')}**")
        
        with col2:
            similarity = result.get('similarity_score', 0)
            color = "🟢" if similarity > 0.7 else "🟡" if similarity > 0.5 else "🔴"
            st.write(f"{color} {similarity:.3f}")
        
        # Card details
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"📂 **Category**: {result.get('category', 'N/A')}")
            st.write(f"🆔 **ID**: `{result.get('id', 'N/A')}`")
        
        with col2:
            bpm = result.get('bpm', 'N/A')
            st.write(f"🥁 **BPM**: {bpm}")
            
            duration = result.get('duration', 'N/A')
            if duration != 'N/A':
                duration = f"{duration:.1f}s"
            st.write(f"⏱️ **Duration**: {duration}")
        
        # Action buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button(f"🔍 Details", key=f"details_{result.get('id')}_{rank}"):
                st.session_state.selected_result = result
                st.session_state.show_details = True
        
        with col2:
            if st.button(f"🎵 Similar", key=f"similar_{result.get('id')}_{rank}"):
                find_similar_to_file(result.get('id'))
        
        with col3:
            if st.button(f"📥 Download", key=f"download_{result.get('id')}_{rank}"):
                st.info("Download functionality coming soon!")
        
        st.write("---")

def render_results_table(page_results: List[Dict[str, Any]], start_idx: int):
    """Render results in table format"""
    # Prepare data for table
    table_data = []
    for i, result in enumerate(page_results):
        table_data.append({
            "Rank": start_idx + i + 1,
            "Filename": result.get('filename', 'Unknown'),
            "Category": result.get('category', 'N/A'),
            "BPM": result.get('bpm', 'N/A'),
            "Duration": f"{result.get('duration', 0):.1f}s" if result.get('duration') else 'N/A',
            "Similarity": f"{result.get('similarity_score', 0):.3f}",
            "ID": result.get('id', 'N/A')
        })
    
    df = pd.DataFrame(table_data)
    
    # Display table
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )
    
    # Action buttons for selected rows
    st.subheader("Actions")
    selected_indices = st.multiselect(
        "Select results for actions",
        options=list(range(len(page_results))),
        format_func=lambda x: f"#{start_idx + x + 1}. {page_results[x].get('filename', 'Unknown')}"
    )
    
    if selected_indices:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔍 Show Details"):
                for idx in selected_indices:
                    st.session_state.selected_result = page_results[idx]
                    break  # Show details for first selected
        
        with col2:
            if st.button("🎵 Find Similar"):
                for idx in selected_indices:
                    find_similar_to_file(page_results[idx].get('id'))
                    break  # Find similar for first selected
        
        with col3:
            if st.button("📥 Export Selected"):
                selected_results = [page_results[idx] for idx in selected_indices]
                export_as_json(selected_results)

def find_similar_to_file(file_id: str):
    """Find files similar to the given file ID"""
    try:
        with st.spinner("Finding similar files..."):
            params = {
                'file_id': file_id,
                'limit': 10,
                'threshold': 0.5
            }
            
            response = requests.get(f"{backend_url}/api/v1/search/similar", params=params, timeout=30)
            
            if response.status_code == 200:
                results = response.json()
                
                # Store as new search results
                st.session_state.search_results = results
                st.session_state.last_query = f"Similar to {file_id}"
                st.session_state.last_search_type = "similarity"
                
                st.success(f"Found {len(results.get('results', []))} similar files")
                st.rerun()
            else:
                st.error(f"Similarity search failed: {response.text}")
                
    except Exception as e:
        st.error(f"Error finding similar files: {str(e)}")

def export_as_json(results: List[Dict[str, Any]]):
    """Export results as JSON"""
    try:
        export_data = {
            'export_timestamp': pd.Timestamp.now().isoformat(),
            'total_results': len(results),
            'query': st.session_state.get('last_query', 'Unknown'),
            'search_type': st.session_state.get('last_search_type', 'unknown'),
            'results': results
        }
        
        json_str = json.dumps(export_data, indent=2, ensure_ascii=False)
        
        st.download_button(
            label="📥 Download JSON",
            data=json_str,
            file_name=f"search_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
        
    except Exception as e:
        st.error(f"Export error: {str(e)}")

def export_as_csv(results: List[Dict[str, Any]]):
    """Export results as CSV"""
    try:
        # Flatten results for CSV
        csv_data = []
        for result in results:
            csv_data.append({
                'filename': result.get('filename', ''),
                'id': result.get('id', ''),
                'category': result.get('category', ''),
                'bpm': result.get('bpm', ''),
                'duration': result.get('duration', ''),
                'similarity_score': result.get('similarity_score', 0)
            })
        
        df = pd.DataFrame(csv_data)
        csv_str = df.to_csv(index=False)
        
        st.download_button(
            label="📥 Download CSV",
            data=csv_str,
            file_name=f"search_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
        
    except Exception as e:
        st.error(f"Export error: {str(e)}")

def render_analytics():
    """Render analytics section"""
    results = st.session_state.search_results.get('results', [])
    
    if not results:
        return
    
    st.header("📊 Analytics")
    
    # Category distribution
    categories = [r.get('category', 'Unknown') for r in results]
    category_counts = pd.Series(categories).value_counts()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Category Distribution")
        st.bar_chart(category_counts)
    
    with col2:
        st.subheader("Similarity Score Distribution")
        similarity_scores = [r.get('similarity_score', 0) for r in results]
        similarity_df = pd.DataFrame({'Similarity': similarity_scores})
        st.histogram_chart(similarity_df['Similarity'])
    
    # BPM analysis
    bpm_values = [r.get('bpm') for r in results if r.get('bpm')]
    if bpm_values:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("BPM Distribution")
            bpm_df = pd.DataFrame({'BPM': bpm_values})
            st.histogram_chart(bpm_df['BPM'])
        
        with col2:
            st.subheader("BPM Statistics")
            st.write(f"**Min BPM**: {min(bpm_values)}")
            st.write(f"**Max BPM**: {max(bpm_values)}")
            st.write(f"**Avg BPM**: {sum(bpm_values) / len(bpm_values):.1f}")
            st.write(f"**Median BPM**: {sorted(bpm_values)[len(bpm_values)//2]:.1f}")

# Main page content
st.title("🎯 Search Results")

# Check if we have results to display
if not st.session_state.get('search_results'):
    st.info("🔍 No search results available. Please perform a search first.")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("🔍 Go to Search", type="primary", use_container_width=True):
            st.info("Navigate to the Search page using the sidebar")
    
    with col2:
        if st.button("📊 Show All Stems", use_container_width=True):
            load_all_stems()
else:
    # Display results
    render_results_header()
    render_filter_controls()
    render_results_grid()
    
    # Export options
    st.header("📤 Export Options")
    
    filtered_results = st.session_state.get('filtered_results', 
                                           st.session_state.search_results.get('results', []))
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📥 Export as JSON", use_container_width=True):
            export_as_json(filtered_results)
    
    with col2:
        if st.button("📥 Export as CSV", use_container_width=True):
            export_as_csv(filtered_results)
    
    with col3:
        if st.button("📋 Copy File IDs", use_container_width=True):
            file_ids = [r.get('id', '') for r in filtered_results]
            ids_text = '\n'.join(file_ids)
            st.text_area("File IDs (copy this)", ids_text, height=100)
    
    # Analytics
    render_analytics()

# Show detailed view if selected
if st.session_state.get('show_details') and st.session_state.get('selected_result'):
    result = st.session_state.selected_result
    
    with st.expander(f"📋 Details: {result.get('filename', 'Unknown')}", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**File ID**: `{result.get('id', 'N/A')}`")
            st.write(f"**Filename**: {result.get('filename', 'N/A')}")
            st.write(f"**Category**: {result.get('category', 'N/A')}")
            st.write(f"**BPM**: {result.get('bpm', 'N/A')}")
        
        with col2:
            st.write(f"**Duration**: {result.get('duration', 'N/A')}s")
            st.write(f"**Similarity Score**: {result.get('similarity_score', 'N/A')}")
            
            # Additional metadata if available
            metadata = result.get('metadata', {})
            if metadata:
                for key, value in metadata.items():
                    st.write(f"**{key.title()}**: {value}")
        
        # Close button
        if st.button("❌ Close Details"):
            st.session_state.show_details = False
            st.rerun()