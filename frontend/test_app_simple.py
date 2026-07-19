import streamlit as st

# Simple test app to verify Streamlit is working
st.set_page_config(
    page_title="Test App",
    page_icon="🧪",
    layout="wide"
)

st.title("🧪 Test App")
st.write("This is a simple test to verify Streamlit is working correctly.")

# Simple sidebar
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose page:", ["Home", "Test 1", "Test 2"])

if page == "Home":
    st.write("Welcome to the home page!")
    st.success("✅ Streamlit is working correctly!")
elif page == "Test 1":
    st.write("This is Test Page 1")
    st.info("ℹ️ Navigation is working!")
else:
    st.write("This is Test Page 2")
    st.warning("⚠️ All components are functional!")

st.sidebar.markdown("---")
st.sidebar.write("Sidebar is working correctly!")