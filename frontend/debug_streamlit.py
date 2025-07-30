import requests
import time

def test_streamlit_app():
    """Test the Streamlit app directly"""
    url = "http://localhost:8501"
    
    print(f"Testing Streamlit app at {url}")
    
    try:
        # Test main page
        response = requests.get(url, timeout=10)
        print(f"Status Code: {response.status_code}")
        print(f"Content Length: {len(response.text)}")
        print(f"Content Type: {response.headers.get('content-type')}")
        
        # Check if it's the basic HTML
        if "<title>Streamlit</title>" in response.text:
            print("✅ Basic Streamlit HTML loaded")
        else:
            print("❌ No Streamlit title found")
            
        # Check for JavaScript
        if "static/js/" in response.text:
            print("✅ JavaScript files referenced")
        else:
            print("❌ No JavaScript files found")
            
        # Check for CSS
        if "static/css/" in response.text:
            print("✅ CSS files referenced")
        else:
            print("❌ No CSS files found")
            
        # Test if we can access the static files
        print("\nTesting static file access...")
        
        # Try to get the main JS file
        js_files = []
        css_files = []
        
        import re
        js_matches = re.findall(r'src="\./static/js/([^"]+)"', response.text)
        css_matches = re.findall(r'href="\./static/css/([^"]+)"', response.text)
        
        if js_matches:
            js_file = js_matches[0]
            js_url = f"{url}/static/js/{js_file}"
            js_response = requests.get(js_url, timeout=5)
            print(f"JS file {js_file}: {js_response.status_code}")
            
        if css_matches:
            css_file = css_matches[0]
            css_url = f"{url}/static/css/{css_file}"
            css_response = requests.get(css_url, timeout=5)
            print(f"CSS file {css_file}: {css_response.status_code}")
            
        print("\nFull HTML content (first 1000 chars):")
        print(response.text[:1000])
        
    except Exception as e:
        print(f"❌ Error testing app: {e}")

if __name__ == "__main__":
    test_streamlit_app()