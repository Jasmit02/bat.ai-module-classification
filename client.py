"""
Streamlit client for the Module Classification System.
"""
import streamlit as st
import requests
import json

# API endpoint
API_BASE_URL = "http://10.7.4.3:5051"

# Streamlit App Configuration
st.set_page_config(page_title="Module Classifier", layout="wide")

def main():
    """Main function for the classification client interface."""
    st.title("Module Classification System")
    
    # Input section
    st.subheader("Input Query")
    syn = st.text_input("Enter Synopsis:")
    des = st.text_input("Enter Description:")
    
    add_info = st.radio("If you want to add direct information like Synopsis+ Description as Query:", ["Yes", "No"], index=1)
    
    direct_query = None
    if add_info == "Yes":
        st.write("Enter your query:")
        direct_query = st.text_area("Enter your query:", height=200)
    
    if st.button("Classify Module"):
        # Validate input
        if (not syn.strip() and not des.strip() and not direct_query) or (add_info == "Yes" and not direct_query):
            st.warning("Please enter a query.")
            return
        
        with st.spinner("Processing..."):
            try:
                # Prepare request payload
                payload = {
                    "synopsis": syn,
                    "description": des,
                    "direct_query": direct_query
                }
                
                # Call API
                response = requests.post(f"{API_BASE_URL}/api/classify", json=payload)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Display results
                    st.subheader("Classification Results")
                    st.markdown(f"**Classification:** {result['classification']}")
                    st.markdown(f"**Confidence:** {result['confidence']}")
                    
                    st.markdown("**Key Supporting Points:**")
                    for point in result['supporting_points']:
                        st.markdown(f"• {point}")
                    
                    # Show processing details
                    with st.expander("Processing Details"):
                        st.write(f"Retrieved {result['processing_details']['num_documents_retrieved']} documents")
                        st.write(f"After reranking: {result['processing_details']['num_documents_after_reranking']} documents")
                else:
                    st.error(f"Error: {response.status_code} - {response.text}")
            except Exception as e:
                st.error(f"Error during classification: {str(e)}")

if __name__ == "__main__":
    main()