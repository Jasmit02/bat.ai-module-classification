"""
Streamlit client for evaluating the Module Classification System.
"""
import streamlit as st
import pandas as pd
import requests
import random
import os

# API endpoint
API_BASE_URL = os.getenv("BASE_URL")


# Streamlit App Configuration
st.set_page_config(page_title="Module Classifier Evaluation", layout="wide")

def main():
    """Main function for the evaluation client interface."""
    st.title("Module Classification System - Evaluation")
    
    try:
        # Load dataset
        df = pd.read_csv("eval.csv")
        
        if st.checkbox("Show sample data"):
            st.write(df.head())
            
        # Evaluation parameters
        sample_size = st.number_input("Number of samples to evaluate", min_value=1, max_value=len(df), value=10)
        random_state = st.number_input("Random seed", value=random.randint(1, 200))
        num_threads = st.slider("Number of threads", min_value=1, max_value=50, value=10)
        
        # Retrieval method options
        retrieval_method = st.radio(
            "Select Retrieval Method for Evaluation",
            [
                "Full Hybrid (Parent + Semantic)", 
                "Parent Only", 
                "Semantic Only"
            ]
        )
        
        # MultiQuery options
        use_multi_query = st.radio(
            "Use MultiQuery Retriever for Evaluation",
            ["Yes", "No"],
            index=1
        )
        
        num_alt_queries = 3
        if use_multi_query == "Yes":
            num_alt_queries = st.slider("Number of alternative queries", min_value=2, max_value=10, value=3)
        
        # Show document count
        st.write(f"Total documents in dataset: {len(df)}")
        st.write(f"Documents to evaluate: {sample_size}")
        
        if st.button("Run Evaluation"):
            with st.spinner("Running evaluation..."):
                try:
                    # Generate random sample for evaluation
                    random_rows = df.sample(n=sample_size, random_state=random_state)
                    
                    # Prepare evaluation request payload
                    request_data = {
                        "samples": random_rows.to_dict('records'),
                        "sample_size": sample_size,
                        "random_seed": random_state,
                        "num_threads": num_threads,
                        "use_multi_query": use_multi_query == "Yes",
                        "num_alt_queries": num_alt_queries,
                        "retrieval_method": retrieval_method
                    }
                    
                    # Create a progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Process samples one by one (the API would ideally handle batch processing)
                    results = []
                    correct = 0
                    total = 0
                    
                    for i, (idx, row) in enumerate(random_rows.iterrows()):
                        status_text.text(f"Processing sample {i+1}/{sample_size}")
                        progress_bar.progress((i+1)/sample_size)
                        
                        # Prepare the query
                        synopsis = str(row['Synopsis'])
                        description = str(row['Description'])
                        true_label = row['Module']
                        
                        # Call classification API
                        payload = {
                            "synopsis": synopsis,
                            "description": description
                        }
                        
                        response = requests.post(f"{API_BASE_URL}/api/classify", json=payload)
                        
                        if response.status_code == 200:
                            result_data = response.json()
                            prediction = result_data['classification'].lower()
                            match = true_label.lower() == prediction.lower()
                            
                            if match:
                                correct += 1
                            total += 1
                            
                            results.append({
                                'Index': idx,
                                'Question': f"{synopsis} {description}",
                                'True Label': true_label,
                                'Prediction': prediction,
                                'Confidence': result_data['confidence'],
                                'Match': match
                            })
                        else:
                            st.error(f"Error processing sample {idx}: {response.text}")
                    
                    # Calculate accuracy
                    if total > 0:
                        accuracy = correct / total
                    else:
                        accuracy = 0
                    
                    # Show results
                    st.subheader("Evaluation Results")
                    st.write(f"Accuracy: {accuracy:.2%} ({correct}/{total})")
                    
                    # Show detailed results
                    st.write("Detailed Results:")
                    results_df = pd.DataFrame(results)
                    st.dataframe(results_df)
                    
                    # Download results
                    csv = results_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "Download Evaluation Results",
                        csv,
                        "evaluation_results.csv",
                        "text/csv",
                        key='download-csv'
                    )
                    
                except Exception as e:
                    st.error(f"Error during evaluation: {str(e)}")
                    
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")

if __name__ == "__main__":
    main()