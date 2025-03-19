"""
API routes for the Module Classification System.
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks
import pandas as pd
import concurrent.futures
from typing import List, Dict, Any
import logging

from models import EvaluationRequest, EvaluationResponse, EvaluationResult
from chains import create_classification_chain, parse_classification_result

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_model(request: EvaluationRequest, background_tasks: BackgroundTasks):
    """
    Evaluate the model's performance on a dataset.
    
    Args:
        request: Evaluation parameters
        background_tasks: FastAPI background tasks
        
    Returns:
        EvaluationResponse with accuracy and results
    """
    try:
        # Load dataset
        df = pd.read_csv('eval.csv')
        
        # Sample the dataset
        random_rows = df.sample(n=request.sample_size, random_state=request.random_seed)
        
        # Create the classification chain
        classification_chain = create_classification_chain()
        
        results = []
        correct = 0
        total = 0
        
        # Define the function to process a single sample
        def process_sample(idx_row):
            """Process a single sample for evaluation."""
            idx, row = idx_row
            try:
                question = str(row['Synopsis']) + " " + str(row['Description'])
                true_label = row['Module']
                
                # Classify the sample
                result = classification_chain.invoke({"question": question})
                classification_data = parse_classification_result(result["classification"])
                
                prediction = classification_data["classification"].lower()
                match = true_label.lower() == prediction.lower()
                
                return {
                    'index': idx,
                    'question': question,
                    'true_label': true_label,
                    'prediction': prediction,
                    'match': match
                }
            except Exception as e:
                logger.error(f"Error processing sample {idx}: {str(e)}")
                return {
                    'index': idx,
                    'error': str(e),
                    'match': False
                }
        
        # Process samples in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=request.num_threads) as executor:
            futures = {executor.submit(process_sample, (idx, row)): idx for idx, row in random_rows.iterrows()}
            
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if 'error' not in result:
                    results.append(EvaluationResult(**result))
                    if result['match']:
                        correct += 1
                    total += 1
                else:
                    results.append(EvaluationResult(
                        index=result['index'],
                        question="",
                        true_label="",
                        prediction="",
                        match=False,
                        error=result['error']
                    ))
        
        # Calculate accuracy
        accuracy = correct / total if total > 0 else 0
        
        return EvaluationResponse(
            accuracy=accuracy,
            correct=correct,
            total=total,
            results=results
        )
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")