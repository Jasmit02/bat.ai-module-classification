"""
FastAPI application for the Module Classification System.
"""
from fastapi import FastAPI, HTTPException
from langserve import add_routes
import logging
import traceback
from typing import Dict, Any

from models import ClassificationRequest, ClassificationResponse
from chains import create_classification_chain, parse_classification_result
from routes import router as evaluation_router

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Module Classification API",
    description="API for classifying modules as script or product",
    version="1.0.0",
)

# Add LangServe routes for the classification chain
try:
    classification_chain = create_classification_chain()
    add_routes(
        app,
        classification_chain,
        path="/api/chain/classification",
    )
    logger.info("Classification chain successfully initialized")
except Exception as e:
    logger.error(f"Error initializing classification chain: {str(e)}")
    logger.error(traceback.format_exc())
    classification_chain = None

# Include evaluation routes
app.include_router(evaluation_router, prefix="/api", tags=["evaluation"])

@app.post("/api/classify", response_model=ClassificationResponse)
async def classify_module(request: ClassificationRequest):
    """
    Classify a module based on synopsis and description.
    
    Args:
        request: ClassificationRequest containing synopsis and description
        
    Returns:
        ClassificationResponse with classification details
    """
    if classification_chain is None:
        raise HTTPException(status_code=503, detail="Classification service is unavailable. Please check logs for details.")
        
    try:
        # Prepare the question
        if request.direct_query:
            question = request.direct_query
        else:
            question = f"Synopsis: {request.synopsis} Description: {request.description}"
        
        # Invoke the classification chain
        result = classification_chain.invoke({"question": question})
        
        # Parse the classification result
        classification_data = parse_classification_result(result["classification"])
        
        # Create response
        return ClassificationResponse(
            classification=classification_data["classification"],
            confidence=classification_data["confidence"],
            supporting_points=classification_data["supporting_points"],
            processing_details={
                "num_documents_retrieved": len(result.get("processed", {}).get("context", [])),
                "num_documents_after_reranking": len(result.get("processed", {}).get("reranked_context", []))
            }
        )
    except Exception as e:
        error_detail = str(e)
        logger.error(f"Error during classification: {error_detail}")
        logger.error(traceback.format_exc())
        
        # Provide a more user-friendly error message
        if "404" in error_detail and "ranking" in error_detail:
            raise HTTPException(
                status_code=503, 
                detail="NVIDIA reranking service is currently unavailable. Please try again later or contact the administrator."
            )
        else:
            raise HTTPException(status_code=500, detail=f"Classification failed: {error_detail}")

@app.get("/api/health")
async def health_check():
    """
    Health check endpoint to verify the API is running.
    
    Returns:
        Dict with status
    """
    status = "healthy" if classification_chain is not None else "degraded"
    return {"status": status, "details": "Classification service is available" if classification_chain else "Classification service unavailable"}