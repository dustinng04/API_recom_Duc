from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import logging
from datetime import datetime
import json
import numpy as np
from pathlib import Path

from .recommender import TutorRecommender

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Tutor Recommendation System",
    description="AI-powered tutor recommendation API",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global recommender instance
recommender = None

# Pydantic models
class RecommendRequest(BaseModel):
    user_ids: List[str]
    k: int = 10

class UserRecommendation(BaseModel):
    user_id: str
    recommended_tutor_ids: List[int]

class RecommendResponse(BaseModel):
    data: List[UserRecommendation]

class MetricsResponse(BaseModel):
    metrics: Dict
    description: str
    top_k: int


class RerankCandidate(BaseModel):
    tutorId: int
    os_score: float


class RerankRequest(BaseModel):
    user_id: Optional[str] = None
    query_vector: List[float]
    candidates: List[RerankCandidate]


class RerankResponse(BaseModel):
    # Keep response minimal and aligned with candidates order for easy merging on caller side
    scores: List[float]


# Startup event
@app.on_event("startup")
async def startup_event():
    global recommender
    try:
        logger.info("Loading tutor recommendation model...")
        recommender = TutorRecommender()

        # Try to load existing model
        model_path = Path("models/recommender.pkl")
        if model_path.exists():
            recommender.load_model("models/recommender.pkl")
            logger.info("Existing model loaded successfully")
            
            # Try to load reranker model from separate file if not already loaded from recommender.pkl
            # Note: After merging, reranker.pkl can be deleted as it's now in recommender.pkl
            if recommender.reranker_model is None:
                reranker_path = Path("models/reranker.pkl")
                if reranker_path.exists():
                    logger.info("Reranker model not found in recommender.pkl. Attempting to load from separate file...")
                    logger.info("Tip: Run 'python scripts/merge_reranker.py' to merge reranker into recommender.pkl")
                    try:
                        recommender.load_reranker_model("models/reranker.pkl")
                    except Exception as e:
                        logger.warning(f"Failed to load reranker from separate file: {e}")
                        logger.info("Reranker model will not be available. API /rerank-new will use fallback method.")
        else:
            # Load and train with all data sources
            tutors_path = Path("data/tutors_adjust.json")
            students_path = Path("data/students_output.json")
            interactions_path = Path("data/interaction_logs.jsonl")
            
            if tutors_path.exists():
                logger.info("No existing model found. Training with all available data...")
                
                # Load tutors
                with open(tutors_path, 'r', encoding='utf-8') as f:
                    tutors_data = json.load(f)
                
                # Load students if available
                students_data = None
                if students_path.exists():
                    with open(students_path, 'r', encoding='utf-8') as f:
                        students_data = json.load(f)
                    logger.info(f"Loaded {len(students_data)} students")
                
                # Load interactions if available
                interactions_data = None
                if interactions_path.exists():
                    with open(interactions_path, 'r', encoding='utf-8') as f:
                        interactions_data = [json.loads(line) for line in f]
                    logger.info(f"Loaded {len(interactions_data)} interactions")
                
                # Train with all data
                recommender.train(tutors_data, students_data, interactions_data)
                recommender.save_model("models/recommender.pkl")
                logger.info(f"Model trained with {len(tutors_data)} tutors, "
                          f"{len(students_data) if students_data else 0} students, "
                          f"{len(interactions_data) if interactions_data else 0} interactions")
            else:
                logger.warning("No data file found. Model will be empty until /train is called.")
        
        # Check reranker model status
        if recommender.reranker_model is not None:
            logger.info("Reranker model is available. /rerank-new endpoint is ready.")
        else:
            logger.info("No reranker model found. /rerank-new endpoint will use fallback method.")
            logger.info("Train a reranker model using: python scripts/train_reranker.py")

    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise


# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Tutor Recommendation System",
        "version": "1.0.0",
        "status": "running"
    }


# Health check
@app.get("/health")
async def health_check():
    if recommender is None:
        return {
            "status": "unhealthy",
            "error": "Model not loaded"
        }

    return {
        "status": "healthy",
        "model_loaded": True,
        "tutors_count": len(recommender.tutors_df) if recommender.tutors_df is not None else 0,
        "students_count": len(recommender.students_df) if recommender.students_df is not None else 0,
        "interactions_count": len(recommender.interactions_df) if recommender.interactions_df is not None else 0,
        "personalization_enabled": recommender.students_df is not None or recommender.interactions_df is not None,
        "reranker_model_loaded": recommender.reranker_model is not None
    }


# Train endpoint
@app.post("/train")
async def train_model(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        tutors_data = json.loads(contents)

        if not tutors_data or len(tutors_data) == 0:
            raise HTTPException(status_code=400, detail="Empty data file")

        logger.info(f"Received {len(tutors_data)} tutors for training")

        global recommender
        if recommender is None:
            recommender = TutorRecommender()

        recommender.train(tutors_data)
        recommender.save_model("models/recommender.pkl")

        return {
            "success": True,
            "message": "Model trained successfully",
            "tutors_count": len(tutors_data)
        }

    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON format")
    except Exception as e:
        logger.error(f"Training error: {e}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


# Get recommendations endpoint
@app.post("/get_recommended", response_model=RecommendResponse)
async def get_recommendations(request: RecommendRequest):
    try:
        if recommender is None or recommender.tutors_df is None:
            raise HTTPException(
                status_code=503,
                detail="Model not loaded. Please train the model first using /train endpoint"
            )

        logger.info(f"Processing recommendations for {len(request.user_ids)} users")

        # Get recommendations for each user
        results = []
        for user_id in request.user_ids:
            try:
                tutor_ids = recommender.recommend(user_id, top_k=request.k)
                results.append(UserRecommendation(
                    user_id=user_id,
                    recommended_tutor_ids=tutor_ids
                ))
            except Exception as e:
                logger.error(f"Error recommending for user {user_id}: {e}")
                results.append(UserRecommendation(
                    user_id=user_id,
                    recommended_tutor_ids=[]
                ))

        return RecommendResponse(data=results)

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Recommendation error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate recommendations: {str(e)}"
        )


# Re-rank endpoint (minimal response: scores aligned with input order)
@app.post("/rerank", response_model=RerankResponse)
async def rerank(request: RerankRequest):
    try:
        if recommender is None or recommender.tutors_df is None:
            raise HTTPException(
                status_code=503,
                detail="Model not loaded. Please train the model first using /train endpoint"
            )

        # Extract tutor ids in the same order as provided
        input_ids = [c.tutorId for c in request.candidates]

        # Map tutorId -> os_score
        os_map = {c.tutorId: c.os_score for c in request.candidates}

        # Personalization scores (raw, not normalized)
        rec_scores = recommender.score_candidates(user_id=request.user_id, tutor_ids=input_ids)

        # Build arrays aligned with input order
        import numpy as np
        os_arr = np.array([os_map.get(tid, 0.0) for tid in input_ids], dtype=float)
        rec_arr = np.array([rec_scores.get(tid, 0.0) for tid in input_ids], dtype=float)

        # Per-request min-max normalization to [0,1]
        def min_max(x: np.ndarray) -> np.ndarray:
            x_min = float(x.min()) if x.size else 0.0
            x_max = float(x.max()) if x.size else 0.0
            if x_max > x_min:
                return (x - x_min) / (x_max - x_min)
            return np.zeros_like(x)

        os_norm = min_max(os_arr)
        rec_norm = min_max(rec_arr)

        # Note: query_vector is accepted for future use (query-tutor similarity),
        # but intentionally unused in this minimal version for simplicity.
        # We'll incorporate it later without changing the API contract.

        # Weighted combination (tunable per business needs)
        w_os, w_rec = 0.6, 0.4
        final = w_os * os_norm + w_rec * rec_norm

        return RerankResponse(scores=[float(s) for s in final.tolist()])

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Rerank error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to rerank: {str(e)}"
        )


# Re-rank endpoint using trained LightGBMRanker model
@app.post("/rerank-new", response_model=RerankResponse)
async def rerank_new(request: RerankRequest):
    """
    Re-rank candidates using trained LightGBMRanker model.
    This endpoint uses a machine learning model trained on historical user interactions
    to predict optimal ranking scores.
    
    This endpoint REQUIRES a trained reranker model. If model is not available, returns 503 error.
    Use /rerank endpoint for weighted combination method.
    """
    try:
        if recommender is None or recommender.tutors_df is None:
            raise HTTPException(
                status_code=503,
                detail="Recommender model not loaded. Please train the model first using /train endpoint"
            )
        
        # Check if reranker model is available
        if recommender.reranker_model is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Reranker model not available. Please train the reranker model first using: "
                    "python scripts/train_reranker.py. "
                    "Alternatively, use /rerank endpoint for weighted combination method."
                )
            )
        
        # Extract tutor ids in the same order as provided
        input_ids = [c.tutorId for c in request.candidates]
        
        if not input_ids:
            return RerankResponse(scores=[])
        
        # Get personalization scores (rerank_score feature)
        rec_scores = recommender.score_candidates(user_id=request.user_id, tutor_ids=input_ids)
        
        # Extract os_score from request candidates
        os_map = {c.tutorId: c.os_score for c in request.candidates}
        
        # Predict using reranker model
        try:
            scores = recommender.predict_rerank_scores(
                tutor_ids=input_ids,
                rerank_scores=rec_scores,
                os_scores=os_map
            )
            return RerankResponse(scores=scores)
        except Exception as e:
            logger.error(f"Reranker model prediction failed: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Failed to predict rerank scores: {str(e)}. Please check model and input data."
            )
    
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Rerank-new error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to rerank: {str(e)}"
        )


# Get metrics endpoint
@app.get("/get_metrics", response_model=MetricsResponse)
async def get_metrics(top_k: int = 10):
    """
    Evaluate recommendation system performance.
    
    Metrics:
    - precision@k: % of recommended items that are relevant
    - recall@k: % of relevant items that are recommended
    - ndcg@k: Normalized Discounted Cumulative Gain (ranking quality)
    - coverage: % of tutors recommended at least once
    
    Args:
        top_k: Number of recommendations to evaluate (default 10)
    """
    try:
        if recommender is None or recommender.tutors_df is None:
            raise HTTPException(
                status_code=503,
                detail="Model not loaded. Please train the model first using /train endpoint"
            )
        
        if recommender.interactions_df is None or len(recommender.interactions_df) == 0:
            raise HTTPException(
                status_code=400,
                detail="No interaction data available for evaluation. Metrics require interaction history."
            )
        
        logger.info(f"Evaluating metrics with top_k={top_k}")
        
        # Evaluate using training data
        metrics = recommender.evaluate_metrics(top_k=top_k)
        
        description = (
            f"Evaluated on {metrics['users_evaluated']} users with interaction history. "
            f"Metrics show how well the system recommends tutors that users actually engaged with."
        )
        
        return MetricsResponse(
            metrics=metrics,
            description=description,
            top_k=top_k
        )
    
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Metrics evaluation error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to evaluate metrics: {str(e)}"
        )
