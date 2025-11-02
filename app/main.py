from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
import logging
from datetime import datetime
import json
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
        "personalization_enabled": recommender.students_df is not None or recommender.interactions_df is not None
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
