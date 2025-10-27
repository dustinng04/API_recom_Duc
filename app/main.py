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
            # Load and train with default data
            data_path = Path("data/tutors_adjust.json")
            if data_path.exists():
                logger.info("No existing model found. Training with default data...")
                with open(data_path, 'r', encoding='utf-8') as f:
                    tutors_data = json.load(f)
                recommender.train(tutors_data)
                recommender.save_model("models/recommender.pkl")
                logger.info(f"Model trained with {len(tutors_data)} tutors")
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
        "tutors_count": len(recommender.tutors_df) if recommender.tutors_df is not None else 0
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
