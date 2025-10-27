import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class TutorRecommender:
    def __init__(self):
        self.tutors_df = None
        self.features = None
        self.similarity_matrix = None
        self.scaler = StandardScaler()
        self.mlb_subjects = MultiLabelBinarizer()
        self.mlb_styles = MultiLabelBinarizer()

    def train(self, tutors_data: List[Dict]):
        logger.info("Starting training...")
        self.tutors_df = pd.DataFrame(tutors_data)
        self._prepare_features()
        self._compute_similarity()
        logger.info("Training completed")

    def _prepare_features(self):
        logger.info("Preparing features...")
        df = self.tutors_df.copy()

        numerical_features = []

        # Price (normalized)
        if df['price'].max() > df['price'].min():
            price_norm = (df['price'] - df['price'].min()) / (df['price'].max() - df['price'].min())
        else:
            price_norm = pd.Series([0.5] * len(df))
        numerical_features.append(price_norm.values.reshape(-1, 1))

        # Rating (normalized)
        rating_norm = df['rating'] / 5.0
        numerical_features.append(rating_norm.values.reshape(-1, 1))

        # Experience (normalized)
        exp_norm = df['yearExperience'] / df['yearExperience'].max()
        numerical_features.append(exp_norm.values.reshape(-1, 1))

        # Students count (log normalized)
        students_log = np.log1p(df['studentsCount'])
        students_norm = students_log / students_log.max()
        numerical_features.append(students_norm.values.reshape(-1, 1))

        # Number of reviews (log normalized)
        reviews_log = np.log1p(df['numberReviews'])
        reviews_norm = reviews_log / reviews_log.max()
        numerical_features.append(reviews_norm.values.reshape(-1, 1))

        # Total lessons (log normalized)
        lessons_log = np.log1p(df['totalLessons'])
        lessons_norm = lessons_log / lessons_log.max()
        numerical_features.append(lessons_norm.values.reshape(-1, 1))

        # Professional flag
        is_prof = df['isProfessional'].astype(int).values.reshape(-1, 1)
        numerical_features.append(is_prof)

        # Categorical features - Subjects
        subject_ids = df['subject'].apply(
            lambda x: [s['id'] for s in x] if isinstance(x, list) else []
        )
        subject_features = self.mlb_subjects.fit_transform(subject_ids)

        # Categorical features - Teaching styles
        teaching_styles = df['teachingStyle'].apply(
            lambda x: [s['en'] for s in x] if isinstance(x, list) else []
        )
        style_features = self.mlb_styles.fit_transform(teaching_styles)

        # Combine all features
        numerical_array = np.hstack(numerical_features)
        self.features = np.hstack([numerical_array, subject_features, style_features])

        logger.info(f"Features prepared: {self.features.shape}")

    def _compute_similarity(self):
        logger.info("Computing similarity matrix...")
        features_scaled = self.scaler.fit_transform(self.features)
        self.similarity_matrix = cosine_similarity(features_scaled)
        logger.info(f"Similarity matrix computed: {self.similarity_matrix.shape}")

    def recommend(self, user_id: str, top_k: int = 10) -> List[int]:
        if self.tutors_df is None:
            raise ValueError("Model not trained")

        df = self.tutors_df.copy()

        # Quality score
        rating_score = df['rating'] / 5.0
        exp_score = np.minimum(df['yearExperience'] / 10.0, 1.0)
        students_score = np.log1p(df['studentsCount']) / np.log1p(df['studentsCount'].max())
        reviews_score = np.log1p(df['numberReviews']) / np.log1p(df['numberReviews'].max())
        prof_bonus = df['isProfessional'].astype(float) * 0.1

        quality_score = (
            0.3 * rating_score +
            0.2 * exp_score +
            0.2 * students_score +
            0.2 * reviews_score +
            0.1 * prof_bonus
        )

        # Get top K tutors
        top_indices = quality_score.nlargest(top_k).index
        tutor_ids = df.loc[top_indices, 'id'].tolist()

        return tutor_ids

    def save_model(self, path: str):
        logger.info(f"Saving model to {path}")
        with open(path, 'wb') as f:
            pickle.dump({
                'tutors_df': self.tutors_df,
                'features': self.features,
                'similarity_matrix': self.similarity_matrix,
                'scaler': self.scaler,
                'mlb_subjects': self.mlb_subjects,
                'mlb_styles': self.mlb_styles
            }, f)
        logger.info("Model saved successfully")

    def load_model(self, path: str):
        logger.info(f"Loading model from {path}")
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.tutors_df = data['tutors_df']
            self.features = data['features']
            self.similarity_matrix = data['similarity_matrix']
            self.scaler = data['scaler']
            self.mlb_subjects = data['mlb_subjects']
            self.mlb_styles = data['mlb_styles']
        logger.info(f"Model loaded: {len(self.tutors_df)} tutors")
