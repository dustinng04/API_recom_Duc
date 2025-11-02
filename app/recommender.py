import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from typing import List, Dict, Optional, Tuple, Set
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class TutorRecommender:
    def __init__(self):
        self.tutors_df = None
        self.students_df = None
        self.interactions_df = None
        self.features = None
        self.similarity_matrix = None
        self.scaler = StandardScaler()
        self.mlb_subjects = MultiLabelBinarizer()
        self.mlb_styles = MultiLabelBinarizer()
        self.student_tutor_matrix = None
        self.weights = {
            'content': 0.30,
            'collaborative': 0.20,
            'behavioral': 0.25,
            'quality': 0.15,
            'popularity': 0.10
        }

    def train(
        self, 
        tutors_data: List[Dict],
        students_data: Optional[List[Dict]] = None,
        interactions_data: Optional[List[Dict]] = None
    ):
        logger.info("Starting training...")
        self.tutors_df = pd.DataFrame(tutors_data)
        self._prepare_features()
        self._compute_similarity()
        
        if students_data:
            logger.info(f"Loading {len(students_data)} students...")
            self.students_df = pd.DataFrame(students_data)
            self._prepare_student_features()
        
        if interactions_data:
            logger.info(f"Loading {len(interactions_data)} interactions...")
            self.interactions_df = pd.DataFrame(interactions_data)
            self._prepare_interaction_matrix()
        
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
    
    def _prepare_student_features(self):
        logger.info("Preparing student features...")
        df = self.students_df.copy()
        df['teaching_styles_list'] = df['preferredTeachingStyles'].apply(
            lambda x: x if isinstance(x, list) else []
        )
        df['subject_ids'] = df['subjectPreferences'].apply(
            lambda x: [str(s['id']) for s in x] if isinstance(x, list) else []
        )
        self.students_df = df
        logger.info(f"Student features prepared for {len(df)} students")
    
    def _prepare_interaction_matrix(self):
        logger.info("Preparing interaction matrix...")
        event_weights = {
            'view': 0.1,
            'click': 0.2,
            'wishlist': 0.4,
            'join': 0.6,
            'conversion': 0.9,
            'rating': 1.0
        }
        
        matrix_data = defaultdict(lambda: defaultdict(float))
        
        for _, interaction in self.interactions_df.iterrows():
            user_id = interaction['userId']
            tutor_id = interaction['tutorId']
            event_type = interaction['eventType']
            
            weight = event_weights.get(event_type, 0)
            
            if event_type == 'rating' and 'value' in interaction:
                weight *= (interaction['value'] / 5.0)
            
            matrix_data[user_id][tutor_id] += weight
        
        # Convert defaultdict to regular dict for pickling
        self.student_tutor_matrix = {k: dict(v) for k, v in matrix_data.items()}
        logger.info(f"Interaction matrix built: {len(matrix_data)} users")
    
    def _find_similar_students(self, target_student: pd.Series, top_k: int = 5) -> List[Tuple[str, float]]:
        if self.students_df is None or len(self.students_df) == 0:
            return []
        
        similarities = []
        target_age = target_student.get('age', 0)
        target_level = target_student.get('schoolLevel', '')
        target_styles = set(target_student.get('teaching_styles_list', []))
        target_subjects = set(target_student.get('subject_ids', []))
        
        for _, student in self.students_df.iterrows():
            if student['id'] == target_student['id']:
                continue
            
            score = 0.0
            
            if student.get('schoolLevel') == target_level:
                score += 0.3
            
            age_diff = abs(student.get('age', 0) - target_age)
            score += max(0, 0.2 - age_diff * 0.05)
            
            student_styles = set(student.get('teaching_styles_list', []))
            if target_styles and student_styles:
                overlap = len(target_styles & student_styles) / len(target_styles | student_styles)
                score += 0.3 * overlap
            
            student_subjects = set(student.get('subject_ids', []))
            if target_subjects and student_subjects:
                overlap = len(target_subjects & student_subjects) / len(target_subjects | student_subjects)
                score += 0.2 * overlap
            
            similarities.append((student['id'], score))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def recommend(self, user_id: Optional[str] = None, top_k: int = 10) -> List[int]:
        if self.tutors_df is None:
            raise ValueError("Model not trained")

        df = self.tutors_df.copy()
        tutor_scores = {}
        
        # Content-based quality score
        rating_score = df['rating'] / 5.0
        exp_score = np.minimum(df['yearExperience'] / 10.0, 1.0)
        students_score = np.log1p(df['studentsCount']) / np.log1p(df['studentsCount'].max())
        reviews_score = np.log1p(df['numberReviews']) / np.log1p(df['numberReviews'].max())
        prof_bonus = df['isProfessional'].astype(float) * 0.1

        base_scores = (
            0.3 * rating_score +
            0.2 * exp_score +
            0.2 * students_score +
            0.2 * reviews_score +
            0.1 * prof_bonus
        )
        
        for idx, tutor_id in enumerate(df['id']):
            tutor_scores[tutor_id] = self.weights['content'] * base_scores.iloc[idx]
        
        # Personalization if user_id provided
        if user_id and self.students_df is not None:
            student = self.students_df[self.students_df['id'] == user_id]
            if not student.empty:
                student_row = student.iloc[0]
                student_styles = set(student_row.get('teaching_styles_list', []))
                
                # Teaching style matching
                for idx, row in df.iterrows():
                    tutor_id = row['id']
                    tutor_styles = set([s['en'] for s in row['teachingStyle']] if isinstance(row['teachingStyle'], list) else [])
                    
                    if student_styles and tutor_styles:
                        overlap = len(student_styles & tutor_styles) / len(student_styles | tutor_styles)
                        tutor_scores[tutor_id] += self.weights['quality'] * overlap
                
                # Collaborative filtering
                similar_students = self._find_similar_students(student_row, top_k=5)
                for similar_id, similarity in similar_students:
                    similar_student = self.students_df[self.students_df['id'] == similar_id]
                    if similar_student.empty:
                        continue
                    
                    enrollments = similar_student.iloc[0].get('enrollments', [])
                    if isinstance(enrollments, list):
                        for enrollment in enrollments:
                            if enrollment.get('status') == 'completed':
                                tutor_id = enrollment['tutorId']
                                if tutor_id in tutor_scores:
                                    tutor_scores[tutor_id] += self.weights['collaborative'] * 0.5 * similarity
                    
                    ratings = similar_student.iloc[0].get('tutorRatings', [])
                    if isinstance(ratings, list):
                        for rating in ratings:
                            if rating.get('rating', 0) >= 4.0:
                                tutor_id = rating['tutorId']
                                if tutor_id in tutor_scores:
                                    tutor_scores[tutor_id] += self.weights['collaborative'] * 0.3 * similarity * (rating['rating'] / 5.0)
                
                # Behavioral scoring
                if self.student_tutor_matrix and user_id in self.student_tutor_matrix:
                    for tutor_id, behavior_score in self.student_tutor_matrix[user_id].items():
                        if tutor_id in tutor_scores:
                            tutor_scores[tutor_id] += self.weights['behavioral'] * behavior_score

        # Sort and return top K
        sorted_tutors = sorted(tutor_scores.items(), key=lambda x: x[1], reverse=True)
        return [tutor_id for tutor_id, _ in sorted_tutors[:top_k]]

    def save_model(self, path: str):
        logger.info(f"Saving model to {path}")
        with open(path, 'wb') as f:
            pickle.dump({
                'tutors_df': self.tutors_df,
                'students_df': self.students_df,
                'interactions_df': self.interactions_df,
                'features': self.features,
                'similarity_matrix': self.similarity_matrix,
                'scaler': self.scaler,
                'mlb_subjects': self.mlb_subjects,
                'mlb_styles': self.mlb_styles,
                'student_tutor_matrix': self.student_tutor_matrix,
                'weights': self.weights
            }, f)
        logger.info("Model saved successfully")

    def load_model(self, path: str):
        logger.info(f"Loading model from {path}")
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.tutors_df = data['tutors_df']
            self.students_df = data.get('students_df')
            self.interactions_df = data.get('interactions_df')
            self.features = data['features']
            self.similarity_matrix = data['similarity_matrix']
            self.scaler = data['scaler']
            self.mlb_subjects = data['mlb_subjects']
            self.mlb_styles = data['mlb_styles']
            self.student_tutor_matrix = data.get('student_tutor_matrix')
            self.weights = data.get('weights', self.weights)
        logger.info(f"Model loaded: {len(self.tutors_df)} tutors")
    
    def evaluate_metrics(self, test_interactions: Optional[List[Dict]] = None, top_k: int = 10) -> Dict:
        """
        Evaluate recommendation quality using precision, recall, NDCG, and coverage.
        
        Args:
            test_interactions: List of test interactions (if None, uses training interactions)
            top_k: Number of recommendations to evaluate
            
        Returns:
            Dictionary with metrics
        """
        if self.tutors_df is None:
            raise ValueError("Model not trained")
        
        # Use training interactions if no test set provided
        interactions = test_interactions if test_interactions else (
            self.interactions_df.to_dict('records') if self.interactions_df is not None else []
        )
        
        if not interactions:
            logger.warning("No interactions available for evaluation")
            return {
                'precision@k': 0.0,
                'recall@k': 0.0,
                'ndcg@k': 0.0,
                'coverage': 0.0,
                'users_evaluated': 0
            }
        
        # Build ground truth set (tutors each user actually interacted with positively)
        ground_truth = defaultdict(set)
        for interaction in interactions:
            user_id = interaction['userId']
            tutor_id = interaction['tutorId']
            event_type = interaction['eventType']
            
            # Consider positive interactions (not just views)
            if event_type in ['click', 'wishlist', 'join', 'conversion', 'rating']:
                ground_truth[user_id].add(tutor_id)
        
        # Evaluate recommendations for each user
        precisions = []
        recalls = []
        ndcgs = []
        all_recommended = set()
        
        for user_id, relevant_items in ground_truth.items():
            if len(relevant_items) == 0:
                continue
            
            # Get recommendations
            try:
                recommended = self.recommend(user_id=user_id, top_k=top_k)
                recommended_set = set(recommended)
                all_recommended.update(recommended_set)
                
                # Calculate hits
                hits = recommended_set & relevant_items
                
                # Precision@K
                precision = len(hits) / len(recommended) if recommended else 0.0
                precisions.append(precision)
                
                # Recall@K
                recall = len(hits) / len(relevant_items)
                recalls.append(recall)
                
                # NDCG@K
                dcg = 0.0
                idcg = 0.0
                for i, tutor_id in enumerate(recommended):
                    if tutor_id in relevant_items:
                        dcg += 1.0 / np.log2(i + 2)  # i+2 because i starts at 0
                
                # Ideal ranking (all relevant items first)
                for i in range(min(len(relevant_items), top_k)):
                    idcg += 1.0 / np.log2(i + 2)
                
                ndcg = dcg / idcg if idcg > 0 else 0.0
                ndcgs.append(ndcg)
                
            except Exception as e:
                logger.warning(f"Error evaluating user {user_id}: {e}")
                continue
        
        # Calculate coverage (% of tutors recommended at least once)
        total_tutors = len(self.tutors_df)
        coverage = len(all_recommended) / total_tutors if total_tutors > 0 else 0.0
        
        # Average metrics
        avg_precision = np.mean(precisions) if precisions else 0.0
        avg_recall = np.mean(recalls) if recalls else 0.0
        avg_ndcg = np.mean(ndcgs) if ndcgs else 0.0
        
        return {
            f'precision@{top_k}': round(avg_precision, 4),
            f'recall@{top_k}': round(avg_recall, 4),
            f'ndcg@{top_k}': round(avg_ndcg, 4),
            'coverage': round(coverage, 4),
            'users_evaluated': len(precisions)
        }
