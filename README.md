# Tutor Recommendation System

API hệ thống gợi ý gia sư thông minh với personalization và machine learning.

## Quick Start

### Bước 1: Clone repository

```bash
git clone <repo-url>
cd API_recom_Duc
```

### Bước 2: Chạy với Docker

```bash
# Build và start (chạy foreground)
docker-compose up --build

# Hoặc chạy background
docker-compose up -d

# Check logs
docker-compose logs -f

# Stop
docker-compose down
```

Nếu gặp lỗi permission:
```bash
sudo docker-compose up -d
```

### Bước 3: Verify API

API chạy tại: **http://localhost:8000**

```bash
# Test health check
curl http://localhost:8000/health

# Kết quả mong đợi: status "healthy" với 475 tutors, 506 students, 2958 interactions
```

## API Endpoints

### 1. Health Check - Kiểm tra trạng thái

**Endpoint:** `GET /health`

**Mục đích:** Kiểm tra API đang chạy và data đã load chưa

**Example:**
```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "tutors_count": 475,
  "students_count": 506,
  "interactions_count": 2958,
  "personalization_enabled": true
}
```

**Ý nghĩa:**
- `model_loaded: true` → Model đã sẵn sàng
- `personalization_enabled: true` → Có đủ data để personalize
- Counts → Số lượng data đã load

---

### 2. Get Recommendations - Nhận gợi ý gia sư
```bash
# Top-10 recommendations
curl http://localhost:8000/get_metrics?top_k=10

# Top-5 recommendations (precision cao hơn)
curl http://localhost:8000/get_metrics?top_k=5
```

---

### 4. Train Model - Retrain với data mới (Optional)

**Endpoint:** `POST /train`

**Mục đích:** Upload data mới và retrain model

**Lưu ý:** Không cần thiết vì model tự động train khi start

**Example:**
```bash
curl -X POST "http://localhost:8000/train" \
  -F "file=@data/tutors_adjust.json"
```

## Ví dụ sử dụng

### Curl

```bash
# Personalized recommendation
curl -X POST "http://localhost:8000/get_recommended" \
  -H "Content-Type: application/json" \
  -d '{"user_ids": ["e8a9f3b5-4d7c-4e12-8f0a-1b6c7d2e3f4a"], "k": 5}'

# Benchmark metrics
curl "http://localhost:8000/get_metrics?top_k=10"
```

### Python

```python
import requests

response = requests.post(
    "http://localhost:8000/get_recommended",
    json={
        "user_ids": ["e8a9f3b5-4d7c-4e12-8f0a-1b6c7d2e3f4a", "a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d"],
        "k": 10
    }
)

print(response.json())
```

### JavaScript

```javascript
fetch('http://localhost:8000/get_recommended', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    user_ids: ['e8a9f3b5-4d7c-4e12-8f0a-1b6c7d2e3f4a'],
    k: 10
  })
})
.then(res => res.json())
.then(data => console.log(data));
```

## Performance

**Training:**
- Load 475 tutors + 506 students + 2,958 interactions: ~2 seconds
- Model size: 3.6 MB

**Inference:**
- Personalized recommendation: ~50ms/user
- General recommendation: ~10ms/user
- Batch 100 users: ~2 seconds

**Memory:**
- Runtime: ~300 MB RAM
- Docker container: ~500 MB

## Pipeline & Thuật toán

### Recommendation Pipeline

```
User Request → Load Profile → Calculate Scores → Rank & Filter → Return Top-K
     ↓              ↓               ↓                ↓              ↓
  user_id    Students Data    5 Algorithms    Sort by score   JSON Response
                  ↓               ↓                
            Interactions    Hybrid Scoring
```

**Chi tiết từng bước:**

1. **Load Profile** (nếu user_id có trong data)
   - Student demographics (age, school level)
   - Teaching style preferences
   - Subject preferences
   - Interaction history (views, clicks, ratings)

2. **Calculate Scores** - Hybrid recommendation với 5 components:

```python
final_score = (
    0.30 × content_based_score +      # Matching teaching style & subjects
    0.20 × collaborative_score +      # Similar students' choices
    0.25 × behavioral_score +         # Interaction history weights
    0.15 × quality_score +            # Rating, experience, reviews
    0.10 × popularity_score           # Students count, total lessons
)
```

3. **Rank & Filter**
   - Sort tutors theo final_score
   - Return top-K tutors

### Personalization Logic

**Case 1: User có trong data** (Personalized)
```
Teaching Style Matching (15%) → Find tutor style matches student preference
      +
Collaborative Filtering (20%) → Find similar students → Get their favorite tutors
      +
Behavioral Scoring (25%) → Weight events: view(0.1) < click(0.2) < join(0.6) < rating(1.0)
      +
Content-based Quality (30%) → Rating, experience, students count, reviews
      +
Popularity (10%) → Overall popularity metrics
      ↓
Personalized Top-K
```

**Case 2: User không có trong data** (General)
```
Quality-based Recommendation:
  - Rating normalization (30%)
  - Years of experience (20%)
  - Students count (20%)
  - Number of reviews (20%)
  - Professional status bonus (10%)
      ↓
General Top-K (backward compatible)
```

## Data Sources

Hệ thống sử dụng 3 nguồn data:

| File | Kích thước | Bắt buộc | Mục đích |
|------|-----------|---------|----------|
| **tutors_adjust.json** | 475 tutors | Required | Profile gia sư (rating, experience, subjects, teaching styles) |
| **students_output.json** | 506 students | Optional | Student preferences (Enables personalization) |
| **interaction_logs.jsonl** | 2,958 events | Optional | User behavior (Enables behavioral learning) |

**Data được load tự động khi start:**
- Nếu có đủ 3 files: Full personalization (precision@10 = 33%)
- Chỉ có tutors: General recommendations (vẫn hoạt động tốt)

**Event types trong interaction_logs:**
```javascript
{
  "userId": "...",
  "tutorId": 123,
  "eventType": "view",     // 0.1 weight
  "eventType": "click",    // 0.2 weight  
  "eventType": "wishlist", // 0.4 weight
  "eventType": "join",     // 0.6 weight
  "eventType": "conversion", // 0.9 weight
  "eventType": "rating",   // 1.0 weight (strongest signal)
  "timestamp": "..."
}
```

## Cấu trúc project

```
API_recom_Duc/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI app, endpoints, startup logic
│   └── recommender.py       # Core recommendation engine
│       ├── TutorRecommender class
│       ├── train()          # Load & train model
│       ├── recommend()      # Generate recommendations
│       └── evaluate_metrics()  # Calculate precision/recall/NDCG
│
├── data/
│   ├── tutors_adjust.json        # 475 tutors (1.5 MB)
│   ├── students_output.json      # 506 students (450 KB)
│   └── interaction_logs.jsonl    # 2,958 events (990 KB)
│
├── models/
│   └── recommender.pkl           # Trained model (3.6 MB, auto-generated)
│
├── docker-compose.yml       # Docker orchestration
├── Dockerfile              # Container definition
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

**Key files:**
- `main.py`: API endpoints, request validation, error handling
- `recommender.py`: Recommendation algorithms, scoring logic
- `recommender.pkl`: Cached model (students_df, interactions_df, similarity_matrix)

## Dừng service

```bash
docker-compose down

# Hoặc với sudo
sudo docker-compose down

# Xóa volumes và rebuild từ đầu
docker-compose down -v
docker-compose up --build
```

## Troubleshooting

**Port 8000 đang được sử dụng:**
```bash
# Kiểm tra process
lsof -i:8000

# Stop container cũ
docker-compose down

# Hoặc đổi port trong docker-compose.yml
ports:
  - "8001:8000"  # External:Internal
```

**Model không load:**
```bash
# Xóa model cũ và retrain
rm models/recommender.pkl
docker-compose restart
```

**Metrics trả về lỗi:**
```bash
# Cần có interaction_logs.jsonl để evaluate
# Check file exists
ls -lh data/interaction_logs.jsonl
```

---
