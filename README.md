# Tutor Recommendation System

API đơn giản để gợi ý gia sư dựa trên quality score (rating, experience, students count, reviews).

## Build và chạy

```bash
docker-compose up --build
```

Nếu gặp lỗi permission:

```bash
sudo docker-compose up --build
```

API chạy tại: http://localhost:8000

## API Endpoints

### 1. Health Check

```bash
GET /health
```

### 2. Train Model (optional - tự động train khi start)

```bash
POST /train
Content-Type: multipart/form-data

curl -X POST "http://localhost:8000/train" -F "file=@data/tutors_adjust.json"
```

### 3. Get Recommendations

```bash
POST /get_recommended
Content-Type: application/json
```

Request body:
```json
{
  "user_ids": ["user1", "user2", "user3"],
  "k": 10
}
```

Response:
```json
{
  "data": [
    {
      "user_id": "user1",
      "recommended_tutor_ids": [123, 456, 789, 234, 567, 890, 345, 678, 901, 432]
    },
    {
      "user_id": "user2",
      "recommended_tutor_ids": [123, 456, 789, 234, 567, 890, 345, 678, 901, 432]
    }
  ]
}
```

## Ví dụ sử dụng

### Curl

```bash
curl -X POST "http://localhost:8000/get_recommended" \
  -H "Content-Type: application/json" \
  -d '{"user_ids": ["user1", "user2"], "k": 5}'
```

### Python

```python
import requests

response = requests.post(
    "http://localhost:8000/get_recommended",
    json={
        "user_ids": ["user1", "user2", "user3"],
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
    user_ids: ['user1', 'user2', 'user3'],
    k: 10
  })
})
.then(res => res.json())
.then(data => console.log(data));
```

## Thuật toán

Gợi ý dựa trên quality score:
- Rating: 30%
- Experience: 20%
- Students count: 20%
- Number of reviews: 20%
- Professional: 10%

**Lưu ý**: Tất cả users nhận cùng recommendations (top tutors theo quality).
Parameter `user_id` được giữ để dễ mở rộng trong tương lai.

## Cấu trúc

```
.
├── app/
│   ├── __init__.py
│   ├── main.py         # FastAPI endpoints
│   └── recommender.py  # Recommendation logic
├── data/
│   └── tutors_adjust.json
├── models/
│   └── recommender.pkl (auto-generated)
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

## Dừng service

```bash
docker-compose down
```
hoặc
```bash
sudo docker-compose down
```
