# image-recognition
this is the project of image recognition using Python, FastAPI, and ML/NLP integration.
# 🖼️ Image Recognition Backend

A production-ready backend for image recognition powered by **ML** (ViT, DETR) and **NLP** (Sentence Transformers, GPT-2 captioning), built with **FastAPI**.

---

## Features

| Feature | Model Used |
|---|---|
| Image Classification | `google/vit-base-patch16-224` (Vision Transformer) |
| Object Detection | `facebook/detr-resnet-50` (DETR) |
| Image Captioning | `nlpconnect/vit-gpt2-image-captioning` |
| Semantic NLP Search | `sentence-transformers/all-MiniLM-L6-v2` |

---

## Architecture

```
app/
├── main.py                  # FastAPI app + lifespan (model loading)
├── schemas.py               # Pydantic request/response models
├── core/
│   ├── config.py            # Settings (env vars, thresholds)
│   ├── model_manager.py     # Loads & runs all ML models
│   └── image_store.py       # In-memory store with semantic search
├── routers/
│   ├── health.py            # GET /health
│   ├── recognition.py       # POST /analyze, /classify, /detect, GET /history
│   └── nlp_query.py         # POST /query, GET /tags
└── utils/
    └── image_utils.py       # Image validation, tag extraction
tests/
└── test_api.py              # Full endpoint test suite (pytest)
```

---

## Quick Start (Backend + Frontend)

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env as needed
```

### 3. Run the backend

```bash
uvicorn app.main:app --reload --port 8000
```

Visit **http://localhost:8000/docs** for the interactive Swagger UI.

### 4. Run the futuristic frontend

From the `frontend` folder:

```bash
cd frontend
npm install
npm run dev
```

This starts the React/Vite UI on **http://localhost:5173** with a dev proxy to your backend on **http://localhost:8000**.  
All API calls from the frontend to `/api/v1/*` are automatically forwarded to the FastAPI server, so you don’t need to worry about CORS or ports during development.

If you prefer to keep the backend in the background on macOS/Linux:

```bash
uvicorn app.main:app --reload --port 8000 &
cd frontend
npm run dev
```

---

## API Reference

### Image Analysis

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/analyze` | Full analysis (classify + detect + caption) |
| `POST` | `/api/v1/classify` | Classification only (faster) |
| `POST` | `/api/v1/detect` | Object detection with bounding boxes |
| `GET` | `/api/v1/history` | List all analyzed images |
| `GET` | `/api/v1/image/{image_id}` | Get single image result |

### NLP / Search

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/query` | Natural language semantic search |
| `GET` | `/api/v1/tags` | List all unique tags |
| `GET` | `/api/v1/tags/{tag}` | Find images by tag |

### Health

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/v1/health` | Model status + device info |

---

## Example Requests

### Analyze an image

```bash
curl -X POST http://localhost:8000/api/v1/analyze \
  -F "file=@my_photo.jpg"
```

**Response:**
```json
{
  "image_id": "abc123",
  "filename": "my_photo.jpg",
  "caption": "a dog running in the park",
  "classifications": [
    {"label": "golden retriever", "confidence": 0.91}
  ],
  "detections": [
    {"label": "dog", "confidence": 0.87, "bbox": {"x_min": 50, "y_min": 30, "x_max": 200, "y_max": 180}}
  ],
  "tags": ["dog", "golden retriever"],
  "timestamp": "2025-01-01T12:00:00"
}
```

### Query with natural language

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "dogs playing outdoors", "top_k": 3}'
```

---

## Testing

```bash
pytest tests/ -v
```

---

## Production Considerations

- **Database**: Replace `ImageStore` in `app/core/image_store.py` with PostgreSQL + pgvector for scalable vector search
- **Object Storage**: Store uploaded images in S3/GCS instead of processing in memory
- **Model Serving**: Use TorchServe or Triton Inference Server for GPU-accelerated inference at scale
- **Caching**: Add Redis for caching repeated image analyses
- **Auth**: Add JWT/OAuth2 middleware for protected endpoints
- **GPU**: Set `CUDA_VISIBLE_DEVICES` to use GPU — models automatically use CUDA if available
