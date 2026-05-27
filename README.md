# Stock Prediction Full-stack Project

這個專案分成兩個服務：

- `backend/`: FastAPI 股票預測 API
- `frontend/`: Next.js 前端網站

## Local Development

Backend:

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Frontend:

```bash
cd frontend
npm install
npm run dev
```

`frontend/.env.local`:

```env
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
```

API checks:

```bash
curl http://localhost:8000/api/health
curl http://localhost:8000/api/trained-stocks
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"code":"2303"}'
```

## Deployment

### Backend

Deploy `backend/` as a Python web service.

Build command:

```bash
pip install -r requirements.txt
```

Start command:

```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```

Environment variables:

```env
ALLOWED_ORIGINS=https://your-frontend-domain.com
```

Use commas for multiple origins:

```env
ALLOWED_ORIGINS=https://your-frontend-domain.com,http://localhost:3000
```

### Frontend

Deploy `frontend/` as a Node/Next.js web service.

Build command:

```bash
npm ci && npm run build
```

Start command:

```bash
npm run start
```

Environment variables:

```env
NEXT_PUBLIC_API_BASE_URL=https://your-backend-domain.com
```

If the frontend host supports Next.js rewrites and you prefer same-origin API calls, leave `NEXT_PUBLIC_API_BASE_URL` empty and set:

```env
API_PROXY_BASE_URL=https://your-backend-domain.com
```

## Render Blueprint

`render.yaml` is included for a two-service Render deployment. After creating the services, set:

- Backend `ALLOWED_ORIGINS` to the final frontend URL
- Frontend `NEXT_PUBLIC_API_BASE_URL` to the final backend URL

## Model Training

Raw stock files live under `backend/data/`.

Train all models:

```bash
cd backend
python train_all.py
```

Force retrain:

```bash
python train_all.py --force
```
