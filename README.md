# ESG Portfolio Optimization — Backend

FastAPI backend for the ESG Portfolio Optimization Chatbot. Provides AI-powered investment recommendations using Deep Reinforcement Learning, multi-agent chat (Google ADK + Gemini), and real-time data processing.

## Tech Stack

- **Framework:** FastAPI + Uvicorn
- **AI Agents:** Google ADK (Agent Development Kit) + Gemini 2.5
- **DRL Engine:** PyTorch (MAPPO / RA-DRL)
- **Database:** PostgreSQL 16 + pgvector
- **Task Queue:** Celery + Redis
- **Real-time:** Socket.IO (python-socketio)
- **Auth:** JWT (python-jose) + bcrypt (passlib)

## Prerequisites

- Python 3.11+
- PostgreSQL 16 with pgvector extension
- Docker (for Redis only)

## Setup

### 1. Clone and install dependencies

```bash
cd esg-portfolio-backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env` with your actual credentials:

```bash
DB_HOST=localhost
DB_PORT=5432
DB_NAME=esg_portfolio
DB_USER=postgres
DB_PASSWORD=your_password

REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

GEMINI_API_KEY=your-gemini-api-key

CORS_ORIGINS=http://localhost:3000

JWT_SECRET_KEY=your-super-secret-key-change-in-production
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=30
JWT_REFRESH_TOKEN_EXPIRE_DAYS=7

MODEL_CHECKPOINT_DIR=./model_checkpoints
```

### 3. Set up PostgreSQL

```bash
# Create database
psql -U postgres -c "CREATE DATABASE esg_portfolio;"

# Enable pgvector extension
psql -U postgres -d esg_portfolio -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

### 4. Start Redis (Docker)

```bash
docker compose up -d
```

### 5. Run database migrations

```bash
alembic init alembic
alembic upgrade head
alembic revision --autogenerate -m "initial"
```

### 6. Start the server

```bash
source venv/bin/activate
uvicorn main:socket_app --host 0.0.0.0 --port 8000 --reload
```

Server runs at: <http://localhost:8000>

## API Documentation

Once the server is running:

- **Swagger UI:** <http://localhost:8000/docs>
- **ReDoc:** <http://localhost:8000/redoc>
- **Health Check:** <http://localhost:8000/health>

## API Endpoints

### Auth

| Method | Endpoint                    | Auth | Description                     |
| ------ | --------------------------- | ---- | ------------------------------- |
| POST   | `/api/auth/register`        | No   | Create account                  |
| POST   | `/api/auth/login`           | No   | Login → access + refresh tokens |
| POST   | `/api/auth/refresh`         | No   | Refresh access token            |
| POST   | `/api/auth/logout`          | Yes  | Revoke refresh tokens           |
| GET    | `/api/auth/me`              | Yes  | Get current user profile        |
| PUT    | `/api/auth/me`              | Yes  | Update profile                  |
| PUT    | `/api/auth/change-password` | Yes  | Change password                 |

### Chat

| Method | Endpoint                  | Auth | Description                |
| ------ | ------------------------- | ---- | -------------------------- |
| POST   | `/api/chat`               | Yes  | Send message → AI response |
| GET    | `/api/chat/sessions`      | Yes  | List chat sessions         |
| GET    | `/api/chat/sessions/{id}` | Yes  | Get conversation history   |
| DELETE | `/api/chat/sessions/{id}` | Yes  | Delete a session           |

### Training (Admin only)

| Method | Endpoint                        | Auth  | Description                       |
| ------ | ------------------------------- | ----- | --------------------------------- |
| POST   | `/api/training/upload`          | Admin | Upload data file (xlsx, csv, pdf) |
| GET    | `/api/training/status/{job_id}` | Admin | Check ingestion progress          |
| GET    | `/api/training/history`         | Admin | List past ingestion jobs          |
| POST   | `/api/training/recompute`       | Admin | Trigger metric recomputation      |

### Data

| Method | Endpoint                          | Auth | Description                            |
| ------ | --------------------------------- | ---- | -------------------------------------- |
| GET    | `/api/sectors`                    | Yes  | Sector rankings                        |
| GET    | `/api/sectors/{sector}`           | Yes  | Sector detail + companies              |
| GET    | `/api/companies`                  | Yes  | Company list with filters              |
| GET    | `/api/companies/{symbol}`         | Yes  | Company detail                         |
| GET    | `/api/companies/{symbol}/similar` | Yes  | Find similar companies (vector search) |
| POST   | `/api/portfolio/optimize`         | Yes  | DRL portfolio optimization             |
| POST   | `/api/portfolio/analyze`          | Yes  | Analyze user portfolio                 |

### Admin

| Method | Endpoint                          | Auth  | Description                |
| ------ | --------------------------------- | ----- | -------------------------- |
| POST   | `/api/admin/models/train`         | Admin | Trigger DRL model training |
| GET    | `/api/admin/models`               | Admin | List trained models        |
| PUT    | `/api/admin/models/{id}/activate` | Admin | Activate a model           |

## Socket.IO Events

### Chat (Investor)

| Direction       | Event                 | Description                              |
| --------------- | --------------------- | ---------------------------------------- |
| Client → Server | `chat:send_message`   | Send a chat message                      |
| Server → Client | `chat:thinking_step`  | Agent/tool step during processing        |
| Server → Client | `chat:response_start` | Thinking done, response streaming begins |
| Server → Client | `chat:response_token` | Streamed response chunk                  |
| Server → Client | `chat:response_end`   | Response complete                        |

### Training (Admin)

| Direction       | Event                     | Description                    |
| --------------- | ------------------------- | ------------------------------ |
| Server → Client | `training:job_started`    | Ingestion pipeline started     |
| Server → Client | `training:chunk_progress` | Per-chunk progress update      |
| Server → Client | `training:job_completed`  | Pipeline finished with summary |

## Project Structure

```bash
esg-portfolio-backend/
├── main.py                     # FastAPI + Socket.IO entry point
├── config.py                   # Pydantic Settings (.env loading)
├── api/                        # Route handlers
│   ├── router.py               # Main API router
│   ├── auth.py                 # Auth endpoints
│   ├── chat.py                 # Chat endpoints (TODO)
│   ├── training.py             # Training endpoints (TODO)
│   ├── sectors.py              # Sector endpoints (TODO)
│   ├── companies.py            # Company endpoints (TODO)
│   ├── portfolio.py            # Portfolio endpoints (TODO)
│   └── admin.py                # Admin endpoints (TODO)
├── core/
│   ├── auth/
│   │   ├── security.py         # JWT + bcrypt
│   │   └── dependencies.py     # get_current_user, require_admin
│   ├── adk/                    # Google ADK agents (TODO)
│   ├── tools/                  # ADK FunctionTools (TODO)
│   ├── parsers/                # File parsers (TODO)
│   ├── metrics.py              # Financial metrics (TODO)
│   ├── drl_engine.py           # DRL inference (TODO)
│   ├── screening.py            # ESG screening (TODO)
│   ├── embeddings.py           # pgvector embeddings (TODO)
│   └── constraints.py          # Portfolio constraints (TODO)
├── db/
│   ├── database.py             # Async SQLAlchemy engine
│   ├── models.py               # 12 ORM models
│   ├── crud.py                 # CRUD operations
│   └── migrations/             # Alembic migrations
├── drl/                        # DRL models + training (TODO)
├── schemas/                    # Pydantic request/response models
│   └── auth.py                 # Auth schemas
├── tasks/                      # Celery tasks (TODO)
├── uploads/                    # Temp file uploads (gitignored)
├── model_checkpoints/          # Trained DRL models (gitignored)
├── tests/                      # Tests (TODO)
├── docker-compose.yml          # Redis service
├── alembic.ini                 # Alembic config
├── requirements.txt
├── pyproject.toml
├── .env                        # Environment variables (gitignored)
└── .env.example                # Template for .env
```

## Running Celery Worker

```bash
source venv/bin/activate
celery -A tasks.celery_app worker --loglevel=info
```

## Running Tests

```bash
source venv/bin/activate
pytest
```

## Creating an Admin User

After running migrations, create an admin user:

```bash
source venv/bin/activate
python -c "
import asyncio
from db.database import async_session
from db.crud import create_user
from core.auth.security import hash_password

async def seed():
    async with async_session() as db:
        await create_user(db, {
            'email': 'admin@example.com',
            'password_hash': hash_password('admin123'),
            'full_name': 'Admin',
            'role': 'admin',
            'is_verified': True,
        })
        print('Admin user created')

asyncio.run(seed())
"
```
