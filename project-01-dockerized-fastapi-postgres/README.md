# Project 1 — Dockerized FastAPI + PostgreSQL CRUD Service

**Org:** Knat LLM · **Maintained by:** Knatware Technology

A small, production-shaped CRUD service — FastAPI on top of PostgreSQL, fully containerized with
Docker Compose. This is the first project in the Knat LLM series and exists to establish the SQL,
container, and API fluency that every later project (RAG pipelines, model-serving platforms,
CI/CD for models) is built on top of.

## Why this project exists

Every project further down the Knat LLM series eventually needs a relational store — for user
history, conversation logs, model metadata, or rollout state. Rather than bolt that on later,
this project builds and exercises that muscle first: a clean, typed, tested CRUD API backed by a
real Postgres instance, running the way it would in a real deployment (containers, health checks,
environment-based config), not a local SQLite shortcut.

## Architecture

```
┌─────────────┐        ┌──────────────┐        ┌──────────────┐
│   Client     │ ────▶ │  FastAPI      │ ────▶ │  PostgreSQL   │
│ (curl/HTTPie/│  HTTP │  (uvicorn)    │  SQL  │  16-alpine    │
│  Swagger UI) │       │  container    │       │  container    │
└─────────────┘        └──────────────┘        └──────────────┘
                              │
                        docker-compose
                     (shared network + volume)
```

- **`app/main.py`** — route definitions (health check + full CRUD for an `items` resource)
- **`app/models.py`** — SQLAlchemy ORM model
- **`app/schemas.py`** — Pydantic request/response schemas (separate from the ORM model on purpose,
  so the API contract can evolve independently of the storage layer)
- **`app/database.py`** — engine/session management, driven entirely by environment variables
- **`docker-compose.yml`** — wires the API container to a Postgres container with a health-check
  gate, so the API never starts before the database is actually ready to accept connections

## Getting started

```bash
git clone <this-repo-url>
cd project-01-dockerized-fastapi-postgres
cp .env.example .env        # adjust credentials if you want
docker compose up --build
```

The API will be available at `http://localhost:8000`, with interactive docs at
`http://localhost:8000/docs`.

## API reference

| Method | Path            | Description                     |
|--------|-----------------|----------------------------------|
| GET    | `/health`       | Liveness check                   |
| POST   | `/items`        | Create an item                   |
| GET    | `/items`        | List items (paginated)           |
| GET    | `/items/{id}`   | Fetch a single item               |
| PUT    | `/items/{id}`   | Partially update an item          |
| DELETE | `/items/{id}`   | Delete an item                    |

Example:

```bash
curl -X POST http://localhost:8000/items \
  -H "Content-Type: application/json" \
  -d '{"name": "GPU credit pack", "description": "Compute credits", "price": 49.99}'
```

## What this project deliberately practices

- Writing SQL-backed models with SQLAlchemy 2.0's declarative style
- Separating storage models from API schemas (Pydantic v2)
- Environment-driven configuration for multi-environment deployment
- Docker Compose service dependencies and health-check gating
- A clean base to extend with auth, migrations (Alembic), and tests

## Roadmap / natural extensions

- Add Alembic migrations instead of `create_all` at startup
- Add JWT-based auth on write endpoints
- Add `pytest` + `httpx` integration tests against a throwaway test database
- Swap `psycopg2-binary` for `asyncpg` + `SQLAlchemy` async engine for higher-concurrency workloads

## Enquiries & implementation support

For enquiries, custom implementation, or extending this into a production deployment, contact
**kayode@knatware.com**.

---
© Knatware Technology — part of the **Knat LLM** project series.
