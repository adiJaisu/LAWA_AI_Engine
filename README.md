
---

# 🚀 Running the Project

This project uses:

* FastAPI
* React
* PostgreSQL
* Docker

---

# 🐳 Option 1: Run with Docker (Recommended)

### 1. Start all services

```bash
docker-compose up --build
```

### 2. What happens automatically

* PostgreSQL container starts
* Database is created
* Tables are created via `create_db_tables.sql`
* Default data is inserted via `insert_default_data.sql`
* Backend and frontend start after DB is ready

### 3. Access the app

* Frontend: [http://localhost:5173](http://localhost:5173)
* Backend API: [http://localhost:8010](http://localhost:8010)
* API Docs: [http://localhost:8010/docs](http://localhost:8010/docs)

### 4. Stop services

```bash
docker-compose down
```

---

# 💻 Option 2: Run Without Docker

## 🗄️ Step 1: Install PostgreSQL

### 👉 Ubuntu / Debian

```bash
sudo apt update
sudo apt install postgresql postgresql-contrib
```

### 👉 macOS (using Homebrew)

```bash
brew install postgresql
brew services start postgresql
```

### 👉 Windows

Download and install from official site: [https://www.postgresql.org/download/](https://www.postgresql.org/download/)

---

## 🗄️ Step 2: Setup Database

### 1. Open PostgreSQL shell

```bash
psql -U postgres
```

### 2. Create database and user

```sql
CREATE USER ai_vms_user WITH PASSWORD 'ai_vms_password';
CREATE DATABASE ai_vms_db;
ALTER DATABASE ai_vms_db OWNER TO ai_vms_user;
```

### 3. Run your SQL scripts

Navigate to project root and execute:

```bash
psql -U ai_vms_user -d ai_vms_db -f postgres-init/create_db_tables.sql
psql -U ai_vms_user -d ai_vms_db -f postgres-init/insert_default_data.sql
```

👉 This will:

* Create all tables
* Insert default seed data

---

## 🧠 Backend (FastAPI)

### 1. Navigate to backend folder

```bash
cd backend
```

### 2. Create virtual environment

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the backend

```bash
uvicorn ai_vms.main:app --host 0.0.0.0 --port 8010 --reload
```

Backend runs at: [http://localhost:8010](http://localhost:8010)

---

## 🎨 Frontend (React)

### 1. Navigate to frontend folder

```bash
cd frontend/ai_vms
```

### 2. Install dependencies

```bash
npm install
```

### 3. Run development server

```bash
npm run dev
```

Frontend runs at: [http://localhost:5173](http://localhost:5173)

---

# ⚙️ Environment Variables

Create `.env` at root:

```env
# Database Configuration
PG_IMAGE_VERSION=15
PG_ROOT_PASSWORD=admin
PG_USER=ai_vms_user
PG_PASSWORD=ai_vms_password
PG_DATABASE=ai_vms_db
 
# Ports
BACKEND_PORT=8010
FRONTEND_PORT=5173
 
# Security
SECRET_KEY=ai-vms-secret-key-12345
 
# Frontend (Vite)
VITE_BACKEND_URL=http://localhost:8010/api/v1
```

---

# 🧪 API Documentation

* Swagger UI: [http://localhost:8010/docs](http://localhost:8010/docs)
* ReDoc: [http://localhost:8010/redoc](http://localhost:8010/redoc)

---

# 📌 Notes

* Your **Docker setup already handles DB initialization automatically**
* For **manual setup**, you must run SQL scripts yourself
* Ensure port **5432** is free before starting PostgreSQL
* If DB connection fails, check credentials in `.env`

---
