# CiteConnect - Quick Start Guide

## Step-by-Step Execution Instructions

### Prerequisites Check
- ✅ Python 3.11 installed
- ✅ Virtual environment created
- ✅ Dependencies installed
- ⚠️ Docker Desktop needs to be running

---

## Step 1: Start Docker Desktop

**IMPORTANT:** Make sure Docker Desktop is running before proceeding!

1. Open Docker Desktop application
2. Wait until it shows "Docker Desktop is running"
3. Verify with: `docker ps` (should not show errors)

---

## Step 2: Start Database Services

Open a terminal and run:

```bash
cd /Users/anusha/CiteConnect/CiteConnect-ModelPipeline
docker-compose up -d
```

This starts:
- PostgreSQL (port 5432)
- Redis (port 6379)
- Weaviate (port 8080)
- Neo4j (port 7474, 7687)

**Wait 10-15 seconds** for services to fully start.

**Verify services are running:**
```bash
docker ps
```

You should see 4 containers running:
- citeconnect-postgres
- citeconnect-redis
- citeconnect-weaviate
- citeconnect-neo4j

---

## Step 3: Verify Database Connections

Activate virtual environment and test connections:

```bash
cd /Users/anusha/CiteConnect/CiteConnect-ModelPipeline
source venv/bin/activate
cd citeconnect-backend
python test_db_connectivity.py
```

**Expected output:** All databases should show "✓ CONNECTED"

If any fail, wait a bit longer and try again.

---

## Step 4: Verify Setup (Optional but Recommended)

Test that all modules load correctly:

```bash
cd citeconnect-backend
python verify_setup.py
```

This checks:
- Module imports
- Configuration loading
- Security functions
- Model instantiation

---

## Step 5: Start FastAPI Server

In a new terminal window:

```bash
cd /Users/anusha/CiteConnect/CiteConnect-ModelPipeline
source venv/bin/activate
cd citeconnect-backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Expected output:**
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Started reloader process
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

**Keep this terminal open!** The server runs here.

---

## Step 6: Test the API

### Option A: Using Browser

1. **Health Check:**
   - Open: http://localhost:8000/api/v1/health
   - Should show status of all services

2. **API Documentation:**
   - Open: http://localhost:8000/docs
   - Interactive Swagger UI to test endpoints

3. **Root Endpoint:**
   - Open: http://localhost:8000/
   - Should show welcome message

### Option B: Using curl (Terminal)

**Health Check:**
```bash
curl http://localhost:8000/api/v1/health
```

**Root Endpoint:**
```bash
curl http://localhost:8000/
```

---

## Step 7: Test User Registration (Example)

Using the Swagger UI at http://localhost:8000/docs:

1. Click on `POST /api/v1/auth/register`
2. Click "Try it out"
3. Enter test data:
```json
{
  "email": "test@example.com",
  "password": "TestPassword123!",
  "name": "Test User",
  "domain": "healthcare",
  "interests": ["machine learning", "healthcare"]
}
```
4. Click "Execute"
5. Should return user data with `user_id`

---

## Step 8: Test Recommendations (If Pickle File Exists)

**Prerequisites:** You need `embeddings_db.pkl` in `citeconnect-backend/working_data/`

Check if pickle exists:
```bash
ls citeconnect-backend/working_data/embeddings_db.pkl
```

If it exists, you can test recommendations via API:
- Use the registered user's token
- Call `GET /api/v1/recommendations` endpoint

---

## Troubleshooting

### Docker Issues
- **Error: "Cannot connect to Docker daemon"**
  - Start Docker Desktop
  - Wait for it to fully start
  - Try `docker ps` to verify

### Database Connection Issues
- **PostgreSQL connection failed**
  - Wait 10-15 seconds after `docker-compose up -d`
  - Check: `docker logs citeconnect-postgres`

- **Weaviate connection failed**
  - Check: `docker logs citeconnect-weaviate`
  - Verify port 8080 is not in use

### FastAPI Server Issues
- **Port 8000 already in use**
  - Change port: `uvicorn app.main:app --reload --port 8001`
  - Or kill process using port 8000

- **Import errors**
  - Make sure virtual environment is activated
  - Reinstall: `pip install -r requirements-dev.txt`

### Missing Pickle File
- **Error: "Embeddings pickle not found"**
  - This is expected if DataPipeline hasn't run yet
  - Recommendations won't work until pickle is generated
  - Other endpoints (auth, health) will still work

---

## Quick Commands Reference

```bash
# Start services
docker-compose up -d

# Stop services
docker-compose down

# View logs
docker-compose logs -f

# Check service status
docker ps

# Activate venv
source venv/bin/activate

# Start server
cd citeconnect-backend
uvicorn app.main:app --reload

# Test database connections
python test_db_connectivity.py

# Verify setup
python verify_setup.py
```

---

## Next Steps

Once everything is running:

1. ✅ Health check works
2. ✅ API docs accessible at /docs
3. ✅ Can register users
4. ✅ Can test endpoints via Swagger UI

**To test recommendations:**
- Need to run DataPipeline first to generate `embeddings_db.pkl`
- Or use existing pickle file if available

---

## Stopping Everything

1. **Stop FastAPI server:** Press `Ctrl+C` in the server terminal
2. **Stop Docker services:**
   ```bash
   docker-compose down
   ```

---

**Need Help?** Check the logs:
- FastAPI: Look at the terminal running uvicorn
- Databases: `docker-compose logs [service-name]`


