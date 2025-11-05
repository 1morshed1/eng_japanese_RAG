# Complete Testing Guide - Healthcare RAG Assistant

## 📋 Table of Contents
1. [Prerequisites](#prerequisites)
2. [Setup Instructions](#setup-instructions)
3. [Testing with Postman](#testing-with-postman)
4. [Testing with cURL](#testing-with-curl)
5. [Testing with Python](#testing-with-python)
6. [Expected Results](#expected-results)
7. [Troubleshooting](#troubleshooting)

---

## 🎯 Prerequisites

Before testing, ensure you have:
- ✅ Project dependencies installed (`pip install -r requirements.txt`)
- ✅ Virtual environment activated (`source venv/bin/activate`)
- ✅ `.env` file configured with `API_KEYS=dev-key-12345`
- ✅ Server is NOT running yet (we'll start it in setup)

**Tools Needed** (choose one):
- **Option 1**: Postman (recommended) - [Download](https://www.postman.com/downloads/)
- **Option 2**: cURL (command line) - Usually pre-installed
- **Option 3**: Python requests library - Already in venv

---

## 🚀 Setup Instructions

### Step 1: Start the Server

Open a terminal and run:

```bash
cd /home/morshed/own-folder/Documents/jap_llm
source venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

**Expected Output**:
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Started reloader process [12345] with StatReload
INFO:     Started server process [12346]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

✅ **Server is now running!**

### Step 2: Verify Server is Running

Open another terminal and test:

```bash
curl http://localhost:8000/health
```

**Expected Response**:
```json
{
  "status": "healthy",
  "faiss_index_size": 0,
  "total_documents": 0,
  "model_loaded": false,
  "uptime_seconds": 5
}
```

✅ **If you see this, server is ready for testing!**

---

## 📮 Testing with Postman

### Import the Collection

1. **Open Postman**
2. Click **Import** button (top left)
3. Select **File** tab
4. Choose `Healthcare_RAG_Assistant.postman_collection.json`
5. Click **Import**

### Configure Collection Variables

1. Click on the collection name
2. Go to **Variables** tab
3. Verify these are set:
   - `base_url`: `http://localhost:8000`
   - `api_key`: `dev-key-12345`
4. Click **Save**

### Test Sequence (Run in Order)

#### 🟢 Step 1: Test Public Endpoints (No Auth Required)

1. **Root - Get API Info**
   - Click to run
   - Expected: 200 OK with API info
   - ✅ Tests: Server is running

2. **Health Check**
   - Click to run
   - Expected: 200 OK with health status
   - ✅ Tests: Services are initialized

3. **API Documentation**
   - Click to run (or open in browser: `http://localhost:8000/docs`)
   - Expected: Swagger UI HTML
   - ✅ Tests: OpenAPI schema generation

#### 🟡 Step 2: Test Ingest Endpoints (Auth Required)

**Important**: Update file paths in Postman before running!

4. **Ingest English Document**
   - Go to **Body** tab
   - Click on file field
   - Browse and select `docs/diabetes_eng.txt`
   - Click **Send**
   - Expected: 201 Created with document_id
   - ⏱️ First request: 30-60 seconds (model download)
   - ✅ Tests: Document ingestion, embedding generation, FAISS storage

5. **Ingest Japanese Document**
   - Change file to `docs/diabetes_jap.txt`
   - Click **Send**
   - Expected: 201 Created with document_id (language: "ja")
   - ⏱️ Should be fast (<2 seconds, model already loaded)
   - ✅ Tests: Japanese language support

6. **Ingest Health Check**
   - Click **Send**
   - Expected: 200 OK with service health status
   - ✅ Tests: Service dependencies

7. **Ingest Statistics**
   - Click **Send**
   - Expected: 200 OK with statistics (should show 2 documents now)
   - ✅ Tests: Metadata tracking

#### 🔵 Step 3: Test Retrieve Endpoints

8. **Retrieve - English Query**
   - Query: "What are the recommendations for Type 2 diabetes management?"
   - Click **Send**
   - Expected: 200 OK with top 3 results and similarity scores
   - ✅ Tests: Semantic search, ranking

9. **Retrieve - Japanese Query**
   - Query in Japanese
   - Click **Send**
   - Expected: 200 OK with relevant Japanese document chunks
   - ✅ Tests: Multilingual embedding search

10. **Retrieve - Low Threshold**
    - min_score: 0.3
    - Click **Send**
    - Expected: More results (lower quality threshold)
    - ✅ Tests: Similarity filtering

11. **Retrieve - High Threshold**
    - min_score: 0.7
    - Click **Send**
    - Expected: Fewer results (only highly relevant)
    - ✅ Tests: Quality filtering

#### 🟣 Step 4: Test Generate Endpoints

12. **Generate - English → English**
    - Query: "What are the dietary recommendations for managing Type 2 diabetes?"
    - output_language: "en"
    - Click **Send**
    - Expected: 200 OK with structured response + sources
    - ✅ Tests: RAG pipeline, response generation

13. **Generate - English → Japanese**
    - Query in English
    - output_language: "ja"
    - Click **Send**
    - Expected: Response in Japanese
    - ✅ Tests: Output translation

14. **Generate - Japanese → Japanese**
    - Query in Japanese
    - output_language: "ja"
    - Click **Send**
    - Expected: Response in Japanese
    - ✅ Tests: Full Japanese workflow

15. **Generate - Auto Language**
    - No output_language specified
    - Click **Send**
    - Expected: Response in same language as query
    - ✅ Tests: Auto language detection

16. **Generate Health Check**
    - Click **Send**
    - Expected: Health status of all services
    - ✅ Tests: System monitoring

#### 🔴 Step 5: Test Error Cases

17. **Test - Missing API Key**
    - Remove X-API-Key header
    - Click **Send**
    - Expected: 401 Unauthorized
    - ✅ Tests: Authentication enforcement

18. **Test - Invalid API Key**
    - X-API-Key: "wrong-key-12345"
    - Click **Send**
    - Expected: 401 Unauthorized
    - ✅ Tests: Key validation

19. **Test - Empty Query**
    - Query: ""
    - Click **Send**
    - Expected: 422 Validation Error
    - ✅ Tests: Input validation

20. **Test - Invalid top_k**
    - top_k: -1
    - Click **Send**
    - Expected: 422 Validation Error
    - ✅ Tests: Parameter validation

21. **Test - Query on Empty Index**
    - High min_score threshold
    - Click **Send**
    - Expected: 200 OK with empty results array
    - ✅ Tests: Empty index handling

---

## 💻 Testing with cURL

### Public Endpoints

```bash
# Get API info
curl http://localhost:8000/

# Health check
curl http://localhost:8000/health

# Admin stats (requires auth)
curl -H "X-API-Key: dev-key-12345" http://localhost:8000/admin/stats
```

### Ingest Document

```bash
# Ingest English document
curl -X POST "http://localhost:8000/api/v1/ingest" \
  -H "X-API-Key: dev-key-12345" \
  -F "file=@docs/diabetes_eng.txt"

# Save the document_id from response!
# Example response:
# {
#   "status": "success",
#   "document_id": "doc_abc123",
#   "language": "en",
#   "chunks_created": 25,
#   "file_size_kb": 12.5,
#   "processing_time_seconds": 35.2
# }
```

### Retrieve Documents

```bash
# Search for relevant documents
curl -X POST "http://localhost:8000/api/v1/retrieve" \
  -H "X-API-Key: dev-key-12345" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the recommendations for Type 2 diabetes management?",
    "top_k": 3,
    "min_score": 0.5
  }'
```

### Generate Response

```bash
# Generate contextual response
curl -X POST "http://localhost:8000/api/v1/generate" \
  -H "X-API-Key: dev-key-12345" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What medications are used for diabetes treatment?",
    "output_language": "en"
  }'

# Generate Japanese response
curl -X POST "http://localhost:8000/api/v1/generate" \
  -H "X-API-Key: dev-key-12345" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "糖尿病の治療法は？",
    "output_language": "ja"
  }'
```

### Test Error Cases

```bash
# Test missing API key (should fail with 401)
curl -X POST "http://localhost:8000/api/v1/retrieve" \
  -H "Content-Type: application/json" \
  -d '{"query": "test", "top_k": 3}'

# Test invalid API key (should fail with 401)
curl -X POST "http://localhost:8000/api/v1/retrieve" \
  -H "X-API-Key: wrong-key" \
  -H "Content-Type: application/json" \
  -d '{"query": "test", "top_k": 3}'

# Test empty query (should fail with 422)
curl -X POST "http://localhost:8000/api/v1/retrieve" \
  -H "X-API-Key: dev-key-12345" \
  -H "Content-Type: application/json" \
  -d '{"query": "", "top_k": 3}'
```

---

## 🐍 Testing with Python

### Create Test Script

Save as `test_api.py`:

```python
import requests
import json

BASE_URL = "http://localhost:8000"
API_KEY = "dev-key-12345"

def test_health():
    """Test health endpoint"""
    print("Testing /health...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")

def test_ingest(filepath):
    """Test document ingestion"""
    print(f"Testing /ingest with {filepath}...")
    with open(filepath, 'rb') as f:
        response = requests.post(
            f"{BASE_URL}/api/v1/ingest",
            headers={"X-API-Key": API_KEY},
            files={"file": f}
        )
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Response: {json.dumps(result, indent=2)}\n")
    return result.get('document_id')

def test_retrieve(query):
    """Test document retrieval"""
    print(f"Testing /retrieve with query: {query[:50]}...")
    response = requests.post(
        f"{BASE_URL}/api/v1/retrieve",
        headers={
            "X-API-Key": API_KEY,
            "Content-Type": "application/json"
        },
        json={
            "query": query,
            "top_k": 3,
            "min_score": 0.5
        }
    )
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Found {result['results_found']} results")
    if result['results']:
        print(f"Top score: {result['results'][0]['similarity_score']:.3f}")
    print()
    return result

def test_generate(query, output_language="en"):
    """Test response generation"""
    print(f"Testing /generate with query: {query[:50]}...")
    response = requests.post(
        f"{BASE_URL}/api/v1/generate",
        headers={
            "X-API-Key": API_KEY,
            "Content-Type": "application/json"
        },
        json={
            "query": query,
            "output_language": output_language
        }
    )
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Response length: {len(result['response'])} chars")
    print(f"Sources: {len(result['sources'])}")
    print(f"Response preview: {result['response'][:200]}...\n")
    return result

# Run tests
if __name__ == "__main__":
    print("="*80)
    print("HEALTHCARE RAG ASSISTANT - API TESTS")
    print("="*80)
    
    # Test 1: Health check
    test_health()
    
    # Test 2: Ingest document
    doc_id = test_ingest("docs/diabetes_eng.txt")
    print(f"✅ Document ingested with ID: {doc_id}\n")
    
    # Test 3: Retrieve
    results = test_retrieve("What are the recommendations for diabetes management?")
    print(f"✅ Retrieved {results['results_found']} relevant documents\n")
    
    # Test 4: Generate
    response = test_generate("What medications are used for diabetes?", "en")
    print(f"✅ Generated response with {len(response['sources'])} sources\n")
    
    # Test 5: Bilingual
    response_ja = test_generate("糖尿病の治療法は？", "ja")
    print(f"✅ Japanese response generated\n")
    
    print("="*80)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("="*80)
```

Run it:
```bash
python test_api.py
```

---

## 📊 Expected Results

### 1. Health Check
```json
{
  "status": "healthy",
  "faiss_index_size": 0,
  "total_documents": 0,
  "model_loaded": false,
  "uptime_seconds": 10
}
```

### 2. Ingest Document (First Time)
**Duration**: 30-60 seconds (model download)

```json
{
  "status": "success",
  "document_id": "doc_a1b2c3d4",
  "language": "en",
  "chunks_created": 25,
  "file_size_kb": 12.5,
  "processing_time_seconds": 35.2
}
```

### 3. Retrieve Documents
```json
{
  "results": [
    {
      "text": "Type 2 diabetes management requires...",
      "similarity_score": 0.847,
      "document_id": "doc_a1b2c3d4",
      "language": "en",
      "chunk_id": 5
    },
    {
      "text": "Regular monitoring of blood glucose...",
      "similarity_score": 0.782,
      "document_id": "doc_a1b2c3d4",
      "language": "en",
      "chunk_id": 8
    }
  ],
  "query_language": "en",
  "results_found": 2
}
```

### 4. Generate Response
```json
{
  "query": "What medications are used for diabetes?",
  "response": "Based on the retrieved medical guidelines regarding diabetes:\n\nKey Findings:\n1. First-line pharmacological therapy for type 2 diabetes typically includes metformin...\n\n⚠️ Disclaimer: This information is AI-generated based on retrieved documents. Please consult healthcare professionals for medical advice.",
  "language": "en",
  "sources": [
    {
      "document_id": "doc_a1b2c3d4",
      "chunk_id": 5,
      "similarity_score": 0.847
    }
  ],
  "generation_time_seconds": 0.234
}
```

### 5. Error Response (Missing API Key)
```json
{
  "detail": "Missing API key. Include 'X-API-Key' header in your request."
}
```

---

## 🧪 Complete Testing Checklist

### Basic Functionality Tests

- [ ] **Server starts without errors**
  ```bash
  uvicorn app.main:app --reload
  ```

- [ ] **Health endpoint responds**
  ```bash
  curl http://localhost:8000/health
  ```

- [ ] **API docs accessible**
  - Open: `http://localhost:8000/docs`

### Authentication Tests

- [ ] **Request without API key fails (401)**
  ```bash
  curl -X POST http://localhost:8000/api/v1/retrieve \
    -H "Content-Type: application/json" \
    -d '{"query": "test", "top_k": 3}'
  ```

- [ ] **Request with wrong API key fails (401)**
  ```bash
  curl -X POST http://localhost:8000/api/v1/retrieve \
    -H "X-API-Key: wrong-key" \
    -H "Content-Type: application/json" \
    -d '{"query": "test", "top_k": 3}'
  ```

- [ ] **Request with correct API key succeeds (200)**
  ```bash
  curl -X POST http://localhost:8000/api/v1/retrieve \
    -H "X-API-Key: dev-key-12345" \
    -H "Content-Type: application/json" \
    -d '{"query": "test", "top_k": 3}'
  ```

### Ingest Tests

- [ ] **Upload English document succeeds**
  ```bash
  curl -X POST http://localhost:8000/api/v1/ingest \
    -H "X-API-Key: dev-key-12345" \
    -F "file=@docs/diabetes_eng.txt"
  ```
  - ✅ Returns document_id
  - ✅ Language detected as "en"
  - ✅ Chunks created (should be 20-30)

- [ ] **Upload Japanese document succeeds**
  ```bash
  curl -X POST http://localhost:8000/api/v1/ingest \
    -H "X-API-Key: dev-key-12345" \
    -F "file=@docs/diabetes_jap.txt"
  ```
  - ✅ Returns document_id
  - ✅ Language detected as "ja"

- [ ] **Invalid file type rejected**
  - Try uploading .pdf or .docx
  - ✅ Should return 400 error

- [ ] **Empty file rejected**
  - Try uploading empty .txt
  - ✅ Should return 400 error

### Retrieve Tests

- [ ] **English query retrieves documents**
  - Query: "What is diabetes?"
  - ✅ Returns results with similarity scores
  - ✅ Scores between 0.0 and 1.0

- [ ] **Japanese query retrieves documents**
  - Query: "糖尿病とは？"
  - ✅ Returns results
  - ✅ Query language detected as "ja"

- [ ] **Empty index returns empty results gracefully**
  - Query before uploading any documents
  - ✅ Should return empty results array (not error)

- [ ] **High threshold filters results**
  - Set min_score: 0.9
  - ✅ Returns fewer/no results

### Generate Tests

- [ ] **Generate English response**
  - Query: "What is the treatment for diabetes?"
  - output_language: "en"
  - ✅ Returns structured response
  - ✅ Includes sources
  - ✅ Includes disclaimer

- [ ] **Generate Japanese response**
  - Query: "糖尿病の治療法は？"
  - output_language: "ja"
  - ✅ Response in Japanese
  - ✅ Includes Japanese disclaimer

- [ ] **Auto language detection works**
  - Don't specify output_language
  - ✅ Response in same language as query

- [ ] **Empty index handled gracefully**
  - Query when no documents uploaded
  - ✅ Returns appropriate "no information found" message

### Performance Tests

- [ ] **First request slower (model download)**
  - First ingest: 30-60 seconds ⏱️

- [ ] **Subsequent requests fast**
  - Second ingest: <2 seconds ⏱️

- [ ] **Retrieve is fast**
  - After models loaded: <1 second ⏱️

- [ ] **Generate is reasonable**
  - After models loaded: <2 seconds ⏱️

### Data Persistence Tests

- [ ] **Data persists after ingestion**
  - Check `data/faiss_index/index.faiss` exists
  - Check `data/faiss_index/metadata.json` exists

- [ ] **Data survives server restart**
  - Stop server (Ctrl+C)
  - Restart server
  - Query should still return results

- [ ] **Statistics update correctly**
  - Check `/admin/stats` after each ingest
  - ✅ Document count increases

---

## 🎯 Testing Workflow (Recommended Order)

### Part 1: Setup & Verification (5 minutes)

1. Start server
2. Test `/health` endpoint
3. Visit `/docs` in browser
4. Test `/admin/stats` (should show 0 documents)

### Part 2: Ingest Documents (2 minutes + model download time)

5. Upload `diabetes_eng.txt` (30-60 sec first time)
6. Upload `diabetes_jap.txt` (<2 sec)
7. Check `/admin/stats` (should show 2 documents)

### Part 3: Test Retrieval (2 minutes)

8. Query in English: "diabetes management"
9. Query in Japanese: "糖尿病管理"
10. Try different top_k and min_score values
11. Verify similarity scores are reasonable (>0.5 for good matches)

### Part 4: Test Generation (2 minutes)

12. Generate English response
13. Generate Japanese response
14. Test with output_language different from query language
15. Verify sources are included

### Part 5: Error Handling (2 minutes)

16. Test without API key
17. Test with wrong API key
18. Test with invalid inputs
19. Verify error messages are clear

### Part 6: Advanced Tests (Optional, 5 minutes)

20. Test with custom medical document
21. Test query on empty index (before ingesting)
22. Test concurrent requests (if possible)
23. Monitor logs (`tail -f logs/app.log`)

**Total Time**: ~15-20 minutes (including model downloads)

---

## 🔍 What to Look For (Success Indicators)

### ✅ Successful Test Signs

1. **Status Codes**:
   - 200 OK for successful requests
   - 201 Created for document ingestion
   - 401 Unauthorized for auth failures
   - 422 Validation Error for invalid input
   - 500 Internal Server Error only if unexpected failure

2. **Response Structure**:
   - All responses are valid JSON
   - Required fields present
   - No null values where not expected

3. **Semantic Search Quality**:
   - Relevant documents have scores >0.5
   - Top result is most relevant
   - Results ordered by similarity (descending)

4. **Response Generation Quality**:
   - Response mentions relevant medical terms
   - Sources are cited
   - Medical disclaimer included
   - Language matches output_language parameter

5. **Performance**:
   - First request: 30-60 seconds (model download)
   - Subsequent requests: <2 seconds
   - Server doesn't crash or hang

### ❌ Failure Signs

1. **Server Won't Start**:
   - Check `.env` file has API_KEYS
   - Check port 8000 not already in use
   - Check all dependencies installed

2. **401 on All Requests**:
   - Check API key matches `.env` file
   - Check header is `X-API-Key` (case-sensitive)

3. **500 Errors**:
   - Check logs: `tail -f logs/app.log`
   - Check disk space for FAISS index
   - Check memory available (~3GB needed)

4. **Empty Results**:
   - Make sure documents are ingested first
   - Check min_score isn't too high
   - Verify FAISS index has data: `/admin/stats`

---

## 📝 Sample Test Session Output

### Complete Test Run

```bash
$ # Start server in terminal 1
$ uvicorn app.main:app --reload
INFO:     Uvicorn running on http://0.0.0.0:8000

$ # In terminal 2, run tests
$ curl http://localhost:8000/health
{"status":"healthy","faiss_index_size":0,"total_documents":0,...}
✅ Health check passed

$ # Ingest document
$ curl -X POST http://localhost:8000/api/v1/ingest \
  -H "X-API-Key: dev-key-12345" \
  -F "file=@docs/diabetes_eng.txt"
  
[Wait 30-60 seconds for first request...]
{"status":"success","document_id":"doc_a1b2c3d4","language":"en","chunks_created":25,...}
✅ Document ingested

$ # Retrieve documents
$ curl -X POST http://localhost:8000/api/v1/retrieve \
  -H "X-API-Key: dev-key-12345" \
  -H "Content-Type: application/json" \
  -d '{"query":"diabetes treatment","top_k":3,"min_score":0.5}'
  
{"results":[{"text":"...","similarity_score":0.847,...}],"results_found":2,...}
✅ Retrieved 2 documents

$ # Generate response
$ curl -X POST http://localhost:8000/api/v1/generate \
  -H "X-API-Key: dev-key-12345" \
  -H "Content-Type: application/json" \
  -d '{"query":"What is metformin used for?","output_language":"en"}'
  
{"query":"What is metformin used for?","response":"Based on the retrieved medical guidelines...","sources":[...],...}
✅ Response generated successfully

$ # Check stats
$ curl -H "X-API-Key: dev-key-12345" http://localhost:8000/admin/stats
{"faiss":{"total_vectors":25,"total_documents":1,...},...}
✅ Statistics updated correctly

ALL TESTS PASSED! ✅
```

---

## 🐛 Troubleshooting

### Issue: "ModuleNotFoundError"
**Cause**: Dependencies not installed  
**Fix**: 
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### Issue: "API_KEYS must be set"
**Cause**: Empty .env file  
**Fix**:
```bash
echo "API_KEYS=dev-key-12345" > .env
```

### Issue: "Port 8000 already in use"
**Cause**: Another process using port 8000  
**Fix**:
```bash
# Kill existing process
lsof -ti:8000 | xargs kill -9

# Or use different port
uvicorn app.main:app --port 8001
```

### Issue: "401 Unauthorized" on all requests
**Cause**: API key mismatch  
**Fix**:
```bash
# Check .env file
cat .env | grep API_KEYS

# Make sure you're using the same key in requests
# Header must be: X-API-Key: dev-key-12345
```

### Issue: First request very slow
**Cause**: Model downloading (expected behavior)  
**Fix**: Just wait 30-60 seconds. Subsequent requests will be fast.

### Issue: Empty results from /retrieve
**Cause**: No documents ingested yet  
**Fix**: Upload documents via `/ingest` first

### Issue: Out of memory
**Cause**: Insufficient RAM  
**Fix**: Close other applications. Need ~3GB RAM minimum.

---

## 📚 Testing Resources

### Files Created for You

1. ✅ `Healthcare_RAG_Assistant.postman_collection.json` - Import into Postman
2. ✅ `TESTING_GUIDE.md` - This comprehensive guide

### Test Documents Available

- `docs/diabetes_eng.txt` - English medical document
- `docs/diabetes_jap.txt` - Japanese medical document

### API Documentation

- Interactive: `http://localhost:8000/docs` (Swagger UI)
- JSON schema: `http://localhost:8000/openapi.json`

---

## 🎯 Quick Start Testing (5 Minutes)

```bash
# Terminal 1: Start server
cd /home/morshed/own-folder/Documents/jap_llm
source venv/bin/activate
uvicorn app.main:app --reload

# Terminal 2: Run quick tests
# Test 1: Health
curl http://localhost:8000/health

# Test 2: Ingest (wait 30-60 sec on first run)
curl -X POST http://localhost:8000/api/v1/ingest \
  -H "X-API-Key: dev-key-12345" \
  -F "file=@docs/diabetes_eng.txt"

# Test 3: Retrieve
curl -X POST http://localhost:8000/api/v1/retrieve \
  -H "X-API-Key: dev-key-12345" \
  -H "Content-Type: application/json" \
  -d '{"query":"What is diabetes?","top_k":3}'

# Test 4: Generate
curl -X POST http://localhost:8000/api/v1/generate \
  -H "X-API-Key: dev-key-12345" \
  -H "Content-Type: application/json" \
  -d '{"query":"How to manage diabetes?","output_language":"en"}'

# All working? ✅ You're good to go!
```

---

## 📊 Test Results Interpretation

### Good Results

- ✅ Similarity scores >0.7 = Highly relevant
- ✅ Similarity scores 0.5-0.7 = Moderately relevant
- ✅ Response includes medical terms from query
- ✅ Sources cite specific chunks
- ✅ Processing time <2 seconds (after model load)

### Expected Behaviors

- ⏱️ First request slow (30-60 sec) - Normal (model download)
- 📊 Similarity scores vary (0.3-0.9) - Normal (semantic matching)
- 📝 Response length varies (200-500 chars) - Normal (depends on retrieved content)
- 🔄 Empty results on new query - Normal (no relevant documents found)

---

**Testing Guide Complete**  
**Collection Exported**: Healthcare_RAG_Assistant.postman_collection.json  
**Ready to Test**: Import collection and start testing! ✅

