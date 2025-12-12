# Multimodal RAG with CLIP

A multimodal Retrieval-Augmented Generation (RAG) system that processes PDFs to extract and embed both text and images using CLIP (Contrastive Language-Image Pre-training). Enables semantic search across text and images, and provides AI-powered answers to questions about uploaded images.

## Features

- **PDF Ingestion**: Extract text and images from PDFs
- **CLIP Embeddings**: 512-dimensional embeddings for both text and images
- **Unified Search**: Query across text and images simultaneously
- **Image-to-Image Search**: Find similar images using visual similarity
- **AI-Powered Image Analysis**: Ask questions about uploaded images with context-aware answers
- **Vector Storage**: Pinecone for scalable vector search

## Prerequisites

- Python 3.11+
- Pinecone account and API key
- Google AI API key (for Gemini)

## Installation

### 1. Create Project Directory

Create a new folder for your project and open it in VS Code:

**macOS/Linux:**
```bash
mkdir multimodal-rag-workspace
cd multimodal-rag-workspace
code .
```

**Windows:**
```bash
mkdir multimodal-rag-workspace
cd multimodal-rag-workspace
code .
```

Open the integrated terminal in VS Code (Terminal → New Terminal).

### 2. Clone the Repository

In the VS Code terminal, clone the repository:

```bash
git clone <repository-url>
cd <repository-name>
```

**Note:** All subsequent commands should be run from inside this cloned repository folder.

### 3. Create Virtual Environment

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

You should see `(venv)` prefix in your terminal after activation.

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install all required packages including PyTorch, Transformers, FastAPI, etc. This may take several minutes.

### 5. Get API Keys

#### Pinecone API Key
- Visit: https://app.pinecone.io/
- Sign up or log in to your account
- Navigate to **API Keys** section in the left sidebar
- Click **Create API Key** and copy the generated key
- Note your cloud provider (AWS/GCP/Azure) and region (e.g., `us-east-1`)

#### Google AI API Key
- Visit: https://makersuite.google.com/app/apikey
- Sign in with your Google account
- Click **Create API Key** or **Get API Key**
- Copy the generated API key

### 6. Configure Environment Variables

Copy the template and fill in your keys:

**macOS/Linux:**
```bash
cp .env.template .env
```

**Windows:**
```bash
copy .env.template .env
```

Edit `.env` file (open in VS Code):
```bash
# Pinecone Configuration
PINECONE_API_KEY=your_actual_pinecone_api_key
PINECONE_CLOUD=aws                    # or gcp, azure
PINECONE_REGION=us-east-1             # your Pinecone region
PINECONE_CLIP_INDEX=multimodal-rag-clip-index

# Google AI Configuration
GOOGLE_API_KEY=your_actual_google_api_key
```

### 6. Create Directory Structure

```bash
mkdir -p static/images routes services
```

Ensure your project structure looks like:
```
your-project/
├── main.py
├── requirements.txt
├── .env
├── .gitignore
├── routes/
│   └── clip_routes.py
├── services/
│   ├── clip_service.py
│   └── local_storage.py
└── static/
    └── images/
```

## Running the Application

### Local Development

Start the FastAPI server:

```bash
python main.py
```

Or with uvicorn directly:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at:
- **API Base**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

**Note**: The CLIP model will download on first startup (~500MB). This may take 2-3 minutes.

## API Endpoints

### 1. Ingest PDF
```bash
POST /clip/ingest
```
Upload a PDF to extract and embed text and images.

**Example:**
```bash
curl -X POST "http://localhost:8000/clip/ingest" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your-document.pdf"
```

### 2. Unified Query (Text/Image Search)
```bash
POST /clip/query?query=citrus%20leaf%20disease&top_k=10
```
Search for semantically similar text and images.

**Example:**
```bash
curl -X POST "http://localhost:8000/clip/query?query=treatment%20methods&top_k=5"
```

### 3. Ask with Image
```bash
POST /clip/ask-with-image
```
Upload an image and ask questions about it.

**Example:**
```bash
curl -X POST "http://localhost:8000/clip/ask-with-image" \
  -F "file=@image.jpg" \
  -F "query=What disease is shown in this image?"
```

### 4. Image-to-Image Search
```bash
POST /clip/query-by-image
```
Upload an image to find visually similar images.

**Example:**
```bash
curl -X POST "http://localhost:8000/clip/query-by-image?top_k=5" \
  -F "file=@reference-image.jpg"
```

## Project Architecture

**Core Components:**
- **FastAPI**: REST API framework
- **CLIP (OpenAI)**: Multimodal embeddings (text + images)
- **Pinecone**: Vector database for similarity search
- **Google Gemini**: LLM for generating answers
- **PyMuPDF**: PDF text and image extraction
- **LangChain**: Text splitting and LLM orchestration

**Data Flow:**
1. PDF uploaded → text and images extracted
2. CLIP encodes both text chunks and images into 512-dim vectors
3. Vectors stored in Pinecone with metadata (type, source, page)
4. Images saved locally in `/static/images/`
5. Query → CLIP embedding → Pinecone search → results returned
6. Image questions → CLIP embedding → context retrieval → LLM answer

## Production Deployment

### Basic Production Setup

1. **Use Production WSGI Server:**
   ```bash
   pip install gunicorn
   gunicorn main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
   ```

2. **Set Environment Variables:**
   - Never commit `.env` files to version control
   - Use environment variables directly or secret management services
   - Set `LANGSMITH_TRACING=false` in production

3. **Configure CORS:**
   - Update `allow_origins` in `main.py` to specific domains instead of `["*"]`
   ```python
   app.add_middleware(
       CORSMiddleware,
       allow_origins=["https://yourdomain.com"],  # Specific domains
       allow_credentials=True,
       allow_methods=["*"],
       allow_headers=["*"],
   )
   ```

4. **Use Reverse Proxy:**
   - Deploy behind Nginx/Apache for SSL termination and load balancing
   - Example Nginx config:
   ```nginx
   server {
       listen 80;
       server_name yourdomain.com;
       
       location / {
           proxy_pass http://localhost:8000;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
       }
   }
   ```

5. **Persistent Storage:**
   - For production, consider using cloud storage (S3, GCS) instead of local `/static/images/`
   - Modify `local_storage.py` to integrate with cloud storage SDK

6. **Scaling Considerations:**
   - CLIP model loading is memory-intensive (~2GB RAM)
   - Use GPU instances for faster embedding generation
   - Consider caching frequently queried embeddings
   - Pinecone scales automatically with serverless indexes

7. **Monitoring:**
   - Enable health checks: `/health` and `/ready` endpoints
   - Monitor Pinecone usage and query latency
   - Track API response times and error rates

### Cloud Platform Deployment

**AWS EC2 / Google Cloud VM:**
- Use t3.medium or larger (minimum 4GB RAM)
- Open port 8000 (or configure with reverse proxy on port 80/443)
- Use systemd service for auto-restart

**Heroku:**
- Requires Procfile: `web: gunicorn main:app --workers 2 --worker-class uvicorn.workers.UvicornWorker`
- Set Config Vars for environment variables

**Railway / Render:**
- Auto-detects FastAPI applications
- Set environment variables in dashboard
- Configure health check endpoint: `/health`

## Troubleshooting

**CLIP model download fails:**
- Check internet connection
- Hugging Face may be temporarily unavailable
- Model cache: `~/.cache/huggingface/`

**Pinecone index creation fails:**
- Verify API key and region match
- Check if index already exists with different dimensions
- Free tier limits: 1 index, 100k vectors

**Out of memory errors:**
- CLIP model requires ~2GB RAM minimum
- Reduce batch size or use smaller model
- Consider upgrading instance size

**Images not displaying:**
- Check `/static/images/` directory exists and is writable
- Verify `STATIC_DIR` path in `local_storage.py`
- Check file permissions (755 for directories, 644 for files)

## License

[Your License Here]

## Contributing

[Your Contributing Guidelines Here]