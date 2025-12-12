from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
import os
import asyncio

# Load environment variables
load_dotenv()

# Import CLIP routes and service
from routes.clip_routes import router as clip_router
from services.clip_service import clip_service

app = FastAPI(
    title="Multimodal RAG API",
    version="1.0.0",
    description="CLIP-based multimodal RAG for text and image ingestion/querying"
)

# Readiness flag
services_ready = False
initialization_error = None

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for serving images
static_dir = os.path.join(os.path.dirname(__file__), "static", "images")
if not os.path.exists(static_dir):
    os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")

# Include CLIP router
app.include_router(clip_router)


async def initialize_services_background():
    """Initialize CLIP service in background"""
    global services_ready, initialization_error
    try:
        print("🚀 Starting CLIP service initialization...")
        await clip_service.initialize()
        services_ready = True
        print("✅✅✅ CLIP SERVICE READY ✅✅✅")
    except Exception as e:
        initialization_error = str(e)
        print(f"❌ Initialization failed: {e}")


@app.on_event("startup")
async def startup_event():
    """Start background initialization"""
    print("🌐 Server starting - port will open immediately")
    print("📦 CLIP service will initialize in background...")
    asyncio.create_task(initialize_services_background())


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Multimodal RAG API",
        "services_ready": services_ready
    }


@app.get("/ready")
async def readiness_check():
    """Readiness check - returns 503 if services are still initializing"""
    if initialization_error:
        return {"status": "error", "detail": initialization_error}, 503
    if not services_ready:
        return {"status": "initializing", "detail": "Services are still loading..."}, 503
    return {"status": "ready", "services_ready": True}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)