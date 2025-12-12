from fastapi import APIRouter, UploadFile, File, HTTPException, Query, Form
from pydantic import BaseModel
from typing import List, Optional
import tempfile
import os
from services.clip_service import clip_service

router = APIRouter(prefix="/clip", tags=["Multimodal RAG"])


# ===== REQUEST/RESPONSE MODELS =====

class IngestResponse(BaseModel):
    message: str
    filename: str
    text_chunks: int
    images_processed: int
    images_stored: int
    errors: List[str]


class UnifiedResult(BaseModel):
    score: float
    type: str  # "text" or "image"
    source: str
    content: Optional[str] = None
    # Image-specific fields (optional)
    image_url: Optional[str] = None
    page: Optional[int] = None
    image_index: Optional[int] = None
    page_text: Optional[str] = None
    # Text-specific fields (optional)
    chunk: Optional[int] = None
    total_chunks: Optional[int] = None


class UnifiedQueryResponse(BaseModel):
    query: str
    results: List[UnifiedResult]
    count: int
    images_count: int
    texts_count: int


class ImageResult(BaseModel):
    score: float
    image_url: str
    source: str
    page: int
    image_index: int


class ImageSearchResponse(BaseModel):
    results: List[ImageResult]
    count: int


class AskWithImageResponse(BaseModel):
    answer: str
    matched_sources: List[str]
    related_images: List[str]
    confidence: float


# ===== ENDPOINTS =====

@router.post("/ingest", response_model=IngestResponse)
async def ingest_pdf(file: UploadFile = File(...)):
    """
    Ingest PDF with CLIP processing:
    1. Extract text and images separately
    2. Embed BOTH text and images with CLIP (512 dimensions)
    3. Store images locally in /static/images/
    4. Store all embeddings in single Pinecone CLIP index
    """
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        # Process the PDF
        results = await clip_service.process_pdf(temp_file_path, file.filename)
        
        # Clean up temporary file
        os.unlink(temp_file_path)
        
        return IngestResponse(
            message=f"PDF processed successfully. {results['text_chunks']} text chunks, {results['images_stored']} images stored.",
            filename=results["filename"],
            text_chunks=results["text_chunks"],
            images_processed=results["images_processed"],
            images_stored=results["images_stored"],
            errors=results["errors"]
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")


@router.post("/query", response_model=UnifiedQueryResponse)
async def unified_query(
    query: str = Query(..., description="Search query"),
    top_k: int = Query(10, description="Number of results to return"),
    filter_type: Optional[str] = Query(None, description="Filter by type: 'text', 'image', or None for both")
):
    """
    Unified search for similar content (text, images, or both)
    
    Uses CLIP embeddings to find semantically similar content.
    Examples:
    - "citrus leaf disease" → returns disease descriptions + disease images
    - "treatment methods" → returns treatment text + diagrams
    """
    if filter_type and filter_type not in ["text", "image"]:
        raise HTTPException(status_code=400, detail="filter_type must be 'text', 'image', or None")
    
    try:
        results = await clip_service.query_unified(query, top_k, filter_type)
        
        # Count by type
        images_count = sum(1 for r in results if r["type"] == "image")
        texts_count = sum(1 for r in results if r["type"] == "text")
        
        return UnifiedQueryResponse(
            query=query,
            results=[UnifiedResult(**r) for r in results],
            count=len(results),
            images_count=images_count,
            texts_count=texts_count
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in query: {str(e)}")


@router.post("/ask-with-image", response_model=AskWithImageResponse)
async def ask_with_image(
    file: Optional[UploadFile] = File(None),
    mediaUrl: Optional[str] = Form(None),
    query: str = Form("What is this image showing?")
):
    """
    Answer questions about an uploaded image:
    1. Upload a crop image OR provide mediaUrl
    2. CLIP finds similar content (both images AND text) in database
    3. LLM generates expert answer using matched content
    """
    # Validate input
    if not file and not mediaUrl:
        raise HTTPException(status_code=400, detail="Provide either an image file or a mediaUrl")

    # Validate file type if present
    if file:
        allowed_extensions = [".jpg", ".jpeg", ".png", ".webp"]
        file_ext = "." + file.filename.split(".")[-1].lower() if "." in file.filename else ""
        if file_ext not in allowed_extensions:
            raise HTTPException(status_code=400, detail="Only image files (JPEG, PNG, WebP) are allowed")
    
    try:
        image_bytes = None
        if file:
            image_bytes = await file.read()
        
        # Service handles media_url download if image_bytes is None
        result = await clip_service.ask_with_image(
            image_bytes=image_bytes, 
            query=query, 
            media_url=mediaUrl
        )
        
        return AskWithImageResponse(
            answer=result["answer"],
            matched_sources=result["matched_sources"],
            related_images=result["related_images"],
            confidence=result["confidence"]
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image query: {str(e)}")


@router.post("/query-by-image", response_model=ImageSearchResponse)
async def query_by_image(
    file: UploadFile = File(...),
    top_k: int = Query(5, description="Number of results")
):
    """
    Image-to-image search:
    1. Upload an image
    2. Find similar images in the database using CLIP
    3. Return matching images with URLs
    """
    # Validate file type
    allowed_extensions = [".jpg", ".jpeg", ".png", ".webp"]
    file_ext = "." + file.filename.split(".")[-1].lower() if "." in file.filename else ""
    if file_ext not in allowed_extensions:
        raise HTTPException(status_code=400, detail="Only image files (JPEG, PNG, WebP) are allowed")
    
    try:
        image_bytes = await file.read()
        images = await clip_service.query_images_by_image(image_bytes, top_k)
        
        return ImageSearchResponse(
            results=[ImageResult(**img) for img in images],
            count=len(images)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in image search: {str(e)}")

@router.post("/ask-by-text")
async def ask_by_text(
    query: str = Query(..., description="Your question"),
    top_k: int = Query(5, description="Number of context chunks to retrieve")
):
    """
    Ask a text-based question and get an AI-generated answer.
    
    Example: "What is citrus canker?" → Returns detailed LLM answer
    """
    try:
        result = await clip_service.ask_by_text(query, top_k)
        
        return {
            "query": query,
            "answer": result["answer"],
            "sources": result["sources"],
            "confidence": result["confidence"]
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")