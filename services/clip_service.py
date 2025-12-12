import os
import io
import asyncio
from typing import List, Dict, Any, Optional
import requests
from dotenv import load_dotenv

# PDF and Image processing
import fitz  # pymupdf
from PIL import Image

# LangChain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI

# Pinecone
from pinecone import Pinecone, ServerlessSpec

# LangSmith
from langsmith import traceable

# Local Storage
from services.local_storage import LocalStorage

load_dotenv()


class ClipService:
    """
    CLIP-based multimodal RAG service:
    - Embeds both text and images with CLIP (512 dimensions)
    - Stores in single Pinecone index with type metadata
    - Supports unified search across text and images
    """
    
    def __init__(self):
        self.clip_model = None
        self.clip_processor = None
        self.pinecone_client = None
        self.clip_index = None
        self.llm = None
        self.local_storage = LocalStorage()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        self.initialized = False
        
    async def initialize(self):
        """Initialize all components"""
        print("🚀 Initializing CLIP Service...")
        
        try:
            # Initialize local storage
            self.local_storage.initialize()
            print("✅ Local storage initialized")
            
            # CLIP model (lazy loaded on first use)
            self.clip_model = None
            self.clip_processor = None
            print("📦 CLIP model will be loaded on first use")
            
            # Initialize LLM
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                google_api_key=os.getenv("GOOGLE_API_KEY"),
                temperature=0.3
            )
            print("✅ LLM initialized (gemini-2.5-flash)")
            
            # Initialize Pinecone
            self.pinecone_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
            
            # Setup CLIP index (512 dimensions)
            clip_index_name = os.getenv("PINECONE_CLIP_INDEX", "multimodal-rag-clip-index")
            self._ensure_index_exists(clip_index_name, 512)
            self.clip_index = self.pinecone_client.Index(clip_index_name)
            print(f"✅ CLIP index initialized: {clip_index_name}")
            
            # Pre-load CLIP model
            print("⏳ Pre-loading CLIP model (may take 2-3 minutes)...")
            self._ensure_clip_loaded()
            print("✅ CLIP model loaded and ready")
            
            self.initialized = True
            print("✅✅✅ CLIP Service initialized successfully! ✅✅✅")
            
        except Exception as e:
            print(f"❌ Error initializing CLIP Service: {str(e)}")
            raise e
    
    def _ensure_clip_loaded(self):
        """Lazy load CLIP model on first use"""
        if self.clip_model is None or self.clip_processor is None:
            print("Loading CLIP model from HuggingFace...")
            from transformers import CLIPProcessor, CLIPModel
            import torch
            
            model_name = "openai/clip-vit-base-patch32"
            self.clip_processor = CLIPProcessor.from_pretrained(model_name)
            self.clip_model = CLIPModel.from_pretrained(model_name)
            
            self.clip_model.eval()
            if torch.cuda.is_available():
                self.clip_model = self.clip_model.cuda()
                print("✅ CLIP model loaded on GPU")
            else:
                print("✅ CLIP model loaded on CPU")
    
    def _ensure_index_exists(self, index_name: str, dimension: int):
        """Ensure Pinecone index exists"""
        existing_indexes = [idx.name for idx in self.pinecone_client.list_indexes()]
        
        if index_name not in existing_indexes:
            print(f"Creating Pinecone index: {index_name}")
            self.pinecone_client.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud=os.getenv("PINECONE_CLOUD", "aws"),
                    region=os.getenv("PINECONE_REGION", "us-east-1")
                )
            )
    
    def extract_text_from_pdf(self, file_path: str) -> str:
        """Extract all text from PDF"""
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text() + "\n"
        doc.close()
        return text
    
    def extract_images_from_pdf(self, file_path: str) -> List[Dict[str, Any]]:
        """Extract images from PDF with context"""
        images = []
        doc = fitz.open(file_path)
        
        for page_num, page in enumerate(doc):
            page_text = page.get_text()
            page_context = page_text[:500] if page_text else ""
            
            image_list = page.get_images(full=True)
            
            for img_index, img_info in enumerate(image_list):
                xref = img_info[0]
                
                try:
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    
                    # Convert to PNG if needed
                    if image_ext.lower() not in ["png", "jpg", "jpeg"]:
                        img = Image.open(io.BytesIO(image_bytes))
                        buffer = io.BytesIO()
                        img.save(buffer, format="PNG")
                        image_bytes = buffer.getvalue()
                        image_ext = "png"
                    
                    images.append({
                        "image_bytes": image_bytes,
                        "page_num": page_num + 1,
                        "image_index": img_index,
                        "ext": image_ext,
                        "page_text": page_context
                    })
                    
                except Exception as e:
                    print(f"Error extracting image: {str(e)}")
                    continue
        
        doc.close()
        return images
    
    def embed_text(self, text: str) -> List[float]:
        """Embed text using CLIP"""
        import torch
        
        self._ensure_clip_loaded()
        
        inputs = self.clip_processor(text=[text], return_tensors="pt", padding=True, truncation=True)
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            text_features = self.clip_model.get_text_features(**inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_features.cpu().numpy()[0].tolist()
    
    def embed_image(self, image_bytes: bytes) -> List[float]:
        """Embed image using CLIP"""
        import torch
        
        self._ensure_clip_loaded()
        
        img = Image.open(io.BytesIO(image_bytes))
        if img.mode != "RGB":
            img = img.convert("RGB")
        
        inputs = self.clip_processor(images=img, return_tensors="pt")
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        return image_features.cpu().numpy()[0].tolist()
    
    @traceable(run_type="chain")
    async def process_pdf(self, file_path: str, filename: str) -> Dict[str, Any]:
        """Process PDF: extract text and images, embed with CLIP"""
        if not self.initialized:
            raise Exception("CLIP Service not initialized")
        
        results = {
            "filename": filename,
            "text_chunks": 0,
            "images_processed": 0,
            "images_stored": 0,
            "errors": []
        }
        
        try:
            # 1. Extract and process text
            print(f"Processing PDF: {filename}")
            text = self.extract_text_from_pdf(file_path)
            
            if text.strip():
                chunks = self.text_splitter.split_text(text)
                
                for i, chunk in enumerate(chunks):
                    try:
                        vector_id = f"{filename}_text_{i}"
                        clip_embedding = self.embed_text(chunk)
                        
                        self.clip_index.upsert(
                            vectors=[{
                                "id": vector_id,
                                "values": clip_embedding,
                                "metadata": {
                                    "type": "text",
                                    "source": filename,
                                    "chunk": i,
                                    "total_chunks": len(chunks),
                                    "content": chunk[:1000]
                                }
                            }]
                        )
                        
                    except Exception as e:
                        results["errors"].append(f"Text chunk {i}: {str(e)}")
                
                results["text_chunks"] = len(chunks)
                print(f"✅ Stored {len(chunks)} text chunks")
            
            # 2. Extract and process images
            images = self.extract_images_from_pdf(file_path)
            results["images_processed"] = len(images)
            
            for img_data in images:
                try:
                    # Save image locally
                    img_filename = f"{filename}_p{img_data['page_num']}_i{img_data['image_index']}.{img_data['ext']}"
                    image_url = self.local_storage.upload_image(
                        img_data["image_bytes"],
                        img_filename
                    )
                    
                    # Embed and store
                    vector_id = f"{filename}_img_{img_data['page_num']}_{img_data['image_index']}"
                    clip_embedding = self.embed_image(img_data["image_bytes"])
                    
                    self.clip_index.upsert(
                        vectors=[{
                            "id": vector_id,
                            "values": clip_embedding,
                            "metadata": {
                                "type": "image",
                                "source": filename,
                                "page": img_data["page_num"],
                                "image_index": img_data["image_index"],
                                "image_url": image_url or "",
                                "page_text": img_data["page_text"][:1000]
                            }
                        }]
                    )
                    
                    results["images_stored"] += 1
                    
                except Exception as e:
                    results["errors"].append(f"Image {img_data['page_num']}-{img_data['image_index']}: {str(e)}")
            
            print(f"✅ Processing complete: {results}")
            return results
            
        except Exception as e:
            raise Exception(f"Error processing PDF: {str(e)}")
    
    @traceable(run_type="chain")
    async def query_unified(
        self, 
        query: str, 
        top_k: int = 5,
        filter_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Unified search across text and images"""
        if not self.initialized:
            raise Exception("CLIP Service not initialized")
        
        try:
            self._ensure_clip_loaded()
            
            query_embedding = self.embed_text(query)
            
            filter_dict = None
            if filter_type:
                filter_dict = {"type": {"$eq": filter_type}}
            
            results = self.clip_index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                filter=filter_dict
            )
            
            unified_results = []
            for match in results.matches:
                result = {
                    "score": match.score,
                    "type": match.metadata.get("type", "unknown"),
                    "source": match.metadata.get("source", ""),
                    "content": match.metadata.get("content", "") if match.metadata.get("type") == "text" else match.metadata.get("page_text", "")
                }
                
                if match.metadata.get("type") == "image":
                    result.update({
                        "image_url": match.metadata.get("image_url", ""),
                        "page": match.metadata.get("page", 0),
                        "image_index": match.metadata.get("image_index", 0),
                        "page_text": match.metadata.get("page_text", "")
                    })
                elif match.metadata.get("type") == "text":
                    result.update({
                        "chunk": match.metadata.get("chunk", 0),
                        "total_chunks": match.metadata.get("total_chunks", 0)
                    })
                
                unified_results.append(result)
            
            return unified_results
            
        except Exception as e:
            raise Exception(f"Error in unified query: {str(e)}")
    
    @traceable(run_type="chain")
    async def ask_with_image(
        self, 
        image_bytes: Optional[bytes] = None, 
        query: str = "", 
        media_url: Optional[str] = None
    ) -> Dict[str, Any]:
        """Answer questions about an uploaded image"""
        if not self.initialized:
            raise Exception("CLIP Service not initialized")

        # Download from URL if needed
        if not image_bytes and media_url:
            try:
                headers = {'User-Agent': 'MultimodalRAG/1.0'}
                response = requests.get(media_url, headers=headers, timeout=10)
                response.raise_for_status()
                
                content_type = response.headers.get('Content-Type', '').lower()
                if 'text' in content_type or 'html' in content_type:
                    raise Exception("URL does not point to an image")
                
                image_bytes = response.content
            except Exception as e:
                raise Exception(f"Failed to download image: {str(e)}")

        if not image_bytes:
            raise Exception("Please provide an image")

        try:
            async def _process():
                self._ensure_clip_loaded()
                
                # Verify it's an image
                try:
                    img = Image.open(io.BytesIO(image_bytes))
                except Exception:
                    raise Exception("Could not decode as image")

                if img.mode != "RGB":
                    img = img.convert("RGB")
                
                # Embed uploaded image
                image_embedding = self.embed_image(image_bytes)
                
                # Search for similar content
                clip_results = self.clip_index.query(
                    vector=image_embedding,
                    top_k=10,
                    include_metadata=True
                )
                
                # Separate by type
                matched_images = []
                matched_texts = []
                
                for match in clip_results.matches:
                    if match.metadata.get("type") == "image":
                        matched_images.append({
                            "page_text": match.metadata.get("page_text", ""),
                            "image_url": match.metadata.get("image_url", ""),
                            "score": match.score
                        })
                    elif match.metadata.get("type") == "text":
                        matched_texts.append({
                            "content": match.metadata.get("content", ""),
                            "score": match.score
                        })
                
                # Build context
                image_contexts = [img["page_text"][:300] for img in matched_images[:3] if img["page_text"]]
                text_contexts = [txt["content"][:300] for txt in matched_texts[:3] if txt["content"]]
                image_urls = [img["image_url"] for img in matched_images[:3] if img["image_url"]]
                
                confidence = clip_results.matches[0].score if clip_results.matches else 0
                
                # Generate answer
                prompt = f"""You are an expert analyzing an uploaded image.

User's question: "{query}"

**Similar images found (context):**
{chr(10).join(image_contexts) or "No similar images found."}

**Similar text found:**
{chr(10).join(text_contexts) or "No similar text found."}

Top match confidence: {confidence:.0%}

Based on the matched content, provide a detailed answer to the user's question."""

                response = self.llm.invoke(prompt)
                
                return {
                    "answer": response.content,
                    "matched_sources": list(set([m.metadata.get("source", "") for m in clip_results.matches])),
                    "related_images": image_urls,
                    "confidence": confidence
                }

            return await asyncio.wait_for(_process(), timeout=120.0)
    
        except asyncio.TimeoutError:
            raise Exception("Processing timed out")
        except Exception as e:
            raise Exception(f"Error: {str(e)}")
    
    @traceable(run_type="chain")
    async def query_images_by_image(self, image_bytes: bytes, top_k: int = 5) -> List[Dict[str, Any]]:
        """Image-to-image search"""
        if not self.initialized:
            raise Exception("CLIP Service not initialized")
        
        try:
            self._ensure_clip_loaded()
            
            img = Image.open(io.BytesIO(image_bytes))
            if img.mode != "RGB":
                img = img.convert("RGB")
            
            image_embedding = self.embed_image(image_bytes)
            
            results = self.clip_index.query(
                vector=image_embedding,
                top_k=top_k,
                include_metadata=True,
                filter={"type": {"$eq": "image"}}
            )
            
            images = []
            for match in results.matches:
                images.append({
                    "score": match.score,
                    "image_url": match.metadata.get("image_url", ""),
                    "source": match.metadata.get("source", ""),
                    "page": match.metadata.get("page", 0),
                    "image_index": match.metadata.get("image_index", 0)
                })
            
            return images
            
        except Exception as e:
            raise Exception(f"Error: {str(e)}")

    @traceable(run_type="chain")
    async def ask_by_text(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Answer text-based questions using retrieved context"""
        if not self.initialized:
            raise Exception("CLIP Service not initialized")
        
        try:
            # Get similar content
            results = await self.query_unified(query, top_k=top_k)
            
            if not results:
                return {
                    "answer": "I couldn't find relevant information to answer this question.",
                    "sources": [],
                    "confidence": 0.0
                }
            
            # Build context from results
            contexts = []
            sources = set()
            
            for result in results:
                sources.add(result["source"])
                if result.get("content"):
                    contexts.append(result["content"][:500])
            
            # Generate answer
            context = "\n\n".join(contexts[:5])
            confidence = results[0]["score"] if results else 0.0
            
            prompt = f"""You are an expert assistant. Answer the user's question based on the provided context.

User's question: "{query}"

Context from documents:
{context}

Provide a clear, detailed answer based on the context above. If the context doesn't contain enough information, say so."""

            response = self.llm.invoke(prompt)
            
            return {
                "answer": response.content,
                "sources": list(sources),
                "confidence": confidence
            }
            
        except Exception as e:
            raise Exception(f"Error: {str(e)}")

# Singleton instance
clip_service = ClipService()