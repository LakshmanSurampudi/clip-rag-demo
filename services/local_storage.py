import os
import uuid
from typing import Optional
from pathlib import Path

# Directory for storing images
STATIC_DIR = Path(__file__).parent.parent / "static" / "images"


class LocalStorage:
    """Local file storage for images"""
    
    def __init__(self):
        self.static_dir = STATIC_DIR
        self.initialized = False
        
    def initialize(self):
        """Initialize local storage directory"""
        try:
            self.static_dir.mkdir(parents=True, exist_ok=True)
            self.initialized = True
            print(f"✅ Local storage initialized: {self.static_dir}")
        except Exception as e:
            print(f"❌ Error initializing local storage: {e}")
            self.initialized = False
    
    def upload_image(self, image_bytes: bytes, filename: str) -> Optional[str]:
        """
        Save image locally and return URL path
        
        Returns:
            Relative URL path (e.g., /static/images/abc123.png)
        """
        if not self.initialized:
            self.initialize()
        
        try:
            # Generate unique filename
            ext = filename.split(".")[-1] if "." in filename else "png"
            unique_filename = f"{uuid.uuid4().hex}.{ext}"
            file_path = self.static_dir / unique_filename
            
            # Write image
            with open(file_path, "wb") as f:
                f.write(image_bytes)
            
            # Return URL path
            return f"/static/images/{unique_filename}"
            
        except Exception as e:
            print(f"❌ Error saving image: {e}")
            return None
    
    def delete_image(self, image_url: str) -> bool:
        """Delete image from local storage"""
        try:
            filename = image_url.split("/")[-1]
            file_path = self.static_dir / filename
            
            if file_path.exists():
                file_path.unlink()
                return True
            return False
            
        except Exception as e:
            print(f"❌ Error deleting image: {e}")
            return False