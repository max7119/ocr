from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
import pytesseract
from PIL import Image
import cv2
import numpy as np
import base64
import io
import logging
import platform
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    os.environ["TESSDATA_PREFIX"] = r"C:\Program Files\Tesseract-OCR\tessdata"

app = FastAPI()

# CORS aktivieren
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def preprocess_image(img: Image.Image) -> Image.Image:
    """Einfache Bildvorverarbeitung für bessere OCR-Erkennung"""
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    # Hochskalierung für bessere Erkennung
    height, width = gray.shape[:2]
    upscaled = cv2.resize(gray, (width*2, height*2), interpolation=cv2.INTER_CUBIC)
    
    # OTSU-Binarization
    _, binary = cv2.threshold(upscaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Leichte Rauschentfernung
    denoised = cv2.fastNlMeansDenoising(binary, None, 10, 7, 21)
    
    return Image.fromarray(denoised)


@app.post("/ocr-fast")
async def ocr_fast(base64_image: str = Body(..., embed=True), lang: str = Body("deu")):
    """Schneller OCR-Endpoint - gibt nur den erkannten Text zurück"""
    try:
        logger.info("OCR Request received")
        
        # Base64-Dekodierung
        if "," in base64_image:
            base64_image = base64_image.split(",")[1]
        
        missing_padding = len(base64_image) % 4
        if missing_padding:
            base64_image += "=" * (4 - missing_padding)
        
        img_bytes = base64.b64decode(base64_image)
        img = Image.open(io.BytesIO(img_bytes))
        
        # Bildvorverarbeitung
        preprocessed_img = preprocess_image(img)
        
        # OCR ausführen
        text = pytesseract.image_to_string(
            preprocessed_img, 
            lang=lang, 
            config="--oem 1 --psm 6 -c preserve_interword_spaces=1"
        )
        
        logger.info(f"OCR completed, text length: {len(text)} characters")
        
        return {"text": text}
        
    except Exception as e:
        logger.error(f"Error in OCR endpoint: {str(e)}", exc_info=True)
        return {"error": str(e), "text": ""}


@app.get("/health")
async def health():
    """Health-Check Endpoint"""
    logger.info("Health check requested")
    return {"status": "healthy", "ocr_engine": "tesseract"}


if __name__ == "__main__":
    import uvicorn
    logger.info("Starting OCR API server...")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
