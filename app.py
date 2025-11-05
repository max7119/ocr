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
from concurrent.futures import ThreadPoolExecutor

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

# Thread-Pool für Preprocessing
executor = ThreadPoolExecutor(max_workers=2)


def preprocess_image(img: Image.Image) -> Image.Image:
    """Optimierte Bildvorverarbeitung für schnellere OCR-Erkennung"""
    # Konvertierung zu Graustufen
    if img.mode == 'RGB':
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    elif img.mode == 'L':
        gray = np.array(img)
    else:
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    height, width = gray.shape[:2]
    
    # Bedingte Hochskalierung: Nur bei kleinen Bildern (< 1000px)
    # Große Bilder werden nicht hochskaliert, um Zeit zu sparen
    if width < 1000 or height < 1000:
        upscaled = cv2.resize(gray, (width*2, height*2), interpolation=cv2.INTER_CUBIC)
    else:
        # Bei großen Bildern direkt verwenden, spart Zeit
        upscaled = gray
    
    # OTSU-Binarization (schnell und effektiv)
    _, binary = cv2.threshold(upscaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Denoising nur bei sehr kleinen Bildern oder bei hohem Rauschen
    # Bei größeren Bildern überspringen wir Denoising für Geschwindigkeit
    if width * height < 500000:  # Nur bei kleinen Bildern (< ~700x700px)
        # Schnellere Denoising-Parameter
        denoised = cv2.fastNlMeansDenoising(binary, None, 5, 5, 7)
        return Image.fromarray(denoised)
    
    return Image.fromarray(binary)


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
        
        # Bildvorverarbeitung (kann parallel laufen, aber hier sequenziell für Einfachheit)
        preprocessed_img = preprocess_image(img)
        
        # OCR ausführen mit optimierter Config:
        # --oem 3: LSTM OCR Engine (schneller als --oem 1 bei ähnlicher Qualität)
        # --psm 6: Einheitlicher Textblock (schneller als andere Modi)
        # preserve_interword_spaces=1: Behält Abstände zwischen Wörtern
        text = pytesseract.image_to_string(
            preprocessed_img, 
            lang=lang, 
            config="--oem 3 --psm 6 -c preserve_interword_spaces=1"
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
