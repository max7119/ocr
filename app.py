from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
import pytesseract
from PIL import Image
import base64, io, re
import logging

logging.basicConfig(level=logging.INFO)

app = FastAPI()

# 🔹 CORS aktivieren
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # für Tests offen lassen, später einschränken
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.post("/ocr")
async def ocr(base64_image: str = Body(..., embed=True), lang: str = Body("deu")):
    try:
        logging.info("OCR-Request gestartet...")
        if "," in base64_image:
            base64_image = base64_image.split(",")[1]

        logging.info("Base64 Länge vor Padding: %d", len(base64_image))
        missing_padding = len(base64_image) % 4
        if missing_padding:
            base64_image += "=" * (4 - missing_padding)
            logging.info("Padding ergänzt, neue Länge: %d", len(base64_image))

        img_bytes = base64.b64decode(base64_image)
        img = Image.open(io.BytesIO(img_bytes))
        logging.info("Bildformat: %s, Größe: %s", img.format, img.size)

        text = pytesseract.image_to_string(img, lang=lang)
        logging.info("OCR Ergebnis: %s", text)

        # 🔹 Regex für Muster: 8-stellige Zahl / weitere Zahl
        matches = re.findall(r"(\d{8})\s*\/\s*\d+", text)

        if matches:
            logging.info("Gefundene Nummern: %s", matches)
            return {"numbers": matches, "raw_text": text.strip()}
        else:
            logging.info("Keine passende Nummer gefunden.")
            return {"numbers": [], "raw_text": text.strip()}

    except Exception as e:
        logging.error("Fehler: %s", e, exc_info=True)
        return {"error": str(e)}
