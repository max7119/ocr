from fastapi import FastAPI, Body, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import cv2
import numpy as np
import base64
import io
import re
import logging
from typing import Optional, List, Dict, Any
import concurrent.futures
from functools import lru_cache

logging.basicConfig(level=logging.INFO)
import platform, os
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

class OCRProcessor:
    """Hochperformante OCR-Verarbeitungsklasse mit mehreren Optimierungsstufen"""
    
    def __init__(self):
        # Tesseract-Konfiguration für maximale Genauigkeit
        self.tesseract_configs = {
            "best": "--oem 1 --psm 3 -c tessedit_char_whitelist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz/-., ' -c tessedit_min_word_length=1",
            "numbers_focused": "--oem 1 --psm 6 -c tessedit_char_whitelist='0123456789/-., ' -c tessedit_min_word_length=1", 
            "lstm": "--oem 1 --psm 3",
            "combined": "--oem 1 --psm 3"  # Statt --oem 2
        }

        
        # Thread-Pool für parallele Verarbeitung
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
    
    def preprocess_image_advanced(self, img: Image.Image) -> List[np.ndarray]:
        """Erweiterte Bildvorverarbeitung mit mehreren Techniken"""
        
        # Konvertierung zu OpenCV-Format
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        processed_images = []
        
        # 1. Basis-Preprocessing
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        processed_images.append(gray)
        
        # 2. Hochskalierung für bessere OCR-Erkennung (2x)
        height, width = gray.shape[:2]
        upscaled = cv2.resize(gray, (width*2, height*2), interpolation=cv2.INTER_CUBIC)
        
        # 3. Adaptive Threshold (für Text mit variablem Hintergrund)
        adaptive_thresh = cv2.adaptiveThreshold(
            upscaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        processed_images.append(adaptive_thresh)
        
        # 4. OTSU Binarization
        _, otsu_thresh = cv2.threshold(upscaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        processed_images.append(otsu_thresh)
        
        # 5. Morphologische Operationen (Rauschentfernung)
        kernel = np.ones((1, 1), np.uint8)
        morph = cv2.morphologyEx(otsu_thresh, cv2.MORPH_CLOSE, kernel)
        morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
        processed_images.append(morph)
        
        # 6. Deskewing (Schräglage korrigieren)
        deskewed = self.deskew_image(upscaled)
        processed_images.append(deskewed)
        
        # 7. Denoising
        denoised = cv2.fastNlMeansDenoising(upscaled, None, 10, 7, 21)
        processed_images.append(denoised)
        
        # 8. Kontrastverbesserung mit CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(upscaled)
        processed_images.append(enhanced)
        
        # 9. Schärfung
        kernel_sharpen = np.array([[-1,-1,-1],
                                   [-1, 9,-1],
                                   [-1,-1,-1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel_sharpen)
        processed_images.append(sharpened)
        
        # 10. Bilateral Filter (erhält Kanten, reduziert Rauschen)
        bilateral = cv2.bilateralFilter(upscaled, 9, 75, 75)
        processed_images.append(bilateral)
        
        return processed_images
    
    def deskew_image(self, image: np.ndarray) -> np.ndarray:
        """Korrigiert Bildschräglage"""
        coords = np.column_stack(np.where(image > 0))
        if len(coords) == 0:
            return image
            
        angle = cv2.minAreaRect(coords)[-1]
        
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
            
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), 
                                 flags=cv2.INTER_CUBIC, 
                                 borderMode=cv2.BORDER_REPLICATE)
        
        return rotated
    
    def preprocess_with_pil(self, img: Image.Image) -> List[Image.Image]:
        """PIL-basierte Bildvorverarbeitung"""
        processed_images = []
        
        # Originalbild
        processed_images.append(img)
        
        # Graustufenkonvertierung
        if img.mode != 'L':
            gray = img.convert('L')
            processed_images.append(gray)
        else:
            gray = img
        
        # Kontrastverbesserung
        enhancer = ImageEnhance.Contrast(gray)
        for factor in [1.5, 2.0, 2.5]:
            enhanced = enhancer.enhance(factor)
            processed_images.append(enhanced)
        
        # Schärfung
        sharpened = gray.filter(ImageFilter.SHARPEN)
        processed_images.append(sharpened)
        
        # Edge Enhancement
        edge_enhanced = gray.filter(ImageFilter.EDGE_ENHANCE_MORE)
        processed_images.append(edge_enhanced)
        
        # Helligkeitsanpassung
        brightness = ImageEnhance.Brightness(gray)
        for factor in [0.8, 1.2]:
            bright_adjusted = brightness.enhance(factor)
            processed_images.append(bright_adjusted)
        
        return processed_images
    
    def extract_roi_regions(self, img_cv: np.ndarray) -> List[np.ndarray]:
        """Extrahiert Regions of Interest (potenzielle Textbereiche)"""
        regions = []
        
        # Konturfindung
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY) if len(img_cv.shape) == 3 else img_cv
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # Mindestgröße für relevante Bereiche
                x, y, w, h = cv2.boundingRect(contour)
                # Padding hinzufügen
                padding = 10
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(gray.shape[1] - x, w + 2*padding)
                h = min(gray.shape[0] - y, h + 2*padding)
                
                roi = gray[y:y+h, x:x+w]
                if roi.size > 0:
                    regions.append(roi)
        
        return regions
    
    def parallel_ocr(self, images: List[np.ndarray], lang: str) -> List[str]:
        """Führt OCR parallel auf mehreren Bildvarianten aus"""
        
        def ocr_single(img_data):
            img, config = img_data
            try:
                if isinstance(img, np.ndarray):
                    img_pil = Image.fromarray(img)
                else:
                    img_pil = img
                
                text = pytesseract.image_to_string(img_pil, lang=lang, config=config)
                return text
            except Exception as e:
                logging.error(f"OCR-Fehler: {e}")
                return ""
        
        # Verschiedene Tesseract-Konfigurationen auf jedes Bild anwenden
        tasks = []
        for img in images:
            for config in self.tesseract_configs.values():
                tasks.append((img, config))
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(ocr_single, tasks))
        
        return results
    
    def ensemble_ocr(self, img: Image.Image, lang: str = "deu") -> Dict[str, Any]:
        """Ensemble-Methode: Kombiniert mehrere OCR-Ansätze für beste Genauigkeit"""
        
        all_results = []
        
        # 1. OpenCV-basierte Vorverarbeitung
        cv_processed = self.preprocess_image_advanced(img)
        cv_results = self.parallel_ocr(cv_processed, lang)
        all_results.extend(cv_results)
        
        # 2. PIL-basierte Vorverarbeitung
        pil_processed = self.preprocess_with_pil(img)
        pil_results = self.parallel_ocr(
            [np.array(p) for p in pil_processed], 
            lang
        )
        all_results.extend(pil_results)
        
        # 3. ROI-basierte Verarbeitung
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        roi_regions = self.extract_roi_regions(img_cv)
        if roi_regions:
            roi_results = self.parallel_ocr(roi_regions[:5], lang)  # Max 5 ROIs
            all_results.extend(roi_results)
        
        # 4. Verschiedene Seitensegmentierungsmodi
        psm_modes = [3, 6, 7, 8, 11, 12, 13]
        for psm in psm_modes:
            try:
                config = f"--oem 3 --psm {psm}"
                text = pytesseract.image_to_string(img, lang=lang, config=config)
                all_results.append(text)
            except:
                pass
        
        return self.aggregate_results(all_results)

    def format_text_with_coords(self, img: Image.Image, lang: str = "deu", config: str = "--oem 1 --psm 6") -> str:
        """Formatiert OCR-Text anhand von Wort-Koordinaten (ähnlich Power Automate Logik)."""
        try:
            data = pytesseract.image_to_data(img, lang=lang, config=config, output_type=pytesseract.Output.DICT)
            n = len(data.get("text", []))
            words = []
            for i in range(n):
                text = data["text"][i].strip()
                if not text:
                    continue
                words.append({
                    "block": data.get("block_num", [0])[i],
                    "para": data.get("par_num", [0])[i],
                    "line": data.get("line_num", [0])[i],
                    "word": data.get("word_num", [0])[i],
                    "left": data.get("left", [0])[i] / max(1, data.get("width", [1])[i]),
                    "x": data.get("left", [0])[i],
                    "y": data.get("top", [0])[i],
                    "w": data.get("width", [0])[i],
                    "h": data.get("height", [0])[i],
                    "text": text,
                })

            # Gruppieren nach Zeilen (block/para/line)
            from collections import defaultdict
            lines = defaultdict(list)
            for w in words:
                key = (w["block"], w["para"], w["line"])
                lines[key].append(w)

            # Zeilen nach y sortieren, Wörter in Zeile nach x
            sorted_line_keys = sorted(lines.keys(), key=lambda k: (min(x["y"] for x in lines[k]), min(x["x"] for x in lines[k])))
            output_lines = []
            for key in sorted_line_keys:
                items = sorted(lines[key], key=lambda w: w["x"])
                if not items:
                    continue
                # Einfaches Spacing: Anzahl Spaces anhand Lücke relativ zur durchschnittl. Zeichenbreite
                avg_char_w = max(1.0, sum(it["w"] for it in items) / max(1, sum(len(it["text"]) for it in items)))
                line_buf = []
                prev_right = items[0]["x"]
                for idx, it in enumerate(items):
                    if idx == 0:
                        line_buf.append(it["text"])
                        prev_right = it["x"] + it["w"]
                        continue
                    gap = max(0, it["x"] - prev_right)
                    spaces = int(round(gap / avg_char_w))
                    line_buf.append((" " * max(1, spaces)) + it["text"])
                    prev_right = it["x"] + it["w"]
                output_lines.append("".join(line_buf))
            return "\n".join(output_lines)
        except Exception as e:
            logging.error("Formatting error: %s", e)
            # Fallback auf normalen Text
            try:
                return pytesseract.image_to_string(img, lang=lang, config=config)
            except Exception:
                return ""
    
    def aggregate_results(self, results: List[str]) -> Dict[str, Any]:
        """Aggregiert und wertet mehrere OCR-Ergebnisse aus"""
        
        # Sammle alle gefundenen Nummern
        all_numbers = []
        number_confidence = {}
        
        # Match 8 digits followed by slash OR any (unicode) space(s) then digits
        # Includes normal spaces, NBSP (\u00A0), and thin/zero-width spaces (\u2000-\u200B)
        pattern = r"(\d{8})(?:[\s\u00A0\u2000-\u200B]*[\/\s\u00A0\u2000-\u200B]+\d+)"
        
        for text in results:
            if text:
                matches = re.findall(pattern, text)
                for match in matches:
                    all_numbers.append(match)
                    # Zähle Vorkommen für Konfidenz
                    number_confidence[match] = number_confidence.get(match, 0) + 1
        
        # Sortiere nach Häufigkeit (höchste Konfidenz zuerst)
        sorted_numbers = sorted(
            number_confidence.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Bereite Ergebnis vor: nur die beste Nummer zurückgeben
        unique_numbers = [sorted_numbers[0][0]] if sorted_numbers else []
        
        # Finde den längsten zusammenhängenden Text
        longest_text = max(results, key=len) if results else ""
        
        # Berechne Konfidenzwerte relativ zum besten Treffer (Top == 100)
        max_count = sorted_numbers[0][1] if sorted_numbers else 0
        confidence_scores = {
            num: (100.0 if max_count == 0 else round((count / max_count) * 100.0, 2))
            for num, count in sorted_numbers
        }
        
        return {
            "numbers": unique_numbers,
            "confidence_scores": confidence_scores,
            "raw_text": longest_text.strip(),
            "total_variants_processed": len(results),
            "consensus_strength": (max(confidence_scores.values()) if confidence_scores else 0)
        }

# Globale Prozessor-Instanz
ocr_processor = OCRProcessor()

@app.post("/ocr")
async def ocr(base64_image: str = Body(..., embed=True), lang: str = Body("deu")):
    """Hochperformanter OCR-Endpoint mit Ensemble-Methode"""
    try:
        logging.info("OCR-Request gestartet...")
        
        # Base64-Dekodierung
        if "," in base64_image:
            base64_image = base64_image.split(",")[1]
        
        # Padding korrigieren
        missing_padding = len(base64_image) % 4
        if missing_padding:
            base64_image += "=" * (4 - missing_padding)
        
        img_bytes = base64.b64decode(base64_image)
        img = Image.open(io.BytesIO(img_bytes))
        logging.info("Bildformat: %s, Größe: %s", img.format, img.size)
        
        # Ensemble-OCR ausführen
        result = ocr_processor.ensemble_ocr(img, lang)

        # Nachgelagerte Formatierung des Textes und Regex nur darauf anwenden
        formatted_text = ocr_processor.format_text_with_coords(img, lang, config="--oem 1 --psm 6")
        pattern = r"(\d{8})(?:[\s\u00A0\u2000-\u200B]*[\/\s\u00A0\u2000-\u200B]+\d+)"
        formatted_matches = re.findall(pattern, formatted_text)
        if formatted_matches:
            best_num = formatted_matches[0]
            result["numbers"] = [best_num]
            result["confidence_scores"] = {best_num: 100.0}
            result["consensus_strength"] = 100.0
        result["raw_text"] = formatted_text.strip()
        
        logging.info("OCR abgeschlossen. Gefundene Nummern: %s", result["numbers"])
        logging.info("Konfidenz: %s", result["confidence_scores"])
        
        return result
        
    except Exception as e:
        logging.error("Fehler: %s", e, exc_info=True)
        return {"error": str(e), "numbers": [], "raw_text": ""}

@app.post("/ocr-fast")
async def ocr_fast(base64_image: str = Body(..., embed=True), lang: str = Body("deu")):
    """Schneller OCR-Endpoint mit Basis-Preprocessing"""
    try:
        if "," in base64_image:
            base64_image = base64_image.split(",")[1]
        
        missing_padding = len(base64_image) % 4
        if missing_padding:
            base64_image += "=" * (4 - missing_padding)
        
        img_bytes = base64.b64decode(base64_image)
        img = Image.open(io.BytesIO(img_bytes))
        
        # Schnellere Formatierung über Koordinaten (direkt auf Originalbild)
        formatted_text = ocr_processor.format_text_with_coords(img, lang, config="--oem 1 --psm 6")
        pattern = r"(\d{8})\s*\/\s*\d+"
        matches = re.findall(pattern, formatted_text)
        best = [matches[0]] if matches else []
        return {"numbers": best, "raw_text": formatted_text.strip()}
        
    except Exception as e:
        logging.error("Fehler: %s", e, exc_info=True)
        return {"error": str(e)}

@app.get("/health")
async def health():
    """Health-Check Endpoint"""
    return {"status": "healthy", "ocr_engine": "optimized"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)


