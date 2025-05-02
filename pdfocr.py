import os
import pdfplumber
import pytesseract
from PIL import Image
import io
import fitz  # PyMuPDF
import warnings
import logging
import sys
import numpy as np
from PIL import ImageEnhance, ImageFilter
import cv2

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Configure Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def preprocess_image(image):
    """Enhance image quality for better OCR results"""
    try:
        # Convert to grayscale
        if image.mode != 'L':
            image = image.convert('L')
        
        # Convert to numpy array for OpenCV processing
        img_array = np.array(image)
        
        # Apply adaptive thresholding
        img_array = cv2.adaptiveThreshold(
            img_array, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Convert back to PIL Image
        image = Image.fromarray(img_array)
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2.0)
        
        # Enhance sharpness
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(2.0)
        
        # Apply slight blur to reduce noise
        image = image.filter(ImageFilter.GaussianBlur(radius=0.5))
        
        return image
    except Exception as e:
        logger.error(f"Error in image preprocessing: {str(e)}")
        return image

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF using pdfplumber"""
    try:
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                try:
                    # Try to extract text with error handling
                    page_text = page.extract_text()
                    if page_text and page_text.strip():
                        text += page_text + "\n"
                    else:
                        logger.warning(f"No text found on page {page_num} using pdfplumber")
                except Exception as e:
                    logger.error(f"Error extracting text from page {page_num}: {str(e)}")
                    continue
        return text.strip() if text else None
    except Exception as e:
        logger.error(f"Error processing PDF with pdfplumber: {str(e)}")
        return None

def extract_text_with_ocr(pdf_path):
    """Extract text from PDF using OCR (for scanned documents)"""
    try:
        text = ""
        doc = fitz.open(pdf_path)
        
        for page_num in range(len(doc)):
            try:
                page = doc[page_num]
                # Get the page as an image with error handling
                try:
                    # Increase DPI for better quality
                    zoom = 2  # Increase zoom for better quality
                    mat = fitz.Matrix(zoom, zoom)
                    pix = page.get_pixmap(matrix=mat)
                    
                    # Convert to PIL Image
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    
                    # Preprocess the image
                    img = preprocess_image(img)
                    
                    # Try different Tesseract configurations
                    configs = [
                        r'--oem 3 --psm 6 -l eng',  # Default
                        r'--oem 3 --psm 1 -l eng',  # Automatic page segmentation
                        r'--oem 3 --psm 3 -l eng',  # Fully automatic page segmentation
                    ]
                    
                    best_text = ""
                    for config in configs:
                        try:
                            page_text = pytesseract.image_to_string(img, config=config)
                            if len(page_text.strip()) > len(best_text.strip()):
                                best_text = page_text
                        except Exception as e:
                            logger.warning(f"Error with config {config}: {str(e)}")
                            continue
                    
                    if best_text.strip():
                        text += best_text + "\n"
                        logger.info(f"Successfully extracted text from page {page_num + 1}")
                    else:
                        logger.warning(f"No text found on page {page_num + 1} using OCR")
                        
                except Exception as e:
                    logger.error(f"Error processing page {page_num + 1} image: {str(e)}")
                    continue
            except Exception as e:
                logger.error(f"Error accessing page {page_num + 1}: {str(e)}")
                continue
            
        return text.strip() if text else None
    except Exception as e:
        logger.error(f"Error performing OCR: {str(e)}")
        return None

def process_downloads_folder():
    """Process all PDFs in the downloads folder"""
    downloads_dir = 'downloads'
    results = {}
    
    if not os.path.exists(downloads_dir):
        logger.error("Downloads directory not found!")
        return results
    
    pdf_files = [f for f in os.listdir(downloads_dir) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        logger.error("No PDF files found in downloads directory!")
        return results
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(downloads_dir, pdf_file)
        logger.info(f"\nProcessing {pdf_file}...")
        
        # Try regular text extraction first
        text = extract_text_from_pdf(pdf_path)
        
        # If no text was extracted, try OCR
        if not text:
            logger.info("No text found with regular extraction, trying OCR...")
            text = extract_text_with_ocr(pdf_path)
        
        if text:
            results[pdf_file] = text
            logger.info(f"Successfully extracted text from {pdf_file}")
        else:
            logger.error(f"Could not extract text from {pdf_file}")
    
    return results

def save_results(results):
    """Save extracted text to files"""
    if not os.path.exists('extracted_text'):
        os.makedirs('extracted_text')
    
    for filename, text in results.items():
        output_file = os.path.join('extracted_text', f"{os.path.splitext(filename)[0]}_text.txt")
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(text)
            logger.info(f"Saved extracted text to {output_file}")
        except Exception as e:
            logger.error(f"Error saving text for {filename}: {str(e)}")

def main():
    logger.info("Starting PDF text extraction and OCR process...")
    try:
        results = process_downloads_folder()
        
        if results:
            logger.info("\nSaving results...")
            save_results(results)
            logger.info("\nProcess completed successfully!")
        else:
            logger.error("\nNo text was extracted from any files.")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
