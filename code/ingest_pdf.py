import fitz  # PyMuPDF
import pytesseract
from pdf2image import convert_from_path

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = []
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text()
        
        if not text.strip():  # Probably scanned page, use OCR
            images = convert_from_path(pdf_path, first_page=page_num+1, last_page=page_num+1)
            ocr_text = pytesseract.image_to_string(images[0])
            full_text.append(ocr_text)
        else:
            full_text.append(text)
    
    return full_text

if __name__ == "__main__":
    pdf_file = "data/What_is_Policy.pdf"
    extracted_text = extract_text_from_pdf(pdf_file)
    for i, text in enumerate(extracted_text):
        print(f"--- Page {i+1} ---\n{text[:200]}...\n")
