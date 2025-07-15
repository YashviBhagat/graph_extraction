import os
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
import uuid
import cv2
import numpy as np

def extract_graphs_from_pdf(pdf_path, output_dir, x_labels=None, y_labels=None):
    """
    Extracts images from a PDF, filters them by axis text using OCR, and saves valid graphs to output_dir.
    Only creates output_dir if at least one valid graph is found.
    Returns a list of (image_path, caption) tuples for valid graphs.
    """
    if x_labels is None:
        x_labels = ['true strain', 'strain', 'engineering strain']
    if y_labels is None:
        y_labels = ['true stress', 'stress', 'engineering stress']
    doc = fitz.open(pdf_path)
    valid_graphs = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        image_list = page.get_images(full=True)
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image['image']
            ext = base_image['ext']
            img_filename = f"graph_{page_num+1}_{img_index+1}.{ext}"
            temp_path = os.path.join('/tmp', str(uuid.uuid4()) + '.' + ext)
            with open(temp_path, 'wb') as img_file:
                img_file.write(image_bytes)
            # OCR to check axis labels
            text = pytesseract.image_to_string(Image.open(temp_path)).lower()
            x_found = any(x in text for x in x_labels)
            y_found = any(y in text for y in y_labels)
            if x_found and y_found:
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir, exist_ok=True)
                final_path = os.path.join(output_dir, img_filename)
                os.rename(temp_path, final_path)
                caption = extract_caption_near_image(page, xref)
                valid_graphs.append((final_path, caption))
            else:
                os.remove(temp_path)
    return valid_graphs

def extract_caption_near_image(page, xref):
    """
    Attempts to extract a figure caption or number near the image in the PDF page.
    This is a heuristic and may not always be accurate.
    """
    # Simple approach: get all text blocks and look for 'fig' or 'figure'
    text = page.get_text()  # type: ignore
    for line in text.split('\n'):
        if line.strip().lower().startswith(('figure', 'fig.')):
            return line.strip()
    return ''

def split_graph_into_subgraphs_with_labels(image_path, output_dir):
    """
    Splits a graph image into sub-graphs (panels) and extracts their labels using OCR.
    Only works well for clear, well-separated sub-graphs (e.g., figures with panels a, b, c, ...).
    Returns a list of (subgraph_path, label) pairs.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Threshold to get binary image
    _, thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)
    # Find contours (external only)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    subgraphs = []
    for i, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        # Filter out small areas (noise)
        if w * h < 5000:
            continue
        # Crop sub-graph
        sub_img = img[y:y+h, x:x+w]
        sub_img_path = os.path.join(output_dir, f'subgraph_{i+1}.png')
        cv2.imwrite(sub_img_path, sub_img)
        # OCR for label (look in top-left corner of sub-graph)
        label_crop = sub_img[0:min(40, h), 0:min(40, w)]
        label = pytesseract.image_to_string(label_crop, config='--psm 8').strip()
        subgraphs.append((sub_img_path, label))
    # Optionally, sort subgraphs left-to-right, top-to-bottom
    subgraphs.sort(key=lambda x: (cv2.boundingRect(cv2.findContours(cv2.cvtColor(cv2.imread(x[0]), cv2.COLOR_BGR2GRAY), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0])[1],
                                  cv2.boundingRect(cv2.findContours(cv2.cvtColor(cv2.imread(x[0]), cv2.COLOR_BGR2GRAY), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0])[0]))
    return subgraphs 