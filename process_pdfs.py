import os
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
import uuid
import cv2
import numpy as np
import re



def get_image_rect(page, xref):
    """
    Returns the rectangle (fitz.Rect) of the image with the given xref on the page.
    Returns None if not found.
    """
    text_dict = page.get_text("dict")
    for block in text_dict.get("blocks", []):
        if block.get("type", None) == 1:  # 1 = image block
            # Some versions store xref directly in block
            if block.get("image") is not None:
                # If it's a dict, try get xref
                if isinstance(block["image"], dict) and block["image"].get("xref") == xref:
                    return fitz.Rect(block["bbox"])
            # Sometimes xref is directly in block (PyMuPDF >=1.22)
            if block.get("xref") == xref:
                return fitz.Rect(block["bbox"])
    return None

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
            try:
                text = pytesseract.image_to_string(Image.open(temp_path)).lower()
                x_found = any(x in text for x in x_labels)
                y_found = any(y in text for y in y_labels)
            except Exception as e:
                print(f"OCR error: {e}")
                os.remove(temp_path)
                continue
            
            if x_found and y_found:
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir, exist_ok=True)
                
                final_path = os.path.join(output_dir, img_filename)
                os.rename(temp_path, final_path)
                
                # Extract caption with improved method
                caption = extract_caption_near_image_improved(page, xref, doc, page_num)
                valid_graphs.append((final_path, caption))
            else:
                os.remove(temp_path)
    
    doc.close()
    return valid_graphs

def extract_caption_fallback(page):
    """
    Fallback: Return the first non-empty line near the bottom of the page that looks like a figure caption,
    or just return an empty string.
    """
    text_blocks = page.get_text("dict")
    possible_captions = []
    for block in text_blocks.get("blocks", []):
        if "lines" in block:
            block_text = ""
            for line in block["lines"]:
                for span in line.get("spans", []):
                    block_text += span.get("text", "")
            if block_text.strip().lower().startswith(("figure", "fig")) and len(block_text.strip()) > 10:
                possible_captions.append(block_text.strip())
    if possible_captions:
        return possible_captions[0]
    return ""


def extract_caption_near_image_improved(page, xref, doc, page_num):
    """
    Improved caption extraction that looks for captions spatially near the image
    and also searches adjacent pages if needed.
    """
    # Get image position
    img_rect = get_image_rect(page, xref)
    if img_rect is None:
        return extract_caption_fallback(page)
    
    # Search for captions in multiple ways
    caption_candidates = []
    
    # Method 1: Look for text blocks near the image
    text_blocks = page.get_text("dict")
    for block in text_blocks.get("blocks", []):
        if "lines" in block:
            block_rect = fitz.Rect(block["bbox"])
            
            # Check if text block is near the image
            if is_near_image(block_rect, img_rect):
                block_text = ""
                for line in block["lines"]:
                    for span in line.get("spans", []):
                        block_text += span.get("text", "") + " "
                
                # Check if this looks like a caption
                caption = extract_figure_caption_from_text(block_text.strip())
                if caption:
                    proximity_score = calculate_proximity_score(block_rect, img_rect)
                    caption_candidates.append((caption, proximity_score))
    
    # Method 2: Look for multi-line captions using text positioning
    multiline_caption = extract_multiline_caption_by_position(page, img_rect)
    if multiline_caption:
        caption_candidates.append((multiline_caption, 0.8))  # High priority for spatial match
    
    # Method 3: Search current page for figure references
    page_text = page.get_text()
    figure_captions = find_figure_captions_in_text(page_text)
    for caption in figure_captions:
        caption_candidates.append((caption, 0.5))  # Medium priority
    
    # Method 4: Search adjacent pages if no good caption found
    if not caption_candidates and len(doc) > 1:
        # Search previous page
        if page_num > 0:
            prev_page = doc[page_num - 1]
            prev_text = prev_page.get_text()
            prev_captions = find_figure_captions_in_text(prev_text)
            for caption in prev_captions:
                caption_candidates.append((caption, 0.3))  # Lower priority
        
        # Search next page
        if page_num < len(doc) - 1:
            next_page = doc[page_num + 1]
            next_text = next_page.get_text()
            next_captions = find_figure_captions_in_text(next_text)
            for caption in next_captions:
                caption_candidates.append((caption, 0.3))  # Lower priority
    
    # Return the best caption candidate
    if caption_candidates:
        # Sort by proximity score (higher is better)
        caption_candidates.sort(key=lambda x: x[1], reverse=True)
        return caption_candidates[0][0]
    
    return extract_caption_fallback(page)

def extract_multiline_caption_by_position(page, img_rect):
    """
    Extract multi-line captions by analyzing text positioning relative to image.
    """
    # Get all text with positioning info
    text_dict = page.get_text("dict")
    caption_lines = []
    
    # Look for text blocks below the image
    below_threshold = img_rect.y1 + 100  # Search area below image
    above_threshold = img_rect.y1 - 10   # Small buffer above
    
    potential_caption_blocks = []
    
    for block in text_dict.get("blocks", []):
        if "lines" not in block:
            continue
            
        block_rect = fitz.Rect(block["bbox"])
        
        # Check if block is positioned below the image
        if (block_rect.y0 >= above_threshold and 
            block_rect.y0 <= below_threshold and
            abs(block_rect.x0 - img_rect.x0) < img_rect.width * 1.2):  # Allow some horizontal tolerance
            
            block_text = ""
            for line in block["lines"]:
                line_text = ""
                for span in line.get("spans", []):
                    line_text += span.get("text", "")
                if line_text.strip():
                    block_text += line_text.strip() + " "
            
            if block_text.strip():
                distance_from_image = block_rect.y0 - img_rect.y1
                potential_caption_blocks.append((block_text.strip(), distance_from_image, block_rect))
    
    # Sort by distance from image (closer is better)
    potential_caption_blocks.sort(key=lambda x: x[1])
    
    # Try to build a complete caption from consecutive blocks
    if potential_caption_blocks:
        complete_caption = ""
        last_y_position = None
        
        for text, distance, rect in potential_caption_blocks[:5]:  # Check first 5 closest blocks
            # Check if this looks like part of a figure caption
            if (text.lower().startswith(('figure', 'fig.', 'fig ')) or 
                complete_caption or  # Continue if we already started a caption
                any(keyword in text.lower() for keyword in ['stress', 'strain', 'tensile', 'compression', 'yield'])):
                
                # Check if this block is reasonably close to the previous one (for multi-line captions)
                if last_y_position is None or abs(rect.y0 - last_y_position) < 30:
                    if complete_caption and not complete_caption.endswith(' '):
                        complete_caption += " "
                    complete_caption += text
                    last_y_position = rect.y1
                elif complete_caption:
                    # If we have a caption and this block is far away, stop here
                    break
        
        if complete_caption:
            return clean_caption_text(complete_caption)
    
    return ""

def extract_figure_caption_from_text(text):
    """
    Enhanced figure caption extraction with better multi-line handling.
    """
    text = text.strip()
    if not text:
        return ""
    
    # Common caption patterns - now with multi-line support
    patterns = [
        r'(?i)(figure?\s*\d+[\.\:\-\s]*[^\.]*\..*?)(?=\s*figure?\s*\d+|$)',  # Until next figure or end
        r'(?i)(fig[\.\s]*\d+[\.\:\-\s]*[^\.]*\..*?)(?=\s*fig[\.\s]*\d+|$)',
        r'(?i)(figure?\s*[a-z][\.\:\-\s]*[^\.]*\..*?)(?=\s*figure?\s*[a-z]|$)',
        r'(?i)(fig[\.\s]*[a-z][\.\:\-\s]*[^\.]*\..*?)(?=\s*fig[\.\s]*[a-z]|$)',
        # Patterns for captions without ending periods
        r'(?i)(figure?\s*\d+[\.\:\-\s]*.*?)(?=\s*\n\s*[A-Z]|\s*$)',
        r'(?i)(fig[\.\s]*\d+[\.\:\-\s]*.*?)(?=\s*\n\s*[A-Z]|\s*$)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.MULTILINE | re.DOTALL)
        if match:
            caption = match.group(1).strip()
            caption = clean_caption_text(caption)
            if len(caption) > 15:  # Ensure it's substantial
                return caption
    
    # If text contains stress/strain related terms, it might be a caption
    stress_strain_keywords = ['stress', 'strain', 'tensile', 'compression', 'yield', 'elastic', 'plastic', 'modulus']
    if any(keyword in text.lower() for keyword in stress_strain_keywords):
        # Look for sentence-like structures
        sentences = re.split(r'[.!?]+', text)
        valid_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 15 and any(keyword in sentence.lower() for keyword in stress_strain_keywords):
                valid_sentences.append(sentence)
        
        if valid_sentences:
            # Join first few sentences to form a complete caption
            caption = '. '.join(valid_sentences[:3]) + '.'
            return clean_caption_text(caption)
        
        # Fallback: return first substantial portion
        if len(text) > 15:
            return clean_caption_text(text[:300].strip() + ('...' if len(text) > 300 else ''))
    
    return ""

def find_figure_captions_in_text(text):
    """
    Enhanced function to find all figure captions in a block of text with better multi-line support.
    """
    captions = []
    
    # Split text into paragraphs (double newlines) first, then lines
    paragraphs = re.split(r'\n\s*\n', text)
    
    for paragraph in paragraphs:
        lines = paragraph.split('\n')
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line:
                i += 1
                continue
            
            # Look for figure caption patterns
            patterns = [
                r'(?i)^(figure?\s*\d+[\.\:\-\s]+.*?)$',
                r'(?i)^(fig[\.\s]*\d+[\.\:\-\s]+.*?)$',
                r'(?i)^(figure?\s*[a-z][\.\:\-\s]+.*?)$',
                r'(?i)^(fig[\.\s]*[a-z][\.\:\-\s]+.*?)$',
            ]
            
            caption_found = False
            for pattern in patterns:
                match = re.match(pattern, line)
                if match:
                    caption = match.group(1)
                    
                    # Try to get following lines that might be part of the caption
                    full_caption = caption
                    j = i + 1
                    
                    # Look ahead for continuation lines
                    while j < len(lines) and j < i + 5:  # Look at next 4 lines max
                        next_line = lines[j].strip()
                        if not next_line:
                            j += 1
                            continue
                            
                        # Stop if we hit another figure caption
                        if re.match(r'(?i)^(figure|fig[\.\s]*\d+|table|equation)', next_line):
                            break
                        
                        # Continue if line looks like a continuation
                        if (not re.match(r'^[A-Z][A-Z\s]*[A-Z]$', next_line) and  # Not all caps heading
                            not re.match(r'^\d+\.', next_line) and  # Not numbered list
                            len(next_line) > 3):  # Has substantial content
                            
                            # Check if it's likely a continuation
                            if (next_line[0].islower() or  # Starts with lowercase
                                full_caption.rstrip()[-1] not in '.!?' or  # Previous doesn't end with sentence
                                any(word in next_line.lower() for word in ['stress', 'strain', 'mpa', 'gpa', 'ksi', 'psi', 'mechanical'])):  # Contains relevant terms
                                
                                full_caption += ' ' + next_line
                                j += 1
                            else:
                                break
                        else:
                            break
                    
                    # Clean up caption
                    full_caption = clean_caption_text(full_caption)
                    if len(full_caption) > 20:  # Minimum length filter
                        captions.append(full_caption)
                    
                    caption_found = True
                    i = j  # Skip the lines we've processed
                    break
            
            if not caption_found:
                i += 1
    
    return captions

def clean_caption_text(text):
    """
    Clean and normalize caption text.
    """
    if not text:
        return ""
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove extra periods at the end
    text = re.sub(r'\.+$', '.', text)
    
    # Ensure proper sentence ending
    if text and not text.endswith(('.', '!', '?', ':')):
        text += '.'
    
    # Remove weird characters that sometimes appear
    text = re.sub(r'[^\w\s\.\,\;\:\!\?\(\)\-\+\=\[\]\/\%\°\μ\α\β\γ\δ\ε]', '', text)
    
    return text

def is_near_image(text_rect, img_rect, threshold=100):
    """
    Enhanced function to check if a text block is spatially near an image.
    Increased threshold for better multi-line caption detection.
    """
    # Check if text is below the image (most common for captions)
    if (text_rect.y0 >= img_rect.y1 - threshold and 
        text_rect.y0 <= img_rect.y1 + threshold * 3 and  # Increased search area
        abs(text_rect.x0 - img_rect.x0) < img_rect.width * 1.5):  # More horizontal tolerance
        return True
    
    # Check if text is above the image
    if (text_rect.y1 <= img_rect.y0 + threshold and 
        text_rect.y1 >= img_rect.y0 - threshold * 2 and
        abs(text_rect.x0 - img_rect.x0) < img_rect.width * 1.5):
        return True
    
    # Check for overlap or very close proximity
    if text_rect.intersects(img_rect):
        return True
    
    return False

def find_alloy_names(text):
    """
    Extract alloy names from a text string using regex patterns.
    """
    patterns = [
        r'\b([A-Z][a-z]?){2,}\b',                      # TaNbHfZrTi, AlCoCrFeNi
        r'\b([A-Z][a-z]?(?:-\d+[A-Z][a-z]?)+)\b',      # Mg-3Al-1Zn
        r'\bInconel\s*\d+\b',                          # Inconel 718
        r'\b[0-9]{2,3}L\b',                            # 316L
        r'\bTi-6Al-4V\b',                              # Ti-6Al-4V
    ]
    alloys = set()
    for pat in patterns:
        for match in re.findall(pat, text):
            if isinstance(match, tuple):
                alloy = ''.join(match)
            else:
                alloy = match
            if len(alloy) >= 5 and not alloy.islower() and not alloy.isupper():
                alloys.add(alloy)
    return list(alloys)

def classify_alloy(alloy):
    """
    Classify the alloy name into type, category, and give a short description.
    """
    HEA_elements = re.findall(r'[A-Z][a-z]?', alloy)
    if 'Inconel' in alloy:
        return ('Superalloy', 'Nickel-based superalloy', 
                'Inconel alloys are high-performance nickel-chromium-based superalloys known for strength and corrosion resistance at high temperatures.')
    elif re.match(r'.*316L.*', alloy):
        return ('Stainless Steel', 'Austenitic Stainless Steel', 
                '316L is a low-carbon, molybdenum-bearing austenitic stainless steel with enhanced corrosion resistance.')
    elif re.match(r'.*Ti-6Al-4V.*', alloy):
        return ('Titanium Alloy', 'Alpha-beta titanium alloy', 
                'Ti-6Al-4V is a popular titanium alloy known for high strength, light weight, and corrosion resistance.')
    elif len(HEA_elements) >= 5:
        return ('High Entropy Alloy', 'HEA', 
                'High Entropy Alloys (HEAs) are composed of five or more principal elements mixed in near-equal proportions, offering unique mechanical properties.')
    elif len(HEA_elements) >= 3:
        return ('Medium Entropy Alloy', 'MEA', 
                'Medium Entropy Alloys (MEAs) contain three or four principal elements, balancing complexity and performance.')
    elif '-' in alloy:
        return ('Conventional Alloy', 'Commercial/Engineering Alloy',
                'A commercial alloy containing major elements, often used in engineering applications.')
    else:
        return ('Unknown', 'Unknown', 'No description available.')

# === Example usage (call this after caption extraction!) ===

def print_alloy_info_from_caption(caption):
    alloys = find_alloy_names(caption)
    for alloy in alloys:
        alloy_type, category, desc = classify_alloy(alloy)
        print(f"Alloy Name: {alloy}\nType: {alloy_type}\nCategory: {category}\nDescription: {desc}\n---")
