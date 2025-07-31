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


def split_graph_into_subgraphs_with_labels(image_path, output_dir, caption=""):
    """
    Split a single image containing multiple subgraphs into individual subgraphs.
    Each subgraph will include its axis labels and be saved as a separate image.
    
    Args:
        image_path: Path to the input image
        output_dir: Directory to save the extracted subgraphs
        caption: Optional caption for the original graph
    
    Returns:
        List of dictionaries containing subgraph information
    """
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    import os
    
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image: {image_path}")
        return []
    
    # Convert to grayscale for processing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply preprocessing to enhance graph detection
    # Remove noise and enhance edges
    denoised = cv2.fastNlMeansDenoising(gray)
    
    # Apply adaptive thresholding to handle varying lighting
    thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY_INV, 11, 2)
    
    # Morphological operations to clean up the image
    kernel = np.ones((2,2), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    
    # Find contours to identify potential graph regions
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours based on size and aspect ratio with improved criteria
    potential_graphs = []
    min_area = 2000  # Increased minimum area for better quality
    max_area = gray.shape[0] * gray.shape[1] * 0.9  # Increased maximum area
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area < area < max_area:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h
            
            # More flexible aspect ratio for different graph types
            if 0.3 < aspect_ratio < 4.0:
                # Additional quality checks
                # Check if the region has reasonable dimensions
                if w > 50 and h > 50:  # Minimum dimensions
                    # Calculate contour perimeter to area ratio (compactness)
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        compactness = (perimeter * perimeter) / area
                        # Prefer more compact shapes (typical of graphs)
                        if compactness < 100:  # Reasonable compactness threshold
                            potential_graphs.append((x, y, w, h, area))
    
    # If no contours found, try alternative methods
    if not potential_graphs:
        potential_graphs = detect_graphs_by_lines(gray)
    
    # Sort by area (larger areas are more likely to be graphs)
    potential_graphs.sort(key=lambda x: x[4], reverse=True)
    
    # Extract subgraphs with their labels
    subgraphs = []
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    
    for i, (x, y, w, h, area) in enumerate(potential_graphs[:10]):  # Limit to top 10 candidates
        # Add moderate padding around the detected region to capture axis labels
        padding_x = max(47, w // 6)  # Reduced by 4% from 49 to 47
        padding_y = max(47, h // 6)  # Reduced by 4% from 49 to 47
        
        # Ensure all coordinates are integers
        x, y, w, h = int(x), int(y), int(w), int(h)
        padding_x, padding_y = int(padding_x), int(padding_y)
        
        x1 = max(0, x - padding_x)
        y1 = max(0, y - padding_y)
        x2 = min(image.shape[1], x + w + padding_x)
        y2 = min(image.shape[0], y + h + padding_y)
        
        # Extract the subgraph region
        subgraph_region = image[y1:y2, x1:x2]
        
        if subgraph_region.size == 0:
            continue
        
        # Create a more comprehensive region that includes potential labels
        expanded_region = expand_region_for_labels(image, x1, y1, x2, y2)
        
        # Try to detect axis labels in the expanded region for better accuracy
        axis_labels = detect_axis_labels_enhanced(expanded_region)
        
        # Save the subgraph
        subgraph_filename = f"{base_filename}_subgraph_{i+1}.jpeg"
        subgraph_path = os.path.join(output_dir, subgraph_filename)
        
        # Save the expanded region
        cv2.imwrite(subgraph_path, expanded_region)
        
        # Create caption for this subgraph with enhanced information
        subgraph_caption = create_subgraph_caption_enhanced(caption, i+1, axis_labels, base_filename)
        
        # Save caption
        caption_path = os.path.splitext(subgraph_path)[0] + '.txt'
        with open(caption_path, 'w') as f:
            f.write(subgraph_caption)
        
        subgraphs.append({
            'path': subgraph_path,
            'caption': subgraph_caption,
            'x_label': axis_labels.get('x', ''),
            'y_label': axis_labels.get('y', ''),
            'region': (x1, y1, x2, y2)
        })
    
    return subgraphs

def detect_graphs_by_lines(gray_image):
    """
    Alternative method to detect graphs by looking for grid lines and axes.
    
    Args:
        gray_image: Grayscale image
        
    Returns:
        List of potential graph regions
    """
    import cv2
    import numpy as np
    
    # Edge detection
    edges = cv2.Canny(gray_image, 50, 150, apertureSize=3)
    
    # Line detection using Hough transform
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=10)
    
    if lines is None:
        return []
    
    # Group lines by orientation to find horizontal and vertical lines
    horizontal_lines = []
    vertical_lines = []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
        
        if abs(angle) < 10:  # Horizontal lines
            horizontal_lines.append((y1, y2))
        elif abs(angle - 90) < 10 or abs(angle + 90) < 10:  # Vertical lines
            vertical_lines.append((x1, x2))
    
    # Find graph regions based on line intersections
    potential_regions = []
    
    if horizontal_lines and vertical_lines:
        # Find the most common horizontal and vertical positions
        y_positions = [y for line in horizontal_lines for y in [line[0], line[1]]]
        x_positions = [x for line in vertical_lines for x in [line[0], line[1]]]
        
        # Group nearby positions
        y_groups = group_nearby_positions(y_positions, threshold=20)
        x_groups = group_nearby_positions(x_positions, threshold=20)
        
        # Create regions from line groups
        for i in range(len(y_groups) - 1):
            for j in range(len(x_groups) - 1):
                y1, y2 = y_groups[i], y_groups[i+1]
                x1, x2 = x_groups[j], x_groups[j+1]
                
                if x2 - x1 > 50 and y2 - y1 > 50:  # Minimum size
                    area = (x2 - x1) * (y2 - y1)
                    potential_regions.append((x1, y1, x2 - x1, y2 - y1, area))
    
    return potential_regions

def group_nearby_positions(positions, threshold=20):
    """
    Group positions that are close to each other.
    
    Args:
        positions: List of positions
        threshold: Distance threshold for grouping
        
    Returns:
        List of grouped positions
    """
    if not positions:
        return []
    
    positions = sorted(positions)
    groups = []
    current_group = [positions[0]]
    
    for pos in positions[1:]:
        if abs(pos - current_group[-1]) <= threshold:
            current_group.append(pos)
        else:
            groups.append(sum(current_group) / len(current_group))  # Average position
            current_group = [pos]
    
    groups.append(sum(current_group) / len(current_group))
    return sorted(groups)

def detect_axis_labels_enhanced(image_region):
    """
    Enhanced axis label detection using multiple methods with improved accuracy.
    
    Args:
        image_region: The image region to analyze
        
    Returns:
        Dictionary with 'x' and 'y' axis labels
    """
    import cv2
    import numpy as np
    from PIL import Image
    
    # Convert to grayscale
    gray = cv2.cvtColor(image_region, cv2.COLOR_BGR2GRAY)
    
    # Apply OCR to the entire region with better configuration
    try:
        # Use better OCR configuration for scientific text
        custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,()[]{}°μ%+-='
        text = pytesseract.image_to_string(Image.fromarray(gray), config=custom_config).lower()
    except:
        text = ""
    
    # Enhanced axis label patterns with more variations
    x_labels = [
        'strain', 'time', 'temperature', 'displacement', 'elongation', 'extension',
        'deformation', 'cycle', 'frequency', 'wavelength', 'energy', 'distance',
        'position', 'angle', 'rotation', 'velocity', 'acceleration', 'force',
        'load', 'pressure', 'volume', 'area', 'length', 'width', 'height'
    ]
    y_labels = [
        'stress', 'load', 'force', 'modulus', 'strength', 'hardness',
        'pressure', 'voltage', 'current', 'power', 'amplitude', 'intensity',
        'strain', 'deformation', 'displacement', 'velocity', 'acceleration',
        'temperature', 'energy', 'power', 'efficiency', 'density', 'mass'
    ]
    
    detected_labels = {'x': '', 'y': ''}
    
    # Method 1: Direct text search with improved matching
    for x_label in x_labels:
        if x_label in text:
            detected_labels['x'] = x_label
            break
    
    for y_label in y_labels:
        if y_label in text:
            detected_labels['y'] = y_label
            break
    
    # Method 2: Region-based detection with larger search areas and better differentiation
    if not detected_labels['x'] or not detected_labels['y']:
        height, width = gray.shape
        
        # Bottom region (X-axis) - larger area
        bottom_height = min(120, height // 4)  # Use 1/4 of height or 120px, whichever is smaller
        bottom_region = gray[max(0, height-bottom_height):height, :]
        try:
            bottom_text = pytesseract.image_to_string(Image.fromarray(bottom_region), config=custom_config).lower()
            for x_label in x_labels:
                if x_label in bottom_text:
                    detected_labels['x'] = x_label
                    break
        except:
            pass
        
        # Left region (Y-axis) - larger area
        left_width = min(150, width // 4)  # Use 1/4 of width or 150px, whichever is smaller
        left_region = gray[:, :left_width]
        try:
            left_text = pytesseract.image_to_string(Image.fromarray(left_region), config=custom_config).lower()
            for y_label in y_labels:
                if y_label in left_text:
                    detected_labels['y'] = y_label
                    break
        except:
            pass
        
        # Method 2.5: Check for axis labels in specific regions with better differentiation
        # Look for Y-axis labels in the left-middle region (avoiding top and bottom)
        if not detected_labels['y']:
            left_middle_region = gray[height//4:3*height//4, :left_width]
            try:
                left_middle_text = pytesseract.image_to_string(Image.fromarray(left_middle_region), config=custom_config).lower()
                for y_label in y_labels:
                    if y_label in left_middle_text:
                        detected_labels['y'] = y_label
                        break
            except:
                pass
        
        # Look for X-axis labels in the bottom-middle region (avoiding left and right edges)
        if not detected_labels['x']:
            bottom_middle_region = gray[max(0, height-bottom_height):height, width//4:3*width//4]
            try:
                bottom_middle_text = pytesseract.image_to_string(Image.fromarray(bottom_middle_region), config=custom_config).lower()
                for x_label in x_labels:
                    if x_label in bottom_middle_text:
                        detected_labels['x'] = x_label
                        break
            except:
                pass
    
    # Method 3: Look for units and common patterns with improved detection
    if not detected_labels['x'] or not detected_labels['y']:
        # Enhanced unit patterns
        units_patterns = {
            'x': ['%', 'mm', 'μm', 'nm', 's', 'min', 'hr', '°c', 'k', 'deg', 'rad', 'm', 'cm'],
            'y': ['mpa', 'gpa', 'n', 'kn', 'psi', 'ksi', 'v', 'a', 'w', 'j', 'kj', 'cal', 'btu']
        }
        
        for axis, patterns in units_patterns.items():
            for pattern in patterns:
                if pattern in text:
                    # Try to find the label before the unit
                    words = text.split()
                    for i, word in enumerate(words):
                        if pattern in word and i > 0:
                            potential_label = words[i-1]
                            # Clean up the potential label
                            potential_label = ''.join(c for c in potential_label if c.isalnum() or c.isspace())
                            if len(potential_label.strip()) > 2:  # Ensure it's substantial
                                if axis == 'x' and not detected_labels['x']:
                                    detected_labels['x'] = potential_label.strip()
                                elif axis == 'y' and not detected_labels['y']:
                                    detected_labels['y'] = potential_label.strip()
                            break
    
    # Method 4: Look for common scientific terms that might indicate axis labels
    if not detected_labels['x'] or not detected_labels['y']:
        scientific_terms = {
            'x': ['axis', 'abscissa', 'horizontal', 'independent'],
            'y': ['axis', 'ordinate', 'vertical', 'dependent', 'response']
        }
        
        for axis, terms in scientific_terms.items():
            for term in terms:
                if term in text:
                    # Look for words near these terms
                    lines = text.split('\n')
                    for line in lines:
                        if term in line:
                            words = line.split()
                            for word in words:
                                if len(word) > 3 and word.isalpha():
                                    if axis == 'x' and not detected_labels['x']:
                                        detected_labels['x'] = word
                                    elif axis == 'y' and not detected_labels['y']:
                                        detected_labels['y'] = word
                                    break
    
    # Method 5: Post-processing to avoid duplicate labels and improve accuracy
    # If both axes have the same label, try to differentiate them
    if detected_labels['x'] == detected_labels['y'] and detected_labels['x']:
        # Common case: if both are "strain", one might actually be "stress"
        if detected_labels['x'] == 'strain':
            # Check if "stress" appears anywhere in the text
            if 'stress' in text:
                detected_labels['y'] = 'stress'
            # Check if "load" appears anywhere in the text
            elif 'load' in text:
                detected_labels['y'] = 'load'
            # Check if "force" appears anywhere in the text
            elif 'force' in text:
                detected_labels['y'] = 'force'
            else:
                # If no better alternative found, keep one as strain and set the other to a generic term
                detected_labels['y'] = 'response'
        
        # Similar logic for other common duplicates
        elif detected_labels['x'] == 'time':
            if 'temperature' in text:
                detected_labels['y'] = 'temperature'
            elif 'pressure' in text:
                detected_labels['y'] = 'pressure'
            else:
                detected_labels['y'] = 'value'
    
    return detected_labels

def expand_region_for_labels(image, x1, y1, x2, y2, expansion=75):
    """
    Expand the detected region to include potential axis labels with wider margins.
    
    Args:
        image: Full image
        x1, y1, x2, y2: Current region coordinates
        expansion: Pixels to expand in each direction (increased default)
        
    Returns:
        Expanded image region
    """
    height, width = image.shape[:2]
    
    # Ensure all coordinates are integers
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    
    # Expand the region moderately to capture axis labels
    # Left expansion (for Y-axis labels)
    left_expansion = int(expansion * 1.13)   # Reduced by 4% from 1.18 to 1.13
    # Right expansion (for additional labels)
    right_expansion = int(expansion * 0.94)  # Reduced by 4% from 0.98 to 0.94
    # Top expansion (for title/header)
    top_expansion = int(expansion * 0.57)    # Reduced by 4% from 0.59 to 0.57
    # Bottom expansion (for X-axis labels)
    bottom_expansion = int(expansion * 0.94) # Reduced by 4% from 0.98 to 0.94
    
    new_x1 = max(0, x1 - left_expansion)
    new_y1 = max(0, y1 - top_expansion)
    new_x2 = min(width, x2 + right_expansion)
    new_y2 = min(height, y2 + bottom_expansion)
    
    # Ensure all slice indices are integers
    new_x1, new_y1, new_x2, new_y2 = int(new_x1), int(new_y1), int(new_x2), int(new_y2)
    
    return image[new_y1:new_y2, new_x1:new_x2]

def create_subgraph_caption_enhanced(original_caption, subgraph_number, axis_labels, base_filename):
    """
    Create an enhanced caption for a subgraph with detailed axis information.
    
    Args:
        original_caption: Original graph caption
        subgraph_number: Number of this subgraph
        axis_labels: Dictionary with 'x' and 'y' axis labels
        base_filename: Original filename for reference
        
    Returns:
        Formatted caption for the subgraph
    """
    caption_parts = []
    
    # Add subgraph identifier with source reference
    caption_parts.append(f"Subgraph {subgraph_number} (extracted from {base_filename})")
    
    # Add detailed axis information
    axis_info = []
    if axis_labels.get('x'):
        axis_info.append(f"X-axis: {axis_labels['x'].title()}")
    if axis_labels.get('y'):
        axis_info.append(f"Y-axis: {axis_labels['y'].title()}")
    
    if axis_info:
        caption_parts.append(f"Axes: {', '.join(axis_info)}")
    else:
        caption_parts.append("Axis labels: Not detected")
    
    # Add original caption if available
    if original_caption and original_caption.strip():
        # Clean up the original caption
        clean_caption = original_caption.strip()
        if len(clean_caption) > 300:
            clean_caption = clean_caption[:300] + "..."
        caption_parts.append(f"Original caption: {clean_caption}")
    else:
        caption_parts.append("No original caption available")
    
    # Add extraction metadata
    caption_parts.append(f"Extracted using computer vision algorithms")
    
    return ". ".join(caption_parts) + "."

def create_subgraph_caption(original_caption, subgraph_number, axis_labels):
    """
    Create a caption for a subgraph based on the original caption and detected axis labels.
    
    Args:
        original_caption: Original graph caption
        subgraph_number: Number of this subgraph
        axis_labels: Dictionary with 'x' and 'y' axis labels
        
    Returns:
        Formatted caption for the subgraph
    """
    caption_parts = []
    
    # Add subgraph identifier
    caption_parts.append(f"Subgraph {subgraph_number}")
    
    # Add axis information if available
    if axis_labels.get('x') or axis_labels.get('y'):
        axis_info = []
        if axis_labels.get('x'):
            axis_info.append(f"X-axis: {axis_labels['x']}")
        if axis_labels.get('y'):
            axis_info.append(f"Y-axis: {axis_labels['y']}")
        caption_parts.append(f"({', '.join(axis_info)})")
    
    # Add original caption if available
    if original_caption:
        # Truncate if too long
        if len(original_caption) > 200:
            original_caption = original_caption[:200] + "..."
        caption_parts.append(f"From: {original_caption}")
    
    return ". ".join(caption_parts) + "."
