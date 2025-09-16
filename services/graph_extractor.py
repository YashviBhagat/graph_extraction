import os
import fitz
from PIL import Image, ImageDraw, ImageFont
import pytesseract
import uuid
import cv2
import numpy as np
import re
import logging
import shutil
import io

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

class GraphExtractor:
    """Precise stress-strain graph extractor with strict axis label validation."""
    
    def __init__(self):
        # Comprehensive stress-strain axis patterns - must be exact matches
        self.x_labels = [
            # Basic strain patterns
            'engineering strain', 'true strain', 'strain',
            'engineering strain (%)', 'true strain (%)', 'strain (%)',
            'engineering strain(-)', 'true strain(-)', 'strain(-)',
            'engineering strain (', 'true strain (', 'strain (',
            'engineering strain(', 'true strain(', 'strain(',
            # With symbols
            'true strain(ε)', 'strain(ε)', 'engineering strain(ε)',
            'true strain (ε)', 'strain (ε)', 'engineering strain (ε)',
            'true strain(ε)', 'strain(ε)', 'engineering strain(ε)',
            # With commas and symbols
            'engineering strain, ε', 'engineering strain, ε (',
            'strain, ε',
            # Symbol with comma and unit
            'ε, %',
            # Symbols only
            'ε', 'epsilon'
        ]
        
        self.y_labels = [
            # Basic stress patterns
            'engineering stress', 'true stress', 'tensile stress', 'stress',
            # With MPa units
            'engineering stress (mpa)', 'true stress (mpa)', 'tensile stress (mpa)',
            'engineering stress (mpa)', 'true stress (mpa)', 'tensile stress (mpa)',
            'engineering stress(mpa)', 'true stress(mpa)', 'tensile stress(mpa)',
            # With GPa units
            'engineering stress (gpa)', 'true stress (gpa)', 'tensile stress (gpa)',
            'engineering stress (gpa)', 'true stress (gpa)', 'tensile stress (gpa)',
            'engineering stress(gpa)', 'true stress(gpa)', 'tensile stress(gpa)',
            # With parentheses
            'engineering stress (', 'true stress (', 'tensile stress (',
            'stress (', 'engineering stress(', 'true stress(', 'tensile stress(',
            # With commas and symbols
            'engineering stress, σ', 'engineering stress, σ (',
            'stress, σ',
            # Symbol with comma and unit
            'σ, mpa', 'σ, gpa',
            # Symbols only
            'σ', 'sigma'
        ]
        
        # Patterns to exclude non-stress-strain graphs
        self.exclude_patterns = [
            'x-ray', 'xrd', 'diffraction', 'intensity', '2theta', 'theta', 'degrees',
            'crystallographic', 'plane', 'indices', 'peak', 'peaks', 'reflection',
            'sem', 'tem', 'microscopy', 'micrograph', 'edx', 'elemental', 'composition',
            'ftir', 'raman', 'spectroscopy', 'absorbance', 'transmittance',
            'voltage', 'current', 'power', 'resistance', 'impedance', 'frequency',
            'temperature', 'heat', 'thermal', 'conductivity', 'enthalpy',
            'time', 'rate', 'kinetics', 'concentration', 'ph', 'catalyst',
            'hardness', 'modulus', 'toughness', 'fatigue', 'creep', 'fracture',
            'displacement', 'load', 'elongation', 'force'
        ]
        
        self.min_confidence = 0.70 # Higher threshold for better accuracy
    
    def _is_valid_stress_strain_graph(self, detected_text, x_found, y_found):
        """Strict validation to ensure it's actually a stress-strain graph."""
        text_lower = detected_text.lower()
        
        # Check for exclusion patterns first
        for pattern in self.exclude_patterns:
            if re.search(r'\b' + re.escape(pattern) + r'\b', text_lower):
                logger.info(f"Excluding graph due to pattern: {pattern}")
                return False, f"Excluded due to {pattern} pattern"
        
        # MUST have at least one axis label to be considered
        if not x_found and not y_found:
            return False, "No axis labels found"
        
        # Check for stress-strain specific context
        has_stress_strain_context = False
        
        # Look for stress-strain specific terms
        stress_strain_terms = [
            'stress', 'strain', 'engineering', 'true', 'tensile',
            'mpa', 'gpa', 'pa', 'pascal', 'ksi', 'psi',
            'ε', 'sigma', 'σ', '%', 'percent'
        ]
        
        for term in stress_strain_terms:
            if term in text_lower:
                has_stress_strain_context = True
                break
        
        # A graph is considered valid if it has BOTH axis labels AND stress-strain context
        if (x_found and y_found) and has_stress_strain_context:
            return True, "Valid stress-strain graph (both axes found)"
        elif (x_found or y_found) and has_stress_strain_context:
            return True, "Valid stress-strain graph (one axis found)"
        
        return False, "No stress-strain context or insufficient axis labels found"

    def _extract_text_around_image(self, page, image_rect, margin=50):
        """Extracts text within a margin around the image."""
        try:
            # Create a simple expanded rectangle
            x0, y0, x1, y1 = image_rect
            expanded_rect = fitz.Rect(x0 - margin, y0 - margin, x1 + margin, y1 + margin)
            
            # Get text blocks within the expanded rectangle
            text_blocks = page.get_text("dict", clip=expanded_rect)["blocks"]
            
            extracted_text = []
            for block in text_blocks:
                if block['type'] == 0:  # Text block
                    for line in block["lines"]:
                        for span in line["spans"]:
                            extracted_text.append(span["text"])
            return " ".join(extracted_text)
        except Exception as e:
            logger.warning(f"Error extracting text around image: {e}")
            return ""

    def _detect_axis_labels(self, image_path, image_text_context):
        """Detects stress and strain labels in the image context."""
        x_axis_label = ""
        y_axis_label = ""
        
        try:
            # Use OCR on the image itself for direct labels
            img = Image.open(image_path)
            img_text = pytesseract.image_to_string(img, config='--psm 6').lower()
            
            # Combine image OCR text with surrounding text context
            full_text_context = (img_text + " " + image_text_context).lower()
            
            # Check for X-axis labels using exact pattern matching
            for pattern in self.x_labels:
                if re.search(r'\b' + re.escape(pattern) + r'\b', full_text_context):
                    x_axis_label = pattern
                    break
            
            # Check for Y-axis labels using exact pattern matching
            for pattern in self.y_labels:
                if re.search(r'\b' + re.escape(pattern) + r'\b', full_text_context):
                    y_axis_label = pattern
                    break
            
            return x_axis_label, y_axis_label, full_text_context
        except Exception as e:
            logger.warning(f"Error in axis label detection: {e}")
            return "", "", image_text_context.lower()

    def extract_graphs_from_pdf(self, pdf_path, output_folder):
        """
        Extracts stress-strain graphs from a PDF, saves them as images,
        and returns their metadata.
        """
        logger.info(f"Starting precise stress-strain graph extraction from: {pdf_path}")
        
        doc = fitz.open(pdf_path)
        graphs_metadata = []
        
        # Ensure output folder exists
        os.makedirs(output_folder, exist_ok=True)

        for page_num, page in enumerate(doc):
            image_list = page.get_images(full=True)
            
            for img_index, img_info in enumerate(image_list):
                xref = img_info[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                
                # Save image temporarily to run OCR
                temp_image_path = os.path.join(output_folder, f"temp_graph_{page_num}_{img_index}.jpeg")
                try:
                    with open(temp_image_path, "wb") as f:
                        f.write(image_bytes)
                except Exception as e:
                    logger.warning(f"Error saving temporary image {temp_image_path}: {e}")
                    continue

                # Try to get image rectangle, but don't fail if we can't
                try:
                    image_rect = self._get_image_rect_simple(page, xref)
                    if image_rect:
                        image_text_context = self._extract_text_around_image(page, image_rect)
                    else:
                        # Fallback: get text from entire page
                        image_text_context = page.get_text()
                except Exception as e:
                    logger.warning(f"Error getting image context: {e}")
                    image_text_context = page.get_text()

                # Detect axis labels and get full text context
                x_axis_label, y_axis_label, full_text_context = self._detect_axis_labels(temp_image_path, image_text_context)
                
                # Validate if it's a stress-strain graph
                is_valid, validation_message = self._is_valid_stress_strain_graph(full_text_context, bool(x_axis_label), bool(y_axis_label))
                
                if is_valid:
                    # Calculate confidence based on axis detection and overall context
                    confidence = 0.0
                    if x_axis_label and y_axis_label:
                        confidence = 0.95 # Very high confidence if both axes are explicitly found
                    elif x_axis_label or y_axis_label:
                        confidence = 0.80 # High confidence if one axis is found
                    
                    # Further adjust confidence based on general stress-strain terms in context
                    stress_strain_terms_in_context = sum(1 for term in ['stress', 'strain', 'mpa', 'gpa', 'ε', 'σ'] if term in full_text_context)
                    if stress_strain_terms_in_context > 2:
                        confidence = max(confidence, 0.75) # Ensure good confidence if context is strong

                    # Only include if confidence is above threshold
                    if confidence >= self.min_confidence:
                        # Generate a unique filename for the graph
                        graph_filename = f"graph_{page_num+1}_{img_index+1}.jpeg"
                        final_graph_path = os.path.join(output_folder, graph_filename)
                        
                        # Move the temporary image to its final name
                        shutil.move(temp_image_path, final_graph_path)

                        graphs_metadata.append({
                            'page': page_num + 1,
                            'filename': graph_filename,
                            'x_axis_label': x_axis_label,
                            'y_axis_label': y_axis_label,
                            'confidence': round(confidence, 2),
                            'validation_message': validation_message
                        })
                        logger.info(f"✅ STRESS-STRAIN GRAPH FOUND: {graph_filename}")
                        logger.info(f"   X-axis: '{x_axis_label}'")
                        logger.info(f"   Y-axis: '{y_axis_label}'")
                        logger.info(f"   Confidence: {confidence:.2f}")
                        logger.info(f"   Validation: {validation_message}")
                    else:
                        logger.info(f"❌ Graph rejected: Low confidence ({confidence:.2f}) for image on page {page_num+1}, index {img_index}")
                        os.remove(temp_image_path)
                else:
                    logger.info(f"❌ Graph rejected: {validation_message} for image on page {page_num+1}, index {img_index}")
                    os.remove(temp_image_path)
        
        doc.close()
        logger.info(f"Precise extraction completed: Found {len(graphs_metadata)} stress-strain graphs")
        return graphs_metadata

    def _get_image_rect_simple(self, page, xref):
        """Simplified method to get image rectangle."""
        try:
            # Try the original method first
            text_dict = page.get_text("dict")
            for block in text_dict.get("blocks", []):
                if block.get("type", None) == 1:  # 1 = image block
                    if block.get("image") is not None:
                        if isinstance(block["image"], dict) and block["image"].get("xref") == xref:
                            return block["bbox"]
                    if block.get("xref") == xref:
                        return block["bbox"]
            
            # Fallback: return a dummy rectangle
            return [0, 0, 100, 100]
        except Exception as e:
            logger.warning(f"Error getting image rectangle: {e}")
            return [0, 0, 100, 100]

# Create a global instance
graph_extractor = GraphExtractor()
