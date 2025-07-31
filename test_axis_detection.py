#!/usr/bin/env python3
"""
Test script for improved axis label detection.
"""

import os
import sys
from process_pdfs import detect_axis_labels_enhanced
import cv2

def test_axis_detection():
    """Test the enhanced axis label detection on existing images."""
    
    # Test with existing images
    test_images = [
        "static/files/graphs/Research_paper/graph_3_1.jpeg",
        "static/files/graphs/Research_paper/graph_3_2.jpeg",
        "static/files/graphs/paper-207/graph_5_1.jpeg",
        "static/files/graphs/srep31028/graph_7_2.jpeg"
    ]
    
    print("Testing enhanced axis label detection...")
    print("=" * 50)
    
    for img_path in test_images:
        if os.path.exists(img_path):
            print(f"\nTesting: {img_path}")
            
            try:
                # Read the image
                img_array = cv2.imread(img_path)
                if img_array is not None:
                    # Test axis label detection
                    axis_labels = detect_axis_labels_enhanced(img_array)
                    
                    print(f"  X-axis: {axis_labels.get('x', 'Not detected')}")
                    print(f"  Y-axis: {axis_labels.get('y', 'Not detected')}")
                    
                    # Test with expanded region
                    height, width = img_array.shape[:2]
                    expanded_region = img_array[max(0, height//8):height, max(0, width//8):width]
                    expanded_labels = detect_axis_labels_enhanced(expanded_region)
                    
                    if expanded_labels.get('x') or expanded_labels.get('y'):
                        print(f"  Expanded region - X-axis: {expanded_labels.get('x', 'Not detected')}")
                        print(f"  Expanded region - Y-axis: {expanded_labels.get('y', 'Not detected')}")
                else:
                    print(f"  Error: Could not read image")
                    
            except Exception as e:
                print(f"  Error: {e}")
        else:
            print(f"\nImage not found: {img_path}")
    
    print("\n" + "=" * 50)
    print("Axis label detection test completed!")

if __name__ == "__main__":
    test_axis_detection() 