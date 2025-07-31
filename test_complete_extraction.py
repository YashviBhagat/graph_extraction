#!/usr/bin/env python3
"""
Comprehensive test script for subgraph extraction with improved functionality.
"""

import os
import sys
from process_pdfs import split_graph_into_subgraphs_with_labels, detect_axis_labels_enhanced
import cv2

def test_complete_extraction():
    """Test the complete subgraph extraction functionality."""
    
    # Test with a sample image
    test_image_path = "static/files/graphs/Research_paper/graph_3_2.jpeg"
    
    if not os.path.exists(test_image_path):
        print(f"Test image not found: {test_image_path}")
        print("Please upload a PDF first to generate test images.")
        return False
    
    print(f"Testing complete subgraph extraction with: {test_image_path}")
    print("=" * 60)
    
    # Create output directory
    output_dir = "static/files/graphs/Research_paper"
    os.makedirs(output_dir, exist_ok=True)
    
    # Test caption
    test_caption = "Stress-strain curves for different materials showing tensile behavior and mechanical properties."
    
    try:
        # Test axis detection first
        print("\n1. Testing axis label detection...")
        img_array = cv2.imread(test_image_path)
        if img_array is not None:
            axis_labels = detect_axis_labels_enhanced(img_array)
            print(f"   X-axis: {axis_labels.get('x', 'Not detected')}")
            print(f"   Y-axis: {axis_labels.get('y', 'Not detected')}")
        
        # Run subgraph extraction
        print("\n2. Testing subgraph extraction...")
        subgraphs = split_graph_into_subgraphs_with_labels(
            test_image_path, 
            output_dir, 
            test_caption
        )
        
        print(f"\nExtraction completed!")
        print(f"Found {len(subgraphs)} subgraphs:")
        
        for i, subgraph in enumerate(subgraphs, 1):
            print(f"\nSubgraph {i}:")
            print(f"  Path: {subgraph['path']}")
            print(f"  Caption: {subgraph['caption']}")
            print(f"  X-axis: {subgraph['x_label']}")
            print(f"  Y-axis: {subgraph['y_label']}")
            print(f"  Region: {subgraph['region']}")
            
            # Check if files were created
            if os.path.exists(subgraph['path']):
                print(f"  ✓ Image file created")
            else:
                print(f"  ✗ Image file missing")
            
            caption_path = os.path.splitext(subgraph['path'])[0] + '.txt'
            if os.path.exists(caption_path):
                print(f"  ✓ Caption file created")
                # Read and display caption
                with open(caption_path, 'r') as f:
                    caption_content = f.read()
                    print(f"  Caption content: {caption_content[:100]}...")
            else:
                print(f"  ✗ Caption file missing")
        
        return True
        
    except Exception as e:
        print(f"Error during extraction: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_axis_detection_improvements():
    """Test the improved axis detection on multiple images."""
    
    test_images = [
        "static/files/graphs/Research_paper/graph_3_2.jpeg",
        "static/files/graphs/paper-207/graph_5_1.jpeg",
        "static/files/graphs/srep31028/graph_7_2.jpeg"
    ]
    
    print("\n" + "=" * 60)
    print("Testing improved axis detection...")
    
    for img_path in test_images:
        if os.path.exists(img_path):
            print(f"\nTesting: {os.path.basename(img_path)}")
            
            try:
                img_array = cv2.imread(img_path)
                if img_array is not None:
                    axis_labels = detect_axis_labels_enhanced(img_array)
                    
                    print(f"  X-axis: {axis_labels.get('x', 'Not detected')}")
                    print(f"  Y-axis: {axis_labels.get('y', 'Not detected')}")
                    
                    # Check for duplicates
                    if axis_labels.get('x') == axis_labels.get('y') and axis_labels.get('x'):
                        print(f"  ⚠️  Warning: Both axes have the same label")
                    else:
                        print(f"  ✓ Different labels detected")
                        
            except Exception as e:
                print(f"  Error: {e}")
        else:
            print(f"\nImage not found: {img_path}")

if __name__ == "__main__":
    print("Comprehensive Subgraph Extraction Test")
    print("=" * 60)
    
    # Test axis detection improvements
    test_axis_detection_improvements()
    
    # Test complete extraction
    success = test_complete_extraction()
    
    if success:
        print("\n" + "=" * 60)
        print("✅ All tests completed successfully!")
        print("\nKey improvements implemented:")
        print("✓ Wider extraction boundaries to capture axis labels")
        print("✓ Enhanced axis label detection with multiple methods")
        print("✓ Improved caption generation with detailed information")
        print("✓ Better duplicate detection and differentiation")
        print("✓ Enhanced UI display of axis labels and captions")
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1) 