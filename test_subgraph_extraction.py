#!/usr/bin/env python3
"""
Test script for subgraph extraction functionality.
"""

import os
import sys
from process_pdfs import split_graph_into_subgraphs_with_labels

def test_subgraph_extraction():
    """Test the subgraph extraction functionality."""
    
    # Test with a sample image if available
    test_image_path = "static/files/graphs/Research_paper/graph_3_1.jpeg"
    
    if not os.path.exists(test_image_path):
        print(f"Test image not found: {test_image_path}")
        print("Please upload a PDF first to generate test images.")
        return False
    
    print(f"Testing subgraph extraction with: {test_image_path}")
    
    # Create output directory
    output_dir = "static/files/graphs/Research_paper"
    os.makedirs(output_dir, exist_ok=True)
    
    # Test caption
    test_caption = "Stress-strain curves for different materials showing tensile behavior."
    
    try:
        # Run subgraph extraction
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
        
        return True
        
    except Exception as e:
        print(f"Error during extraction: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_subgraph_extraction()
    if success:
        print("\n✅ Subgraph extraction test completed successfully!")
    else:
        print("\n❌ Subgraph extraction test failed!")
        sys.exit(1) 