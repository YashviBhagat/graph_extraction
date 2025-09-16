from flask import Blueprint, render_template, request, current_app, send_file
import os
import json
from services.graph_extractor import GraphExtractor

# Create blueprint
graphs_bp = Blueprint('graphs', __name__)

# Create graph extractor instance
graph_extractor = GraphExtractor()

@graphs_bp.route('/graphs/<filename>')
def view_graphs(filename):
    """View extracted stress-strain graphs for a specific PDF - Full page view"""
    try:
        # Security check
        if '..' in filename or '/' in filename or '\\' in filename:
            return "Invalid filename", 400
        
        pdf_name_without_ext = filename.replace('.pdf', '')
        graph_folder = os.path.join(current_app.config['GRAPH_FOLDER'], pdf_name_without_ext)
        
        # Ensure graph folder exists
        os.makedirs(graph_folder, exist_ok=True)

        # Check if graphs already exist and load metadata
        graphs_metadata_path = os.path.join(graph_folder, 'graphs_metadata.json')
        graphs = []

        if os.path.exists(graphs_metadata_path):
            with open(graphs_metadata_path, 'r') as f:
                graphs = json.load(f)
            print(f"Loaded {len(graphs)} existing graphs from {graph_folder}")
        
        # If no graphs found or metadata is empty, try to extract them
        if not graphs:
            print(f"No existing graphs found for {filename}, attempting extraction...")
            pdf_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
            extracted_graphs = graph_extractor.extract_graphs_from_pdf(pdf_path, graph_folder)
            
            # Save metadata
            with open(graphs_metadata_path, 'w') as f:
                json.dump(extracted_graphs, f, indent=4)
            graphs = extracted_graphs
            print(f"Extracted {len(graphs)} valid stress-strain graphs for {filename}")
        
        # Use the full-page template
        return render_template('graphs_fullpage.html', 
                             pdf_filename=filename, 
                             graphs=graphs)
    
    except Exception as e:
        print(f"Error viewing graphs for {filename}: {e}")
        return f"Error: {str(e)}", 500

@graphs_bp.route('/graph_image/<filename>')
def get_graph_image(filename):
    """Serve graph images"""
    try:
        # Security check
        if '..' in filename or '/' in filename or '\\' in filename:
            return "Invalid filename", 400
        
        # Find the graph file in any subfolder
        graph_folder_base = current_app.config['GRAPH_FOLDER']
        for root, dirs, files in os.walk(graph_folder_base):
            if filename in files:
                file_path = os.path.join(root, filename)
                return send_file(file_path)
        
        return "Graph not found", 404
    
    except Exception as e:
        print(f"Error serving graph image {filename}: {e}")
        return f"Error: {str(e)}", 500
