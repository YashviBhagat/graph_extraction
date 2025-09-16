from flask import Blueprint, request, jsonify, current_app
import os

# Create blueprint
api_bp = Blueprint('api', __name__)

@api_bp.route('/delete_pdf', methods=['POST'])
def delete_pdf():
    """Delete a PDF file and its associated graphs"""
    try:
        data = request.get_json()
        filename = data.get('filename')
        
        if not filename:
            return jsonify({'success': False, 'error': 'No filename provided'}), 400
        
        # Security check - ensure filename is safe
        if '..' in filename or '/' in filename or '\\' in filename:
            return jsonify({'success': False, 'error': 'Invalid filename'}), 400
        
        # Paths
        pdf_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        graph_folder = os.path.join(current_app.config['GRAPH_FOLDER'], filename.replace('.pdf', ''))
        
        # Delete PDF file
        if os.path.exists(pdf_path):
            os.remove(pdf_path)
            print(f"✅ PDF deleted: {filename}")
        
        # Delete associated graphs folder
        if os.path.exists(graph_folder):
            import shutil
            shutil.rmtree(graph_folder)
            print(f"✅ Graph folder deleted: {graph_folder}")
        
        return jsonify({'success': True, 'message': 'PDF and graphs deleted successfully'}), 200
        
    except Exception as e:
        print(f"Error deleting PDF: {e}")
        return jsonify({'success': False, 'error': 'Failed to delete PDF'}), 500
