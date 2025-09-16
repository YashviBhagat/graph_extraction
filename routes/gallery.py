import os
from flask import Blueprint, render_template, current_app

# Create blueprint
gallery_bp = Blueprint('gallery', __name__)

def get_uploaded_pdfs():
    """Get list of all uploaded PDFs with basic info"""
    pdfs = []
    upload_folder = current_app.config['UPLOAD_FOLDER']
    
    if os.path.exists(upload_folder):
        for filename in os.listdir(upload_folder):
            if filename.lower().endswith('.pdf'):
                file_path = os.path.join(upload_folder, filename)
                file_size = os.path.getsize(file_path)
                file_size_mb = round(file_size / (1024 * 1024), 2)
                
                # Get file creation time
                import time
                creation_time = os.path.getctime(file_path)
                upload_date = time.strftime('%Y-%m-%d %H:%M', time.localtime(creation_time))
                
                pdfs.append({
                    'filename': filename,
                    'name': os.path.splitext(filename)[0],  # Remove .pdf extension
                    'size_mb': file_size_mb,
                    'upload_date': upload_date,
                    'file_path': f'files/{filename}'  # For download link
                })
    
    return sorted(pdfs, key=lambda x: x['name'], reverse=True)  # Sort by name

@gallery_bp.route('/gallery')
def gallery():
    """Display gallery of uploaded PDFs"""
    pdfs = get_uploaded_pdfs()
    return render_template('gallery.html', pdfs=pdfs)
