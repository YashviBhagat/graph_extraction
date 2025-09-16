import os
from flask import Blueprint, render_template, request, redirect, url_for, flash, current_app
from werkzeug.utils import secure_filename

# Create blueprint
upload_bp = Blueprint('upload', __name__)

@upload_bp.route('/', methods=['GET', 'POST'])
def index():
    """Handle PDF uploads"""
    if request.method == 'POST':
        uploaded_count = 0
        
        # Check if files were uploaded
        if 'pdfs' not in request.files:
            flash('No files selected', 'warning')
            return redirect(url_for('upload.index'))
        
        files = request.files.getlist('pdfs')
        
        for file in files:
            if file and file.filename and file.filename.lower().endswith('.pdf'):
                filename = secure_filename(file.filename)
                if filename:  # Make sure filename is not empty
                    save_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
                    file.save(save_path)
                    uploaded_count += 1
                    print(f"✅ PDF uploaded: {filename}")
        
        if uploaded_count > 0:
            flash(f'{uploaded_count} PDF(s) uploaded successfully!', 'success')
            # Redirect to gallery after successful upload
            return redirect(url_for('gallery.gallery'))
        else:
            flash('No valid PDF files were uploaded.', 'warning')
            return redirect(url_for('upload.index'))
    
    return render_template('index.html')
