import os
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename
from flask_wtf import FlaskForm
from wtforms import MultipleFileField, SubmitField
from wtforms.validators import DataRequired
from flask_wtf.csrf import CSRFProtect
from process_pdfs import extract_graphs_from_pdf
import shutil

app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'  # Change for production
app.config['UPLOAD_FOLDER'] = 'static/files/'
app.config['GRAPH_FOLDER'] = os.path.join(app.config['UPLOAD_FOLDER'], 'graphs')
csrf = CSRFProtect(app)

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['GRAPH_FOLDER'], exist_ok=True)

class PDFUploadForm(FlaskForm):
    pdfs = MultipleFileField('Upload PDFs', validators=[DataRequired()])
    submit = SubmitField('Upload')

def get_gallery_data():
    """Scan static/files/ and static/files/graphs/ to build gallery data structure."""
    gallery = []
    pdfs = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) if f.lower().endswith('.pdf')]
    for pdf in pdfs:
        pdf_name = os.path.splitext(pdf)[0]
        pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], pdf)
        graph_dir = os.path.join(app.config['GRAPH_FOLDER'], pdf_name)
        graphs = []
        if os.path.isdir(graph_dir):
            for img in os.listdir(graph_dir):
                if img.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(graph_dir, img)
                    rel_img_path = img_path.replace('static/', '', 1)
                    # Try to get caption from a .txt file with the same name
                    caption = ''
                    caption_path = os.path.splitext(img_path)[0] + '.txt'
                    if os.path.exists(caption_path):
                        with open(caption_path, 'r') as f:
                            caption = f.read().strip()
                    graphs.append({'img': rel_img_path, 'caption': caption, 'filename': img})
        gallery.append({'pdf': pdf, 'pdf_name': pdf_name, 'graphs': graphs})
    return gallery

@app.route('/', methods=['GET', 'POST'])
def index():
    form = PDFUploadForm()
    if form.validate_on_submit():
        for file in form.pdfs.data:
            if file and file.filename.lower().endswith('.pdf'):
                filename = secure_filename(file.filename)
                save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(save_path)
                # Extract graphs and captions
                graph_output_dir = os.path.join(app.config['GRAPH_FOLDER'], os.path.splitext(filename)[0])
                valid_graphs = extract_graphs_from_pdf(save_path, graph_output_dir)
                # Save captions as .txt files next to images
                for img_path, caption in valid_graphs:
                    if caption:
                        with open(os.path.splitext(img_path)[0] + '.txt', 'w') as f:
                            f.write(caption)
                # If no valid graphs, remove empty folder
                if not valid_graphs and os.path.exists(graph_output_dir):
                    shutil.rmtree(graph_output_dir)
        flash('PDF(s) uploaded and processed!', 'success')
        return redirect(url_for('gallery'))
    return render_template('index.html', form=form)

@app.route('/gallery')
def gallery():
    gallery_data = get_gallery_data()
    return render_template('gallery.html', gallery=gallery_data)

@app.route('/delete_graphs', methods=['POST'])
def delete_graphs():
    data = request.get_json()
    print('Received data for deletion:', data)
    pdf_name = data.get('pdf_name')
    graphs = data.get('graphs', [])
    graph_dir = os.path.join(app.config['GRAPH_FOLDER'], pdf_name)
    print('Graph directory:', graph_dir)
    deleted = 0
    for graph in graphs:
        img_path = os.path.join(graph_dir, graph)
        txt_path = os.path.splitext(img_path)[0] + '.txt'
        print('Attempting to delete:', img_path)
        if os.path.exists(img_path):
            os.remove(img_path)
            print('Deleted image:', img_path)
            deleted += 1
        else:
            print('Image not found:', img_path)
        if os.path.exists(txt_path):
            os.remove(txt_path)
            print('Deleted caption:', txt_path)
        else:
            print('Caption not found:', txt_path)
    print(f'Total deleted: {deleted}')
    return jsonify({'success': True, 'deleted': deleted})

@app.route('/delete_pdf', methods=['POST'])
def delete_pdf():
    data = request.get_json()
    pdf_name = data.get('pdf_name')
    pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], pdf_name + '.pdf')
    graph_dir = os.path.join(app.config['GRAPH_FOLDER'], pdf_name)
    if os.path.exists(pdf_path):
        os.remove(pdf_path)
    if os.path.isdir(graph_dir):
        shutil.rmtree(graph_dir)
    return jsonify({'success': True})

if __name__ == '__main__':
    app.run(debug=True) 