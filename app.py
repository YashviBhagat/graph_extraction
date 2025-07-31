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
    # Make sure to import these at the top: from process_pdfs import find_alloy_names, classify_alloy
    from process_pdfs import find_alloy_names, classify_alloy

    gallery = []
    pdfs = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) if f.lower().endswith('.pdf')]
    for pdf in pdfs:
        pdf_name = os.path.splitext(pdf)[0]
        graph_dir = os.path.join(app.config['GRAPH_FOLDER'], pdf_name)
        graphs = []
        captions_all = []  # <-- Move here, per-PDF!
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
                            if caption:
                                captions_all.append(caption)
                    # Try to detect axis labels from the image with enhanced detection
                    x_label, y_label = '', ''
                    try:
                        from process_pdfs import detect_axis_labels_enhanced
                        import cv2
                        img_array = cv2.imread(img_path)
                        if img_array is not None:
                            # Use enhanced axis label detection
                            axis_labels = detect_axis_labels_enhanced(img_array)
                            x_label = axis_labels.get('x', '')
                            y_label = axis_labels.get('y', '')
                            
                            # If no labels detected, try with a larger region
                            if not x_label and not y_label:
                                # Expand the image region and try again
                                height, width = img_array.shape[:2]
                                expanded_region = img_array[max(0, height//8):height, max(0, width//8):width]
                                axis_labels = detect_axis_labels_enhanced(expanded_region)
                                x_label = axis_labels.get('x', '')
                                y_label = axis_labels.get('y', '')
                    except Exception as e:
                        print(f"Error detecting axis labels for {img}: {e}")
                    
                    graphs.append({
                        'img': rel_img_path, 
                        'caption': caption, 
                        'filename': img,
                        'x_label': x_label,
                        'y_label': y_label
                    })
        # --- Extract alloy info from all captions for this paper ---
        alloy_set = set()
        alloy_infos = []
        for caption in captions_all:
            alloys = find_alloy_names(caption)
            for alloy in alloys:
                if alloy not in alloy_set:
                    alloy_set.add(alloy)
                    atype, category, desc = classify_alloy(alloy)
                    alloy_infos.append({
                        "name": alloy, "type": atype, "category": category, "description": desc
                    })
        gallery.append({'pdf': pdf, 'pdf_name': pdf_name, 'graphs': graphs, 'alloys': alloy_infos})
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

@app.route('/extract_subgraphs', methods=['POST'])
def extract_subgraphs():
    data = request.get_json()
    pdf_name = data.get('pdf_name')
    graph_filename = data.get('graph_filename')
    if not pdf_name or not graph_filename:
        return jsonify({'success': False, 'error': 'Missing parameters'}), 400

    graph_dir = os.path.join(app.config['GRAPH_FOLDER'], pdf_name)
    graph_path = os.path.join(graph_dir, graph_filename)
    if not os.path.exists(graph_path):
        return jsonify({'success': False, 'error': 'Graph image not found'}), 404

    # Optional: get caption if exists
    caption = ''
    caption_path = os.path.splitext(graph_path)[0] + '.txt'
    if os.path.exists(caption_path):
        with open(caption_path, 'r') as f:
            caption = f.read().strip()

    # Output directory for subgraphs (same as graph_dir)
    output_dir = graph_dir
    from process_pdfs import split_graph_into_subgraphs_with_labels
    subgraphs = split_graph_into_subgraphs_with_labels(graph_path, output_dir, caption)

    if subgraphs:
        # Return detailed information about extracted subgraphs
        subgraph_info = []
        for subgraph in subgraphs:
            rel_path = subgraph['path'].replace('static/', '', 1)
            subgraph_info.append({
                'filename': os.path.basename(subgraph['path']),
                'path': rel_path,
                'caption': subgraph['caption'],
                'x_label': subgraph['x_label'],
                'y_label': subgraph['y_label']
            })
        
        return jsonify({
            'success': True, 
            'message': f'Successfully extracted {len(subgraphs)} subgraphs',
            'subgraphs': subgraph_info
        })
    else:
        return jsonify({'success': False, 'error': 'No valid subgraphs found'})

if __name__ == '__main__':
    app.run(debug=True)