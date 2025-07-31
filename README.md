# Material Science Graph Extraction Website

A Flask-based web application for extracting and analyzing graphs from scientific PDFs, with advanced subgraph extraction capabilities.

## Features

### Core Functionality
- **PDF Upload**: Upload scientific PDFs containing graphs
- **Graph Extraction**: Automatically extract graphs from PDFs using OCR and image processing
- **Caption Detection**: Extract figure captions and descriptions
- **Alloy Classification**: Identify and classify material alloys mentioned in captions

### Advanced Subgraph Extraction
- **Multi-Graph Detection**: Automatically detect when a single image contains multiple subgraphs
- **Axis Label Recognition**: Extract X and Y axis labels using OCR
- **Intelligent Splitting**: Use computer vision algorithms to accurately separate individual subgraphs
- **Enhanced Captions**: Generate detailed captions for each extracted subgraph

## Installation

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Install Tesseract OCR** (required for text extraction):
   - **macOS**: `brew install tesseract`
   - **Ubuntu/Debian**: `sudo apt-get install tesseract-ocr`
   - **Windows**: Download from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)

3. **Run the Application**:
   ```bash
   python app.py
   ```

4. **Access the Website**: Open `http://localhost:5000` in your browser

## Usage

### Basic Workflow

1. **Upload PDFs**: Go to the home page and upload scientific PDFs containing graphs
2. **View Gallery**: Navigate to the gallery to see extracted graphs
3. **Extract Subgraphs**: Right-click on any graph and select "Extract Sub-Graphs"
4. **Review Results**: The extracted subgraphs will appear in the gallery with their axis labels

### Subgraph Extraction

The subgraph extraction feature uses advanced computer vision techniques:

- **Contour Detection**: Identifies potential graph regions using image contours
- **Line Detection**: Uses Hough transform to detect grid lines and axes
- **OCR Analysis**: Extracts text to identify axis labels and units
- **Region Expansion**: Ensures axis labels are included in extracted subgraphs

#### Supported Axis Labels

**X-axis**: strain, time, temperature, displacement, elongation, extension, deformation, cycle, frequency, wavelength, energy, distance

**Y-axis**: stress, load, force, modulus, strength, hardness, pressure, voltage, current, power, amplitude, intensity

#### Units Recognition

The system can recognize common units:
- **X-axis units**: %, mm, μm, nm, s, min, hr, °C, K
- **Y-axis units**: MPa, GPa, N, kN, psi, ksi, V, A, W

### Alloy Classification

The system automatically identifies and classifies material alloys:

- **High Entropy Alloys (HEA)**: 5+ principal elements
- **Medium Entropy Alloys (MEA)**: 3-4 principal elements  
- **Conventional Alloys**: Commercial/engineering alloys
- **Superalloys**: Nickel-based materials like Inconel
- **Stainless Steels**: Austenitic steels like 316L
- **Titanium Alloys**: Alpha-beta alloys like Ti-6Al-4V

## File Structure

```
material science/
├── app.py                          # Main Flask application
├── process_pdfs.py                 # PDF processing and graph extraction
├── requirements.txt                # Python dependencies
├── test_subgraph_extraction.py    # Test script for subgraph extraction
├── static/
│   ├── files/                     # Uploaded PDFs and extracted graphs
│   │   └── graphs/               # Extracted graph images
│   └── css/
│       └── styles.css            # Custom styles
└── templates/
    ├── index.html                 # Upload page
    └── gallery.html              # Graph gallery with extraction features
```

## API Endpoints

### Graph Management
- `GET /` - Upload page
- `GET /gallery` - Graph gallery
- `POST /delete_graphs` - Delete selected graphs
- `POST /delete_pdf` - Delete entire PDF and its graphs

### Subgraph Extraction
- `POST /extract_subgraphs` - Extract subgraphs from a graph image

**Request Body**:
```json
{
  "pdf_name": "Research_paper",
  "graph_filename": "graph_3_1.jpeg"
}
```

**Response**:
```json
{
  "success": true,
  "message": "Successfully extracted 3 subgraphs",
  "subgraphs": [
    {
      "filename": "graph_3_1_subgraph_1.jpeg",
      "path": "files/graphs/Research_paper/graph_3_1_subgraph_1.jpeg",
      "caption": "Subgraph 1 (X-axis: strain, Y-axis: stress). From: Original caption...",
      "x_label": "strain",
      "y_label": "stress"
    }
  ]
}
```

## Testing

Run the test script to verify subgraph extraction:

```bash
python test_subgraph_extraction.py
```

## Technical Details

### Graph Detection Algorithm

1. **Preprocessing**: Denoise and enhance image edges
2. **Thresholding**: Apply adaptive thresholding for varying lighting
3. **Morphology**: Clean up noise using morphological operations
4. **Contour Detection**: Find potential graph regions
5. **Line Detection**: Use Hough transform for grid lines (fallback)
6. **Region Filtering**: Filter by size and aspect ratio
7. **Label Extraction**: OCR analysis for axis labels

### Axis Label Detection

1. **Full Image OCR**: Extract all text from the image
2. **Region Analysis**: Focus on bottom (X-axis) and left (Y-axis) regions
3. **Pattern Matching**: Match against known axis label patterns
4. **Unit Recognition**: Identify common units and their associated labels

## Dependencies

- **Flask**: Web framework
- **PyMuPDF**: PDF processing
- **OpenCV**: Computer vision and image processing
- **Pillow**: Image manipulation
- **pytesseract**: OCR text extraction
- **NumPy**: Numerical computing

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License. 