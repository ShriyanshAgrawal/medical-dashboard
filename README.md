# NeuroScan: Medical Imaging Analysis Platform

NeuroScan is a professional web application for analyzing medical images using machine learning models. The platform supports three types of analysis:

1. **Brain Tumor Detection**: Analyzes MRI scans to detect potential brain tumors
2. **ECG Analysis**: Processes electrocardiogram graphs to identify normal/abnormal patterns
3. **EEG Analysis**: Analyzes electroencephalogram signals to provide diagnostic insights

## Features

- Modern, responsive dashboard interface
- Real-time data visualization with interactive charts
- Easy image upload functionality
- Analysis history tracking
- Multi-model support for different types of medical imaging

## Installation

1. Clone the repository:
```
git clone https://github.com/yourusername/neuroscan.git
cd neuroscan
```

2. Install required dependencies:
```
pip install -r requirements.txt
```

3. Start the application:
```
python app.py
```

4. Open your browser and navigate to `http://localhost:5000`

## Usage

1. Navigate to the appropriate analysis section (Brain Tumor, ECG, or EEG)
2. Upload your medical image by clicking on the upload area
3. View the analysis results, which will be displayed as a processed image with annotations
4. Check the dashboard for a summary of all analyses performed

## Technologies Used

- Backend: Flask (Python)
- Frontend: HTML, CSS, JavaScript
- Data Visualization: Chart.js
- UI Components: Bootstrap

## Project Structure

- `app.py` - Main Flask application
- `templates/` - HTML templates
- `static/` - CSS, JavaScript, and other static files
- `models/` - Machine learning model files (not included in this repository)

## License

MIT License 