# NeuroScan: Medical Imaging Analysis Platform

NeuroScan is a professional web application for analyzing medical images using machine learning models. The platform supports three types of analysis:

1. **Brain Tumor Detection**: Analyzes MRI scans to detect potential brain tumors  
2. **ECG Analysis** *(In Progress)*: Will process electrocardiogram graphs to identify normal/abnormal patterns  
3. **EEG Analysis** *(In Progress)*: Will analyze electroencephalogram signals to provide diagnostic insights for seizures

## Features

- Modern, responsive dashboard interface
- Real-time data visualization with interactive charts
- Easy image upload functionality
- Analysis history tracking
- Multi-model support for different types of medical imaging
- **Planned additions:** EEG and ECG signal analysis components (UI and backend integration underway)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/neuroscan.git
cd neuroscan
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Start the application:
```bash
python app.py
```

4. Open your browser and navigate to `http://localhost:5000`

## Usage

1. Navigate to the appropriate analysis section (Brain Tumor, ECG, or EEG)
2. Upload your medical image or signal data by clicking on the upload area
3. View the analysis results, which will be displayed as a processed image or chart with annotations
4. Check the dashboard for a summary of all analyses performed

> ⚠️ EEG and ECG functionality are under active development. Placeholder UI components and backend route files have been added to show upcoming progress.

## Technologies Used

- **Backend**: Flask (Python)
- **Frontend**: React.js *(in progress)*, HTML, CSS, JavaScript
- **Data Visualization**: Chart.js
- **UI Components**: Bootstrap

## Project Structure

```
medical-dashboard/
│
├── backend/
│   ├── app.py
│   ├── models/
│   │   ├── tumor_model.keras
│   │   ├── eeg_model_placeholder.txt
│   │   └── ecg_model_placeholder.txt
│   ├── routes/
│       ├── tumor.py
│       ├── eeg.py  # placeholder
│       └── ecg.py  # placeholder
│
├── frontend/
│   └── src/
│       └── components/
│           ├── TumorAnalysis.jsx
│           ├── EEGAnalysis.jsx  # placeholder
│           └── ECGAnalysis.jsx  # placeholder
│
└── README.md
```

## License

MIT License
