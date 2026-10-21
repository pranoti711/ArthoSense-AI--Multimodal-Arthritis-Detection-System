# Multimodal Arthritis Detection System

An academic AI/ML project exploring a multimodal approach to arthritis-related assessment by combining medical imaging, laboratory biomarkers, genomic markers, voice features, and wearable-sensor data in one application.

> **Research and educational prototype — not a medical device.** Outputs are experimental and must not be used to diagnose, treat, or make healthcare decisions. Clinical validation by qualified professionals is required before any real-world use.

## Overview

Arthritis assessment can involve multiple sources of information that are often reviewed separately. This project explores how machine-learning models for different data types could contribute complementary signals to a consolidated assessment workflow.

The project report describes a web-based interface with individual analysis modules, a full-assessment workflow, visual summaries, and report generation. The precise runtime entry point, model versions, fusion implementation, and reproducibility should be confirmed against the current source code before release.

## Diagnostic modalities

| Modality | Input described in the report | Methods described |
|---|---|---|
| Medical imaging | Joint X-ray images; the report also discusses spine MRI and synovial-fluid microscopy | CNN / transfer learning; model variants mentioned include VGG-16 and DenseNet121 |
| Laboratory analysis | Structured clinical laboratory values, such as ESR, CRP, rheumatoid factor, and related markers | Random Forest and other classical ML approaches |
| Genomic risk profiling | Genetic markers / SNP data | Marker encoding and classification models |
| Voice analysis | Audio recordings | MFCC and other acoustic features with a neural-network/classifier approach |
| Wearable / temporal analysis | Motion or wearable sensor time-series and related structured features | LSTM or dense neural-network approaches, depending on the module |
| Consolidated assessment | Outputs from available modules | A fusion or scoring workflow is described; verify the exact implementation in the current code |

The report includes multiple implementation descriptions across its sections. Treat the table above as a summary of the project scope, not a guarantee that every listed model is implemented and connected in the current checkout.

## Key capabilities described

- A dashboard-style interface for navigating assessment modules.
- Upload and preprocessing workflows for supported modality-specific files.
- Image, laboratory, genomic, voice, and wearable/temporal analysis modules.
- A consolidated assessment view intended to bring module outputs together.
- Visual result summaries and report-generation functionality.
- A modular Python codebase intended for experimentation and further development.

Availability of each feature depends on the code and model artifacts included in the repository.

## Technology stack

The project report references the following technologies:

- **Python** for application and model code
- **PyTorch / torchvision** and **TensorFlow / Keras** in different model descriptions
- **scikit-learn** for classical machine-learning workflows and evaluation
- **OpenCV**, **NumPy**, and **Pandas** for data processing
- **Librosa** for audio-feature extraction
- **Flask** and **Streamlit** in different descriptions of the web interface
- **Matplotlib / Seaborn** for plots and evaluation visuals
- **Git and GitHub** for version control

> **Before publishing:** the report mentions both Flask and Streamlit, and describes more than one deep-learning stack and image-model variant. Update this section to match the actual imports, dependency versions, and app entry point in the repository.

## Repository structure

Use this as a suggested structure if you are organizing the project for GitHub. Adjust it to match the files that actually exist:

```text
multimodal-arthritis-detection-system/
├── app/                    # Web application and UI
├── modules/                # Modality-specific inference / preprocessing
│   ├── vision/
│   ├── laboratory/
│   ├── genomics/
│   ├── voice/
│   └── wearable/
├── data/
│   ├── raw/                # Local only; normally not committed
│   └── processed/          # Local only; normally not committed
├── models/                 # Trained model artifacts; see note below
├── notebooks/              # Experiments and analysis
├── reports/                # Documentation and figures safe to share
├── tests/
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

Do not create empty folders solely to match this example. Keep the layout aligned with the actual codebase.

## Getting started

### 1. Prerequisites

- Python **3.8 or newer** is stated in the project report; use the version supported by your pinned dependencies.
- Git
- Enough disk space for dependencies and any datasets/model artifacts you need.
- A GPU may help with training deep-learning models; CPU inference may be possible depending on the model and configuration.

### 2. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd <YOUR_REPOSITORY_DIRECTORY>
```

Replace the placeholders with your repository URL and directory name.

### 3. Create and activate a virtual environment

**Windows (PowerShell)**

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

**Linux / macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 4. Install dependencies

If the repository contains a verified dependency file:

```bash
pip install -r requirements.txt
```

If `requirements.txt` has not yet been created, generate it from the project's actual imports and tested environment first. Avoid installing an unpinned mixture of TensorFlow, PyTorch, and other packages without checking compatibility.

### 5. Configure datasets and model artifacts

Datasets and trained weights may be excluded from Git because of size, licensing, privacy, or redistribution restrictions. Obtain each dataset from its original source and follow its terms of use.

If inference requires trained weights (`.pt`, `.pth`, `.h5`, `.keras`, `.pkl`, or similar files), place them in the paths expected by the code. Document the expected filenames and provide a legitimate download/source separately where redistribution is permitted. **Never load pickle files from untrusted sources.**

### 6. Run the application

The project report refers to both Flask and Streamlit, so the correct launch command must be confirmed from the actual entry-point file.

Examples only — use the one that matches the code:

```bash
streamlit run app.py
```

or

```bash
python app.py
```

For Flask, follow the application's actual entry point and configuration. Do not assume either command will work until you verify the repository structure and installed dependencies.

## Model evaluation

The project report describes module-level experiments and reports different metrics in different sections. These figures should be treated as **reported results, not independently reproduced benchmarks**, until the datasets, split strategy, evaluation scripts, and trained checkpoints are available and the results have been rerun.

For a reproducible evaluation, document at least:

- Dataset source, version, licence, and class distribution
- Patient-level or other leakage-safe train/validation/test split strategy
- Preprocessing and feature-extraction settings
- Model architecture and checkpoint provenance
- Accuracy, precision, recall, F1-score, confusion matrix, and ROC-AUC where appropriate
- Random seeds, software versions, and hardware
- Per-modality and combined-model results, including missing-modality behavior

Do not compare scores across datasets or splits as if they were directly comparable.

## Data privacy and security

- Do not commit patient-identifiable, private, or sensitive health data.
- Do not commit credentials, API keys, access tokens, private paths, or local environment files.
- Validate uploaded file type, size, and content before processing.
- Use synthetic or appropriately de-identified sample data for demonstrations.
- Review dataset licences and consent restrictions before redistributing data.
- Do not expose a development server publicly without appropriate authentication, secure deployment configuration, and privacy review.

## Limitations

- This is an academic research prototype, not a clinically validated diagnostic system.
- Model outputs may be inaccurate, biased, or unreliable on data unlike the training data.
- Voice, genomic, laboratory, wearable, and imaging signals are not independently conclusive evidence of arthritis.
- The exact model and interface implementation should be verified against the current source code because the report contains multiple descriptions of the stack and architecture.
- Risk scores or confidence values must not be interpreted as calibrated clinical probabilities unless calibration has been evaluated.

## Responsible use

This project is intended for learning, research, and software demonstration only. It does not replace a clinician, radiologist, rheumatologist, laboratory assessment, or professional medical advice. Do not use it to make treatment decisions or delay seeking medical care.

## Author

Pranoti Ashok Munjankar

