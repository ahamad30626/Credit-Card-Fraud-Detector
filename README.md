# Full-Stack Credit Card Fraud Detection System
## Powered by PyTorch, WGAN-GP, and Flask
This project implements a complete, production-ready, deep-learning-based credit card fraud detection system. It employs a Wasserstein Generative Adversarial Network with Gradient Penalty (WGAN-GP) to generate synthetic fraud samples to combat severe class imbalances common in financial datasets. A PyTorch Neural Network robustly classifies normal and fraudulent transactions with high precision and recall. 

This repository encapsulates modular, clean, and extensible machine learning code architectures paired perfectly with a modern, beautifully designed frontend using full-stack web integration.

### System Architecture
The application runs as a cohesive pipeline with the following structure:
```
CreditCardFraudSystem/
│
├── backend/
│   ├── models/            # PyTorch Deep Learning Models (.py)
│   │   ├── wgan.py        # WGAN-GP Generator & Critic Models with conditional label injections
│   │   └── classifier.py  # Feed-Forward Neural Network model architecture
│   │
│   ├── training/          # Training pipelines for ML workflows
│   │   ├── train_wgan.py       # Trains Generator only on Fraud distributions
│   │   └── train_classifier.py # Synthetic dataset balancing & NN metric evaluation
│   │
│   └── app.py             # Flask Web Server providing REST API & UI rendering
│
├── frontend/
│   ├── templates/         # HTML structure
│   │   └── index.html
│   └── static/            # Styling and client-side behavior JS
│       ├── style.css      # Dark modern fintech-themed UI
│       └── script.js      # Frontend transaction simulation scripts
│
├── requirements.txt       # Environment dependencies
└── README.md              # Project execution guide
```

### Setup Instructions

1. **Install Virtual Environment** (Optional but highly recommended)
   Create and activate a Python virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate       # Linux/Mac
   venv\Scripts\activate          # Windows
   ```

2. **Install Dependencies**
   Navigate to the repository and run:
   ```bash
   pip install -r requirements.txt
   ```

3. **Data Requirements**
   Ensure the `creditcard.csv` (Kaggle dataset) is located at `../Dataset/creditcard.csv` with respect to the backend directory, or modify the dataset paths in `train_wgan.py` and `train_classifier.py`.

### Execution Guide

#### Phase 1: Train the Generative Network
To generate valid, usable financial data synthetics, we first train a specialized Conditional WGAN.
```bash
cd backend/training
python train_wgan.py
```
This generates `fraud_generator.pth` inside `backend/`. 

#### Phase 2: Train the Classification Network
With the generator saved, we proceed to produce balanced datasets and train our binary classifier Neural Network. Evaluation metrics alongside precise performance graphs (Confusion matrix, Precision-Recall Curve) will automatically be saved into `backend/metrics/`.
```bash
python train_classifier.py
```
Once complete, `fraud_model.pth` and `scaler.pkl` will be initialized in `backend/`.

#### Phase 3: Launch Production Web Application
The full stack application will now ingest incoming transaction hashes and verify fraud validity interactively.
```bash
cd ../  # Navigate back to backend/
python app.py
```

### Access System UI
Open any modern web browser and traverse to:
`http://127.0.0.1:5000/`

You can manually provide up to 30 transaction features or easily load pre-configured Normal vs Fraud datasets through the provided demo testing buttons.

### Key Highlights
* **Pytorch only codebase**: Replaced Keras with a modular PyTorch architecture for finer control over gradient calculations and complex gradient penalty losses.
* **Deep Learning Pipeline**: End-to-end framework, balancing severe dataset deviations dynamically over epochs using GAN architectures rather than conventional interpolation (SMOTE).
* **Fintech UX/UI Styling**: Immersive dashboard simulating the environment of enterprise cybersecurity financial platforms.
# Credit-Card-Fraud-Detector
