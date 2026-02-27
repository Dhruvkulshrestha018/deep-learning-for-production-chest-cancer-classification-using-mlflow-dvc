Chest Cancer Classification using MLflow & DVC

https://img.shields.io/badge/python-3.8-blue.svg
![MLflow](https://img.shields.io/badge/MLflow- Tracking-0194E2.svg)
https://img.shields.io/badge/DVC-Data%2520Version%2520Control-945DD6.svg
https://img.shields.io/badge/Docker-Containerized-2496ED.svg
https://img.shields.io/badge/AWS-EC2%2520Deployed-FF9900.svg

📋 Project Overview

End-to-end deep learning project for classifying chest CT scans to detect adenocarcinoma cancer. The project implements MLOps best practices including experiment tracking, data version control, pipeline automation, and cloud deployment.

🎯 Key Features

CNN Model: VGG16-based transfer learning for image classification
Experiment Tracking: MLflow integrated with DagsHub
Pipeline Automation: DVC for data and model versioning
Containerization: Docker for consistent deployment
Cloud Deployment: AWS EC2 with GitHub Actions CI/CD

🏗️ Architecture

┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Data Ingestion │────▶│Prepare BaseModel│────▶│    Training     │
└─────────────────┘     └─────────────────┘     └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 ▼
                        ┌─────────────────┐
                        │   Evaluation    │
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  MLflow Tracking│
                        └─────────────────┘


Project Structure

.
├── app.py
├── artifacts
│   ├── data_ingestion
│   │   ├── __MACOSX
│   │   ├── Chest-CT-Scan-data
│   │   └── data.zip
│   ├── prepare_base_model
│   │   ├── base_model_updated.h5
│   │   └── base_model.h5
│   └── training
│       └── model.h5
├── config
│   └── config.yaml
├── dockerfile
├── dvc.lock
├── dvc.yaml
├── inputImage.jpg
├── LICENSE
├── logs
│   └── running_logs.log
├── main.py
├── mlruns
│   ├── 0
│   │   ├── 1594a35983e34edd948e458df920a9c2
│   │   └── meta.yaml
│   └── models
├── params.yaml
├── requirements.txt
├── research
│   ├── 01_data_ingestion.ipynb
│   ├── 02_prepare_base_model.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_model_evaluation.ipynb
│   ├── Chest-CT-Scan-data.zip
│   ├── logs
│   │   └── running_logs.log
│   └── trials.ipynb
├── scores.json
├── setup.py
├── src
│   ├── cnnClassifier
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   ├── components
│   │   ├── config
│   │   ├── constants
│   │   ├── entity
│   │   ├── pipeline
│   │   └── utils
│   └── cnnClassifier.egg-info
│       ├── dependency_links.txt
│       ├── PKG-INFO
│       ├── SOURCES.txt
│       └── top_level.txt
├── template.py
└── templates
    └── index.html

Installation & Setup

Prerequisites

Python 3.8+
Docker (optional)
AWS Account (for deployment)
Local Setup

Clone the repository

git clone https://github.com/Dhruvkulshrestha018/deep-learning-for-production-chest-cancer-classification-using-mlflow-dvc.git
cd deep-learning-for-production-chest-cancer-classification-using-mlflow-dvc

Install dependencies
pip install -r requirements.txt

Running the Pipeline

Using DVC

bash
# Reproduce entire pipeline
dvc repro

# Run specific stage
dvc repro training

# Visualize pipeline
dvc dag

Training Manually

bash
# Run each stage sequentially
python src/cnnClassifier/pipeline/stage_01_data_ingestion.py
python src/cnnClassifier/pipeline/stage_02_prepare_base_model.py
python src/cnnClassifier/pipeline/stage_03_model_trainer.py
python src/cnnClassifier/pipeline/stage_04_model_evaluation.py
🌐 Web Application

Local Testing

bash
# Run Flask app
python app.py

# Access at http://localhost:8080
Docker Deployment

bash
# Build image
docker build -t chest-cancer-classifier .

# Run container
docker run -p 8080:8080 chest-cancer-classifier
📊 MLflow Tracking

Experiments are tracked on DagsHub

bash
# View local MLflow UI
mlflow ui

# Track experiments
# Automatically logged during evaluation stage
☁️ AWS Deployment

EC2 Setup

Launch EC2 instance (Ubuntu, t2.large or similar)
Install Docker
Configure GitHub Actions runner
Deploy using CI/CD pipeline
GitHub Actions CI/CD

The project includes automated deployment:

Builds Docker image
Pushes to AWS ECR
Deploys to EC2 instance

Sample Prediction

python
from src.cnnClassifier.pipeline.prediction import PredictionPipeline

# Initialize predictor
predictor = PredictionPipeline("path/to/image.jpg")

# Get prediction
result = predictor.predict()
# Output: [{"image": "Normal"}] or [{"image": "Adenocarcinoma Cancer"}]
🔧 Configuration

Key parameters in config/config.yaml:

yaml
IMAGE_SIZE: [224, 224, 3]
BATCH_SIZE: 16
EPOCHS: 10
LEARNING_RATE: 0.001
AUGMENTATION: True
📚 Dependencies

text
tensorflow==2.13.0
mlflow==2.5.0
dvc==3.0.0
flask==2.3.0
flask-cors==4.0.0
python-dotenv==1.0.0
numpy==1.24.0
pillow==10.0.0
🎯 Key Learnings

✅ End-to-end ML project implementation
✅ MLOps best practices with DVC & MLflow
✅ Containerization with Docker
✅ Cloud deployment on AWS EC2
✅ CI/CD automation with GitHub Actions
✅ Experiment tracking and version control
🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

📝 License

This project is licensed under the MIT License.

👨‍💻 Author

Dhruv Kulshrestha

GitHub: @Dhruvkulshrestha018
Project Link: Repository
🙏 Acknowledgments

Dataset: Chest CT-Scan images dataset
VGG16 architecture for transfer learning
MLflow & DVC for MLOps
AWS for cloud infrastructure
📧 Contact

For questions or feedback, please open an issue in the repository.

⭐ Star this repository if you found it helpful!
