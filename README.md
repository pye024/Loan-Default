# Credit Risk Modeling & Prediction System

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-ff6b6b?style=for-the-badge&logo=streamlit)](https://credit-risk-predicti0n.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/ML-XGBoost%20%7C%20Random%20Forest-green?style=flat-square)](https://xgboost.readthedocs.io/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ed?style=flat-square&logo=docker)](https://www.docker.com/)

> **An end-to-end machine learning solution for credit risk assessment**

## Project Summary

This project delivers a production-ready credit risk prediction system that helps financial institutions make data-driven lending decisions. Usingmachine learning techniques, the model achieves 86% AUC on test data and provides actionable business insights for risk management.

# [Try the Live Application](https://credit-risk-predicti0n.streamlit.app/)

## Analysis/ Walkthrough

For a detailed, scrollable walkthrough of this project, visit the Full Project Walkthrough

[![Read the Docs](https://img.shields.io/badge/Project%20Walkthrough-Click%20Here-blue?style=for-the-badge)](https://docs.google.com/document/d/e/2PACX-1vRIadN3BzcTTg_pHQHjFtgkxNjWrNu2Xc2ltv3vikDQXbCzpFtMES23eqDiXO9G8N1Stbqk1hHD1qGO/pub)


## Application Preview

<div align="center">

![Credit Risk App Interface](images/app.png)

*Interactive web interface for real-time credit risk assessment*

</div>

## Features

- **High-Performance ML Models**: XGBoost classifier achieving 86% AUC
- **Interactive Web Application**: User-friendly Streamlit interface
- **Production-Ready Deployment**: Docker containerization for scalability
- **Comprehensive Analysis**: Complete EDA and feature engineering pipeline


##  Project Structure

```
├──  Data Analysis & EDA     → notebooks/
├──  Trained Models          → models/
├──  Web Application         → app/app.py
├──  Docker Configuration    → Dockerfile
└──  Requirements            → requirements.txt
```

### Model Performance
| Model | AUC Score | Precision | Recall |
|-------|-----------|-----------|---------|
| **XGBoost** | **86%** | 92% | 87% |
| Random Forest | 84% | 87% | 85% |
| Logistic Regression| 74% | 70% | 75% |
### Key Predictive Features
- Debt-to-income ratio
- Credit utilization
- Previous Default history
- Employment length
- Loan amount relative to income

##  Technology Stack

**Core Technologies:**
- **Python**
- **Scikit-learn**
- **XGBoost**
- **Pandas & NumPy**
- **Streamlit**
- **Docker**

**Visualization & Deployment:**
- **Streamlit** - Interactive web application framework
- **Seaborn & Matplotlib** - Data visualization
- **Docker** - Containerization for deployment
- **Jupyter Notebook** - Development and analysis environment

##  Quick Start Guide

### Option 1: Local Development Setup

```bash
# Clone the repository
git clone https://github.com/pye024/Credit-Risk-Prediction.git
cd Credit-Risk-Prediction

# Install dependencies
pip install -r requirements.txt

# Launch the application
streamlit run app/app.py
```

**Access**: Open [http://localhost:8501](http://localhost:8501) in your browser

### Option 2: Docker Deployment

```bash
# Clone and navigate to project
git clone https://github.com/pye024/Credit-Risk-Prediction.git
cd Credit-Risk-Prediction

# Build Docker image
docker build -t credit-risk-app .

# Run container
docker run -p 8501:8501 credit-risk-app
```
**Access**: Navigate to [http://localhost:8501](http://localhost:8501)


