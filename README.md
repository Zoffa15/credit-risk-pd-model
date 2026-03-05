# Credit Risk Probability of Default Model

This project implements a Probability of Default (PD) model for unsecured personal loans using machine learning and traditional statistical methods.

The objective is to estimate the likelihood that a borrower will default, supporting lending decisions, pricing, and risk management.

---

## Project Overview

The project demonstrates a full credit risk modelling workflow:

- Exploratory Data Analysis
- Feature selection
- Data leakage mitigation
- Model development
- Model validation
- Risk segmentation
- Stability analysis

Two modelling approaches were implemented:

1. LightGBM Gradient Boosting
2. Logistic Regression (benchmark model)

---

## Dataset

The dataset contains anonymized loan application and credit information.

- 10,000 observations
- 196 features
- Target variable: `TARGET`

TARGET definition:
- 1 = Default
- 0 = Non-default

Due to confidentiality, the raw dataset is not included in this repository.

---

## Methodology

### Feature Engineering
Potential post-event variables were removed to avoid data leakage.

### Validation Strategy

Robust validation techniques were used:

- 5-Fold Stratified Cross Validation
- Time-based validation
- Decile analysis
- Population Stability Index (PSI)

### Evaluation Metrics

- ROC AUC
- Gini coefficient
- Default rate by decile
- Population Stability Index

---

## Results

| Model | CV AUC | Gini |
|------|------|------|
| Logistic Regression | ~0.85–0.88 | ~0.70–0.76 |
| LightGBM | ~0.90–0.91 | ~0.80–0.82 |

The LightGBM model achieved higher predictive performance while Logistic Regression provided strong interpretability.

---

## Outputs

The model generates predicted Probability of Default (PD) for each loan application.

Example output:



---

## Technologies Used

- Python
- Pandas
- Scikit-learn
- LightGBM
- Matplotlib
- Jupyter Notebook

---

## Project Structure

credit-risk-pd-model/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   └── data_description.txt
│
├── notebooks/
│   └── pd_modeling.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── train_models.py
│   ├── validation.py
│   └── evaluation.py
│
├── outputs/
│   ├── pd_predictions.csv
│   └── model_metrics.txt
│
├── images/
│   ├── roc_curve.png
│   ├── decile_analysis.png
│   └── feature_importance.png
│
└── docs/
    └── executive_summary.md

---

## Future Improvements

Possible extensions include:

- Probability calibration
- SHAP feature importance
- Macroeconomic stress testing
- Reject inference
- Model monitoring framework

---
## License

This project is licensed under the MIT License.

You are free to:

- Use
- Modify
- Distribute
- Apply commercially

Attribution is required.

---
## Author

Zofia Olszewska

## Contact

For questions, collaboration, or discussion: sofie.olszewska@gmail.com
