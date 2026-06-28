# Reducing Diabetic Patient Readmissions

Hospital Analytics Dashboard for 30-Day Risk Prediction

## Project Description

This Streamlit project analyzes diabetic patient hospital encounters and helps hospital decision-makers explore patterns associated with 30-day readmission. The dashboard includes descriptive analytics, operational views, clinical profile charts, a predictive model, and an interactive risk calculator.

## Research Question

Which patient, admission, treatment, and clinical factors are associated with 30-day hospital readmission among diabetic patients?

## Dataset Files Required

- `diabetic_data.csv`
- `IDS_mapping.csv`

The dashboard requires `diabetic_data.csv`. The mapping file is optional, but recommended because it converts admission, discharge, and source IDs into readable labels.

## How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Dashboard Password

Default password:

```text
diabetes2026
```

To change it, edit the `PASSWORD` value near the top of `app.py`.

## Dashboard Sections

- Executive Overview
- Patient Demographics
- Admission & Hospital Utilization
- Treatment & Medication
- Clinical Profile
- Predictive Model
- Readmission Risk Calculator
- Data Dictionary & Methodology

## Model Explanation

The prediction task uses a binary target:

- `1` = patient was readmitted within 30 days
- `0` = patient was not readmitted within 30 days

The model is a Random Forest Classifier using demographic, admission, utilization, clinical, and treatment features. Categorical variables are one-hot encoded, numeric variables are median-imputed, and the train/test split is stratified. The dashboard reports accuracy, precision, recall, F1-score, and ROC-AUC.

Recall is highlighted because hospitals often want to identify as many truly high-risk patients as possible for follow-up and care coordination.

## Deployment Steps for Streamlit Community Cloud

1. Upload the project files to a GitHub repository.
2. Include `app.py`, `requirements.txt`, `README.md`, and `consultant_report_outline.md`.
3. Go to Streamlit Community Cloud.
4. Connect the GitHub repository.
5. Select `app.py` as the main file.
6. Deploy the app.
7. Upload the CSV files through the dashboard interface.

## Limitations

- This is encounter-level data, so some patients may appear more than once.
- Readmission is affected by social, behavioral, and healthcare access factors not fully captured in the dataset.
- The model identifies associations and risk patterns, not causal effects.
- Diagnosis categories are simplified from ICD-9 codes.
- The dashboard is for decision support and education, not clinical diagnosis.
