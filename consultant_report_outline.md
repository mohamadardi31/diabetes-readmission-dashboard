# Consultant Report Outline

# Reducing Diabetic Patient Readmissions: A Hospital Analytics Dashboard for 30-Day Risk Prediction

## 1. Executive Summary

Diabetic patient readmissions are a major concern for hospitals because they affect patient outcomes, bed capacity, care continuity, and operational cost. A 30-day readmission can indicate unresolved clinical issues, gaps in discharge planning, medication challenges, or limited follow-up after discharge.

This dashboard was developed as a healthcare analytics decision-support tool for hospital managers, clinical teams, and care coordination staff. It allows users to explore diabetic patient encounter data, identify segments with higher readmission rates, and estimate the probability that a patient encounter will result in readmission within 30 days.

The dashboard combines descriptive analytics and predictive modeling. Descriptive pages show readmission patterns by demographics, admission pathway, hospital utilization, medication indicators, and clinical profile. The model page evaluates a Random Forest classifier, and the risk calculator translates the trained model into a practical screening tool.

The dashboard is intended to support targeted discharge planning, follow-up prioritization, and operational discussion. It is not intended to replace clinical judgment or serve as a diagnostic system.

## 2. Data Sources and Variables

The project uses two files:

- `diabetic_data.csv`: the main encounter-level dataset for diabetic patients.
- `IDS_mapping.csv`: a supporting mapping file that translates numeric admission, discharge, and admission source IDs into readable labels.

The main outcome is the `readmitted` column. For prediction, the original outcome is transformed into a binary target called `readmitted_30`.

- `1`: the patient was readmitted within 30 days.
- `0`: the patient was not readmitted within 30 days.

Variable groups used in the dashboard include:

- Demographics: age, gender, race.
- Admission and utilization: admission type, discharge disposition, admission source, time in hospital, previous outpatient visits, previous emergency visits, and previous inpatient visits.
- Treatment indicators: number of medications, number of lab procedures, number of procedures, insulin use, medication change, and diabetes medication use.
- Clinical indicators: diagnosis count, A1C result, glucose serum result, and grouped ICD diagnosis categories.
- Readmission outcome: original readmission status and 30-day readmission target.

Before analysis, the dashboard cleans the data by replacing question marks with missing values, removing duplicates, creating the binary target, filling missing values, mapping ID columns when possible, and grouping diagnosis codes into broad clinical categories.

## 3. Dashboard Components

### Executive Overview

The executive page provides a high-level snapshot for decision-makers. It shows total hospital encounters, unique patients when available, the 30-day readmission rate, the overall readmission rate, average length of stay, average medication count, average lab procedures, and average number of diagnoses.

It also shows readmission distribution and readmission rates by age, gender, and race. This page is designed for quick briefing and presentation.

### Patient Demographics

This page focuses on age, gender, and race. It helps identify whether readmission patterns vary across patient groups. The page includes counts by age group, readmission rates by age, gender, and race, and a stacked view of readmission status by age group.

### Admission & Hospital Utilization

This page examines operational and utilization factors. It analyzes readmission rates by admission type, admission source, and discharge disposition. It also compares time in hospital and prior inpatient or emergency use across readmission outcomes.

This section is especially useful for hospital operations because previous utilization may indicate patients who need stronger discharge support.

### Treatment & Medication

This page explores medication and treatment intensity. It compares number of medications and lab procedures by readmission outcome and examines readmission rates by insulin use, medication changes, and diabetes medication use.

Medication variables should be interpreted carefully. They may reflect underlying patient complexity rather than directly causing readmission.

### Clinical Profile

This page summarizes clinical complexity. It includes diagnosis count, A1C result, max glucose serum result, broad diagnosis groups, and a correlation heatmap for available numeric variables.

The goal is to help decision-makers understand how clinical burden and broad diagnosis categories relate to readmission risk.

### Predictive Model

The predictive model page trains a Random Forest classifier to estimate whether an encounter is likely to result in readmission within 30 days. It reports accuracy, precision, recall, F1-score, and ROC-AUC.

The page also includes a confusion matrix, ROC curve, top feature importance chart, and predicted probability distribution. Recall is emphasized because hospitals often want to identify as many true high-risk patients as possible.

### Risk Calculator

The calculator allows users to enter patient and encounter characteristics and receive an estimated probability of 30-day readmission. It classifies risk into:

- Low Risk: less than 10%.
- Medium Risk: 10% to 20%.
- High Risk: greater than 20%.

Each category includes a suggested hospital action, such as routine discharge instructions, follow-up call, medication review, early follow-up appointment, care coordination, or discharge planning review.

### Methodology

The methodology page explains the dataset, target variable, variable groups, cleaning steps, modeling method, limitations, and recommendations. It also includes a preview of the filtered dataset.

## 4. Key Findings

The dashboard is designed to support findings such as:

- 30-day readmission varies by age group and patient profile.
- Patients with higher previous inpatient or emergency utilization may show elevated readmission risk.
- Treatment intensity, medication changes, and insulin use may indicate more complex cases.
- Number of diagnoses and diagnosis group can help identify clinical complexity.
- Model results can help prioritize discharge planning and care coordination.

These findings should be updated after reviewing the dashboard outputs for the final filtered or full dataset used in presentation.

## 5. Predictive Model

The model uses a Random Forest Classifier. Random Forest is useful for this project because it can handle nonlinear relationships, mixed feature types after preprocessing, and interactions among patient, admission, utilization, treatment, and clinical variables.

The model workflow includes:

- Selection of available demographic, admission, utilization, treatment, and clinical features.
- Median imputation for numeric variables.
- Most-frequent imputation and one-hot encoding for categorical variables.
- Stratified train/test split to preserve the readmission class balance.
- Class balancing to reduce the impact of the smaller 30-day readmission class.
- Evaluation using accuracy, precision, recall, F1-score, and ROC-AUC.

Recall is particularly important in a readmission context. A model with higher recall is better at finding patients who truly are at risk, which supports hospital efforts to intervene before a preventable readmission occurs.

## 6. Limitations

- The dataset is encounter-level, not purely patient-level. Some patients may appear more than once.
- Readmission risk is influenced by social determinants, behavioral factors, access to care, home support, and insurance context that may not be fully captured.
- The prediction model identifies associations and risk patterns, not causal relationships.
- Diagnosis codes are grouped into broad categories, which simplifies clinical detail.
- Mapping values may be incomplete or simplified depending on the structure of the mapping file.
- The dashboard should support decision-making, not replace clinical judgment.

## 7. Recommendations

Hospitals can use this dashboard to guide operational and clinical improvement discussions:

- Prioritize discharge planning for patients with high prior inpatient or emergency utilization.
- Use the risk calculator as a screening aid for care coordination.
- Focus follow-up resources on older patients, patients with many diagnoses, and patients with high predicted risk.
- Review medication changes and insulin use as potential markers of complex care needs.
- Combine dashboard outputs with local clinical judgment, social support information, and care access data.
- Track readmission performance over time after implementing discharge planning or follow-up interventions.
