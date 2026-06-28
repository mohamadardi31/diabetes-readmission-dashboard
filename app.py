import csv
import io
import re

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


PASSWORD = "diabetes2026"
RANDOM_STATE = 42

AGE_ORDER = [
    "[0-10)",
    "[10-20)",
    "[20-30)",
    "[30-40)",
    "[40-50)",
    "[50-60)",
    "[60-70)",
    "[70-80)",
    "[80-90)",
    "[90-100)",
]

AGE_MIDPOINTS = {
    "[0-10)": 5,
    "[10-20)": 15,
    "[20-30)": 25,
    "[30-40)": 35,
    "[40-50)": 45,
    "[50-60)": 55,
    "[60-70)": 65,
    "[70-80)": 75,
    "[80-90)": 85,
    "[90-100)": 95,
}

NUMERIC_COLUMNS = [
    "encounter_id",
    "patient_nbr",
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
]


st.set_page_config(
    page_title="Diabetic Readmission Analytics",
    page_icon=":hospital:",
    layout="wide",
)


def password_gate():
    if st.session_state.get("authenticated"):
        return True

    st.title("Reducing Diabetic Patient Readmissions")
    st.subheader("Hospital Analytics Dashboard for 30-Day Risk Prediction")
    entered = st.text_input("Dashboard password", type="password")

    if st.button("Enter dashboard", use_container_width=True):
        if entered == PASSWORD:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Incorrect password. Please try again.")

    st.caption("Default password: diabetes2026")
    return False


@st.cache_data(show_spinner=False)
def load_data(file_bytes):
    df = pd.read_csv(io.BytesIO(file_bytes), low_memory=False)
    df.columns = df.columns.astype(str).str.strip()
    return df


def load_mapping_file(uploaded_file):
    if uploaded_file is None:
        return None
    return uploaded_file.getvalue()


def parse_id_mappings(file_bytes):
    mappings = {
        "admission_type_id": {},
        "discharge_disposition_id": {},
        "admission_source_id": {},
    }

    if not file_bytes:
        return mappings

    text = file_bytes.decode("utf-8", errors="ignore")
    rows = list(csv.reader(io.StringIO(text)))
    current_section = None

    for row in rows:
        cleaned = [cell.strip() for cell in row if cell is not None]
        if not cleaned or all(cell == "" for cell in cleaned):
            continue

        first = cleaned[0]
        if first in mappings:
            current_section = first
            continue

        if current_section and first.lower() not in {"id", current_section.lower()}:
            if len(cleaned) >= 2 and first != "":
                try:
                    key = int(float(first))
                    value = cleaned[1].strip() or "Unknown"
                    if value.upper() == "NULL":
                        value = "Unknown / Not Available"
                    mappings[current_section][key] = value
                except ValueError:
                    continue

    return mappings


def load_id_mappings(uploaded_file):
    try:
        return parse_id_mappings(load_mapping_file(uploaded_file))
    except Exception:
        st.warning("Mapping file could not be parsed. ID fields will remain numeric.")
        return {
            "admission_type_id": {},
            "discharge_disposition_id": {},
            "admission_source_id": {},
        }


def categorize_diagnosis(code):
    if pd.isna(code):
        return "Other"

    value = str(code).strip()
    if value in {"", "Unknown", "nan", "None"}:
        return "Other"
    if value.upper().startswith(("V", "E")):
        return "Other"

    match = re.search(r"\d+(\.\d+)?", value)
    if not match:
        return "Other"

    number = float(match.group())
    whole_number = int(number)

    if whole_number == 250:
        return "Diabetes"
    if 390 <= whole_number <= 459 or whole_number == 785:
        return "Circulatory"
    if 460 <= whole_number <= 519 or whole_number == 786:
        return "Respiratory"
    if 520 <= whole_number <= 579 or whole_number == 787:
        return "Digestive"
    if 800 <= whole_number <= 999:
        return "Injury"
    if 710 <= whole_number <= 739:
        return "Musculoskeletal"
    if 580 <= whole_number <= 629 or whole_number == 788:
        return "Genitourinary"
    if 140 <= whole_number <= 239:
        return "Neoplasms"
    return "Other"


@st.cache_data(show_spinner=False)
def clean_data(df):
    data = df.copy()
    data.columns = data.columns.astype(str).str.strip()
    data = data.replace("?", np.nan)
    data = data.drop_duplicates()

    if "readmitted" not in data.columns:
        raise ValueError("The uploaded file must include a readmitted column.")

    data = data.dropna(subset=["readmitted"])
    data["readmitted"] = data["readmitted"].astype(str).str.strip()
    data["readmitted_30"] = np.where(data["readmitted"].eq("<30"), 1, 0)
    data["readmission_label"] = data["readmitted"].map(
        {
            "<30": "Readmitted <30 Days",
            ">30": "Readmitted >30 Days",
            "NO": "Not Readmitted",
        }
    ).fillna("Unknown")

    for col in NUMERIC_COLUMNS:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    if "age" in data.columns:
        data["age"] = pd.Categorical(data["age"], categories=AGE_ORDER, ordered=True)
        data["age_midpoint"] = data["age"].map(AGE_MIDPOINTS).astype(float)

    for diag_col in ["diag_1", "diag_2", "diag_3"]:
        if diag_col in data.columns:
            data[f"{diag_col}_group"] = data[diag_col].apply(categorize_diagnosis)

    numeric_cols = data.select_dtypes(include=["number"]).columns
    for col in numeric_cols:
        if col not in {"readmitted_30"}:
            median = data[col].median()
            if pd.isna(median):
                median = 0
            data[col] = data[col].fillna(median)

    categorical_cols = data.select_dtypes(exclude=["number"]).columns
    for col in categorical_cols:
        data[col] = data[col].astype("object").where(data[col].notna(), "Unknown")

    return data


def apply_mappings(df, mappings):
    data = df.copy()
    mapping_specs = {
        "admission_type_id": "admission_type_label",
        "discharge_disposition_id": "discharge_disposition_label",
        "admission_source_id": "admission_source_label",
    }

    for id_col, label_col in mapping_specs.items():
        if id_col in data.columns:
            mapping = mappings.get(id_col, {}) if mappings else {}
            mapped = data[id_col].astype(int).map(mapping) if mapping else pd.Series(index=data.index, dtype="object")
            data[label_col] = mapped.fillna(data[id_col].astype(int).astype(str))

    return data


def page_title(title, subtitle):
    st.title(title)
    st.caption(subtitle)


def available_options(df, column):
    if column not in df.columns:
        return []
    values = df[column].dropna().astype(str).unique().tolist()
    if column == "age":
        return [age for age in AGE_ORDER if age in values] + sorted([v for v in values if v not in AGE_ORDER])
    return sorted(values)


def apply_filters(df, filters):
    data = df.copy()
    for column, selected in filters.items():
        if column in data.columns and selected:
            data = data[data[column].astype(str).isin(selected)]
    return data


def sidebar_filters(df):
    st.sidebar.header("Dashboard filters")
    filter_columns = {
        "age": "Age group",
        "gender": "Gender",
        "race": "Race",
        "readmission_label": "Readmission status",
        "admission_type_label": "Admission type",
        "insulin": "Insulin use",
        "diabetesMed": "Diabetes medication",
        "A1Cresult": "A1C result",
    }

    filters = {}
    for column, label in filter_columns.items():
        options = available_options(df, column)
        if options:
            filters[column] = st.sidebar.multiselect(label, options, default=options)

    use_full_dataset = st.sidebar.checkbox("Use full dataset", value=True)
    sample_n = len(df)
    if not use_full_dataset and len(df) > 1000:
        sample_n = st.sidebar.slider("Sample size", 1000, len(df), min(25000, len(df)), step=1000)

    filtered = apply_filters(df, filters)
    if not use_full_dataset and len(filtered) > sample_n:
        filtered = filtered.sample(sample_n, random_state=RANDOM_STATE)

    return filtered


def readmission_rate_by_group(df, group_col, top_n=15):
    if group_col not in df.columns or "readmitted_30" not in df.columns:
        return pd.DataFrame()

    grouped = (
        df.groupby(group_col, observed=False)
        .agg(encounters=("readmitted_30", "size"), readmission_rate=("readmitted_30", "mean"))
        .reset_index()
    )
    grouped["readmission_rate_pct"] = grouped["readmission_rate"] * 100
    grouped[group_col] = grouped[group_col].astype(str)
    grouped = grouped.sort_values(["encounters", "readmission_rate_pct"], ascending=[False, False]).head(top_n)
    return grouped


def make_kpi_cards(df):
    total_encounters = len(df)
    unique_patients = df["patient_nbr"].nunique() if "patient_nbr" in df.columns else None
    readmit_30_rate = df["readmitted_30"].mean() * 100 if len(df) else 0
    overall_readmit_rate = df["readmitted"].isin(["<30", ">30"]).mean() * 100 if "readmitted" in df.columns else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Hospital encounters", f"{total_encounters:,.0f}")
    c2.metric("Unique patients", f"{unique_patients:,.0f}" if unique_patients is not None else "N/A")
    c3.metric("30-day readmission", f"{readmit_30_rate:.1f}%")
    c4.metric("Any readmission", f"{overall_readmit_rate:.1f}%")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Avg. time in hospital", metric_average(df, "time_in_hospital", "days"))
    c6.metric("Avg. medications", metric_average(df, "num_medications"))
    c7.metric("Avg. lab procedures", metric_average(df, "num_lab_procedures"))
    c8.metric("Avg. diagnoses", metric_average(df, "number_diagnoses"))


def metric_average(df, column, suffix=""):
    if column not in df.columns or df.empty:
        return "N/A"
    value = df[column].mean()
    label = f"{value:.1f}"
    return f"{label} {suffix}".strip()


def plot_bar(df, x, y=None, title="", labels=None, color=None, top_n=15):
    if x not in df.columns:
        st.info(f"{x} is not available in this dataset.")
        return

    chart_data = df.copy()
    if y is None:
        chart_data = chart_data[x].astype(str).value_counts().head(top_n).reset_index()
        chart_data.columns = [x, "count"]
        y = "count"
    else:
        chart_data = chart_data.head(top_n)

    fig = px.bar(chart_data, x=x, y=y, color=color, title=title, labels=labels)
    fig.update_layout(xaxis_tickangle=-35, margin=dict(l=20, r=20, t=60, b=20))
    st.plotly_chart(fig, use_container_width=True)


def plot_rate_chart(df, group_col, title, top_n=15):
    rate_df = readmission_rate_by_group(df, group_col, top_n=top_n)
    if rate_df.empty:
        st.info(f"{group_col} is not available in this dataset.")
        return

    fig = px.bar(
        rate_df.sort_values("readmission_rate_pct", ascending=False),
        x=group_col,
        y="readmission_rate_pct",
        text="readmission_rate_pct",
        title=title,
        labels={group_col: group_col.replace("_", " ").title(), "readmission_rate_pct": "30-day readmission rate (%)"},
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(xaxis_tickangle=-35, yaxis_ticksuffix="%", margin=dict(l=20, r=20, t=60, b=20))
    st.plotly_chart(fig, use_container_width=True)


def plot_box(df, x, y, title):
    if x not in df.columns or y not in df.columns:
        st.info(f"{title} cannot be shown because one or more columns are unavailable.")
        return
    fig = px.box(df, x=x, y=y, color=x, title=title, points=False)
    fig.update_layout(showlegend=False, xaxis_tickangle=-25, margin=dict(l=20, r=20, t=60, b=20))
    st.plotly_chart(fig, use_container_width=True)


def plot_confusion_matrix(y_true, y_pred):
    matrix = confusion_matrix(y_true, y_pred)
    fig = px.imshow(
        matrix,
        text_auto=True,
        color_continuous_scale="Blues",
        labels=dict(x="Predicted", y="Actual", color="Encounters"),
        x=["Not <30", "<30"],
        y=["Not <30", "<30"],
        title="Confusion Matrix",
    )
    st.plotly_chart(fig, use_container_width=True)


def model_feature_columns(df):
    preferred = [
        "race",
        "gender",
        "age",
        "age_midpoint",
        "admission_type_label",
        "discharge_disposition_label",
        "admission_source_label",
        "time_in_hospital",
        "num_lab_procedures",
        "num_procedures",
        "num_medications",
        "number_outpatient",
        "number_emergency",
        "number_inpatient",
        "number_diagnoses",
        "max_glu_serum",
        "A1Cresult",
        "diag_1_group",
        "insulin",
        "change",
        "diabetesMed",
        "metformin",
        "glipizide",
        "glyburide",
    ]
    return [col for col in preferred if col in df.columns]


@st.cache_resource(show_spinner=False)
def train_readmission_model(df, feature_cols):
    model_df = df[feature_cols + ["readmitted_30"]].copy()
    model_df = model_df.dropna(subset=["readmitted_30"])

    if model_df["readmitted_30"].nunique() < 2 or len(model_df) < 300:
        return None

    X = model_df[feature_cols]
    y = model_df["readmitted_30"].astype(int)

    categorical_cols = X.select_dtypes(exclude=["number"]).columns.tolist()
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_cols),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_cols,
            ),
        ],
        remainder="drop",
    )

    model = RandomForestClassifier(
        n_estimators=160,
        max_depth=12,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }

    return {
        "pipeline": pipeline,
        "features": feature_cols,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "metrics": metrics,
    }


def get_feature_importance(model_result, top_n=15):
    if not model_result:
        return pd.DataFrame()

    pipeline = model_result["pipeline"]
    preprocessor = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]

    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        feature_names = np.array(model_result["numeric_cols"] + model_result["categorical_cols"])

    importance = pd.DataFrame(
        {
            "feature": [name.replace("num__", "").replace("cat__", "") for name in feature_names],
            "importance": model.feature_importances_,
        }
    )
    return importance.sort_values("importance", ascending=False).head(top_n)


def create_prediction_input(df, model_result):
    inputs = {}
    features = model_result["features"]

    c1, c2, c3 = st.columns(3)
    column_containers = [c1, c2, c3]

    for idx, feature in enumerate(features):
        container = column_containers[idx % 3]
        with container:
            if feature in model_result["numeric_cols"]:
                series = pd.to_numeric(df[feature], errors="coerce") if feature in df.columns else pd.Series([0])
                min_value = float(series.min()) if series.notna().any() else 0.0
                max_value = float(series.max()) if series.notna().any() else 100.0
                default = float(series.median()) if series.notna().any() else 0.0
                if min_value == max_value:
                    max_value = min_value + 1
                inputs[feature] = st.number_input(
                    feature.replace("_", " ").title(),
                    min_value=min_value,
                    max_value=max_value,
                    value=default,
                    step=1.0,
                )
            else:
                options = available_options(df, feature)
                if not options:
                    options = ["Unknown"]
                inputs[feature] = st.selectbox(feature.replace("_", " ").title(), options)

    return pd.DataFrame([inputs])


def insight_box(text):
    st.info(text)


def executive_overview(df):
    page_title(
        "Executive Overview",
        "A board-level summary of diabetic patient encounters and 30-day readmission patterns.",
    )
    make_kpi_cards(df)

    c1, c2 = st.columns(2)
    with c1:
        plot_bar(df, "readmission_label", title="Readmission Outcome Distribution")
    with c2:
        plot_rate_chart(df, "age", "30-Day Readmission Rate by Age Group", top_n=10)

    c3, c4 = st.columns(2)
    with c3:
        plot_rate_chart(df, "gender", "30-Day Readmission Rate by Gender")
    with c4:
        plot_rate_chart(df, "race", "30-Day Readmission Rate by Race")

    plot_box(df, "readmission_label", "time_in_hospital", "Time in Hospital by Readmission Status")
    insight_box(
        f"Within the selected population, the 30-day readmission rate is {df['readmitted_30'].mean() * 100:.1f}%. "
        "Older age groups and patients with heavier prior utilization may need stronger discharge planning."
    )


def patient_demographics(df):
    page_title("Patient Demographics", "Age, gender, and race patterns across readmission outcomes.")
    c1, c2 = st.columns(2)
    with c1:
        plot_bar(df, "age", title="Patient Count by Age Group", top_n=10)
    with c2:
        plot_rate_chart(df, "age", "30-Day Readmission Rate by Age Group", top_n=10)

    c3, c4 = st.columns(2)
    with c3:
        plot_rate_chart(df, "gender", "30-Day Readmission Rate by Gender")
    with c4:
        plot_rate_chart(df, "race", "30-Day Readmission Rate by Race")

    if {"age", "readmission_label"}.issubset(df.columns):
        stacked = df.groupby(["age", "readmission_label"], observed=False).size().reset_index(name="encounters")
        fig = px.bar(
            stacked,
            x="age",
            y="encounters",
            color="readmission_label",
            title="Readmission Status by Age Group",
            labels={"age": "Age group", "encounters": "Encounters", "readmission_label": "Readmission status"},
        )
        st.plotly_chart(fig, use_container_width=True)

    insight_box("Demographic patterns help identify patient groups that may require stronger discharge planning or follow-up.")


def admission_utilization(df):
    page_title(
        "Admission & Hospital Utilization",
        "How admission pathways, discharge destinations, and previous utilization relate to readmission.",
    )
    c1, c2 = st.columns(2)
    with c1:
        plot_rate_chart(df, "admission_type_label", "30-Day Readmission Rate by Admission Type")
    with c2:
        plot_rate_chart(df, "admission_source_label", "30-Day Readmission Rate by Admission Source")

    c3, c4 = st.columns(2)
    with c3:
        plot_rate_chart(df, "discharge_disposition_label", "30-Day Readmission Rate by Discharge Disposition", top_n=10)
    with c4:
        plot_box(df, "readmission_label", "time_in_hospital", "Time in Hospital by Readmission Status")

    utilization_cols = [col for col in ["number_inpatient", "number_emergency"] if col in df.columns]
    if utilization_cols:
        summary = df.groupby("readmission_label")[utilization_cols].mean().reset_index()
        melted = summary.melt(id_vars="readmission_label", var_name="Utilization metric", value_name="Average visits")
        fig = px.bar(
            melted,
            x="readmission_label",
            y="Average visits",
            color="Utilization metric",
            barmode="group",
            title="Average Prior Utilization by Readmission Status",
        )
        st.plotly_chart(fig, use_container_width=True)

    insight_box("Patients with greater previous inpatient or emergency utilization may represent a higher-risk segment.")


def treatment_medication(df):
    page_title("Treatment & Medication", "Treatment intensity and medication signals related to patient complexity.")
    c1, c2 = st.columns(2)
    with c1:
        plot_box(df, "readmission_label", "num_medications", "Number of Medications by Readmission Status")
    with c2:
        plot_box(df, "readmission_label", "num_lab_procedures", "Lab Procedures by Readmission Status")

    c3, c4, c5 = st.columns(3)
    with c3:
        plot_rate_chart(df, "insulin", "30-Day Readmission Rate by Insulin Use")
    with c4:
        plot_rate_chart(df, "change", "30-Day Readmission Rate by Medication Change")
    with c5:
        plot_rate_chart(df, "diabetesMed", "30-Day Readmission Rate by Diabetes Medication Use")

    med_cols = [col for col in ["metformin", "glipizide", "glyburide", "pioglitazone", "rosiglitazone"] if col in df.columns]
    if med_cols:
        rows = []
        for col in med_cols:
            active = df[df[col].astype(str).ne("No")]
            if not active.empty:
                rows.append(
                    {
                        "Medication": col.title(),
                        "Encounters": len(active),
                        "30-day readmission rate (%)": active["readmitted_30"].mean() * 100,
                    }
                )
        med_summary = pd.DataFrame(rows)
        if not med_summary.empty:
            fig = px.bar(
                med_summary,
                x="Medication",
                y="30-day readmission rate (%)",
                text="30-day readmission rate (%)",
                title="Readmission Rate Among Encounters With Selected Medications",
            )
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig.update_layout(yaxis_ticksuffix="%")
            st.plotly_chart(fig, use_container_width=True)

    insight_box("Medication intensity and treatment changes may reflect patient complexity rather than directly causing readmission.")


def clinical_profile(df):
    page_title("Clinical Profile", "Diagnosis burden, lab markers, and broad ICD-9 diagnosis groups.")
    c1, c2 = st.columns(2)
    with c1:
        plot_box(df, "readmission_label", "number_diagnoses", "Number of Diagnoses by Readmission Status")
    with c2:
        plot_rate_chart(df, "A1Cresult", "30-Day Readmission Rate by A1C Result")

    c3, c4 = st.columns(2)
    with c3:
        plot_rate_chart(df, "max_glu_serum", "30-Day Readmission Rate by Max Glucose Serum")
    with c4:
        plot_bar(df, "diag_1_group", title="Top Primary Diagnosis Groups")

    plot_rate_chart(df, "diag_1_group", "30-Day Readmission Rate by Primary Diagnosis Group")

    corr_cols = [
        col
        for col in [
            "time_in_hospital",
            "num_lab_procedures",
            "num_procedures",
            "num_medications",
            "number_outpatient",
            "number_emergency",
            "number_inpatient",
            "number_diagnoses",
            "readmitted_30",
        ]
        if col in df.columns
    ]
    if len(corr_cols) >= 2:
        corr = df[corr_cols].corr()
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", title="Correlation Heatmap")
        st.plotly_chart(fig, use_container_width=True)

    insight_box("Clinical complexity, represented by diagnosis count and diagnosis category, can help identify higher-risk patients.")


def predictive_model(df):
    page_title("Predictive Model", "Random Forest classifier for 30-day readmission risk.")
    feature_cols = model_feature_columns(df)

    if len(feature_cols) < 3 or df["readmitted_30"].nunique() < 2 or len(df) < 300:
        st.warning("Not enough class diversity to train the model under current filters. Please expand filters.")
        return None

    with st.spinner("Training readmission model..."):
        model_result = train_readmission_model(df, feature_cols)

    if model_result is None:
        st.warning("Not enough class diversity to train the model under current filters. Please expand filters.")
        return None

    metrics = model_result["metrics"]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy", f"{metrics['accuracy']:.2f}")
    c2.metric("Precision", f"{metrics['precision']:.2f}")
    c3.metric("Recall", f"{metrics['recall']:.2f}")
    c4.metric("F1-score", f"{metrics['f1']:.2f}")
    c5.metric("ROC-AUC", f"{metrics['roc_auc']:.2f}")

    st.info(
        "For readmission prediction, recall is especially important because hospitals want to identify as many truly high-risk patients as possible."
    )

    c6, c7 = st.columns(2)
    with c6:
        plot_confusion_matrix(model_result["y_test"], model_result["y_pred"])
    with c7:
        fpr, tpr, _ = roc_curve(model_result["y_test"], model_result["y_proba"])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="Random Forest"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Baseline", line=dict(dash="dash")))
        fig.update_layout(title="ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
        st.plotly_chart(fig, use_container_width=True)

    importance = get_feature_importance(model_result)
    if not importance.empty:
        fig = px.bar(
            importance.sort_values("importance"),
            x="importance",
            y="feature",
            orientation="h",
            title="Top 15 Predictors",
            labels={"importance": "Importance", "feature": "Feature"},
        )
        st.plotly_chart(fig, use_container_width=True)

    probability_df = pd.DataFrame(
        {
            "Predicted probability": model_result["y_proba"],
            "Actual class": np.where(model_result["y_test"].to_numpy() == 1, "Readmitted <30 Days", "Not <30 Days"),
        }
    )
    fig = px.histogram(
        probability_df,
        x="Predicted probability",
        color="Actual class",
        nbins=30,
        barmode="overlay",
        title="Predicted Probability Distribution by Actual Class",
    )
    st.plotly_chart(fig, use_container_width=True)

    return model_result


def risk_calculator(df):
    page_title("Readmission Risk Calculator", "Estimate 30-day readmission probability for a patient encounter.")
    feature_cols = model_feature_columns(df)

    if len(feature_cols) < 3 or df["readmitted_30"].nunique() < 2 or len(df) < 300:
        st.warning("Not enough class diversity to train the model under current filters. Please expand filters.")
        return

    with st.spinner("Preparing calculator model..."):
        model_result = train_readmission_model(df, feature_cols)

    if model_result is None:
        st.warning("Not enough class diversity to train the model under current filters. Please expand filters.")
        return

    patient_input = create_prediction_input(df, model_result)

    if st.button("Predict Readmission Risk", type="primary", use_container_width=True):
        probability = model_result["pipeline"].predict_proba(patient_input[model_result["features"]])[:, 1][0]
        if probability < 0.10:
            category = "Low Risk"
            action = "Routine discharge instructions."
        elif probability <= 0.20:
            category = "Medium Risk"
            action = "Follow-up call within 7 days and medication review."
        else:
            category = "High Risk"
            action = "Early follow-up appointment, care coordination, and discharge planning review."

        c1, c2 = st.columns(2)
        c1.metric("Predicted 30-day readmission probability", f"{probability * 100:.1f}%")
        c2.metric("Risk category", category)
        st.success(f"Suggested hospital action: {action}")


def data_dictionary_methodology(df):
    page_title("Data Dictionary & Methodology", "How the dashboard prepares the data and builds the prediction model.")
    st.markdown(
        """
### Dataset
The dataset contains diabetic patient hospital encounters with demographic, admission, utilization, treatment, clinical, and readmission variables.

### Target variable
The original `readmitted` column is transformed into `readmitted_30`:
- `1` = readmitted within 30 days
- `0` = not readmitted within 30 days

### Key variable groups
- Patient demographics: age, gender, race
- Admission and utilization: admission type, discharge disposition, admission source, length of stay, prior outpatient, emergency, and inpatient visits
- Treatment: medications, lab procedures, insulin, medication changes, diabetes medication use
- Clinical: diagnosis count, A1C result, glucose serum result, diagnosis groups

### Cleaning steps
- Replaced `?` values with missing values
- Removed duplicate rows
- Created binary 30-day readmission target
- Filled numeric missing values with medians
- Filled categorical missing values with `Unknown`
- Mapped admission, discharge, and source IDs where possible
- Grouped diagnosis codes into broad ICD-9 categories

### Model methodology
The prediction page uses a Random Forest Classifier with one-hot encoding for categorical variables, median imputation for numeric variables, a stratified train/test split, and class balancing.

### Limitations
- This is encounter-level data, and some patients may appear more than once.
- Readmission is influenced by social, behavioral, and healthcare access factors not fully captured here.
- The model identifies associations and risk patterns, not causal effects.
- Diagnosis groups and mapping fields are simplified for dashboard communication.
- The dashboard supports decision-making, but it is not a clinical diagnostic tool.

### Recommendations
- Focus discharge planning on patients with high prior inpatient or emergency utilization.
- Prioritize follow-up for older patients and patients with many diagnoses.
- Review medication changes and insulin use as markers of complex cases.
- Use the prediction model as a screening tool for care coordination.
"""
    )

    with st.expander("Raw filtered data preview"):
        st.dataframe(df.head(500), use_container_width=True)

    with st.expander("Available columns"):
        st.write(", ".join(df.columns))


def main():
    if not password_gate():
        return

    st.sidebar.title("Diabetes Readmission Dashboard")
    st.sidebar.caption("Consultant analytics view for hospital decision-makers")

    with st.sidebar.expander("Upload data", expanded=True):
        diabetic_file = st.file_uploader("Upload diabetic_data.csv", type=["csv"])
        mapping_file = st.file_uploader("Upload IDS_mapping.csv", type=["csv"])

    if diabetic_file is None:
        st.warning("Please upload diabetic_data.csv to begin.")
        return

    if mapping_file is None:
        st.info("Mapping file not uploaded. ID fields will be displayed as numeric codes.")

    try:
        raw_df = load_data(diabetic_file.getvalue())
        mappings = load_id_mappings(mapping_file)
        cleaned = clean_data(raw_df)
        data = apply_mappings(cleaned, mappings)
    except Exception as exc:
        st.error(f"The dashboard could not prepare the uploaded data: {exc}")
        return

    filtered = sidebar_filters(data)
    if filtered.empty:
        st.warning("No data available for the selected filters.")
        return

    pages = [
        "Executive Overview",
        "Patient Demographics",
        "Admission & Hospital Utilization",
        "Treatment & Medication",
        "Clinical Profile",
        "Predictive Model",
        "Readmission Risk Calculator",
        "Data Dictionary & Methodology",
    ]
    page = st.sidebar.radio("Dashboard page", pages)

    if page == "Executive Overview":
        executive_overview(filtered)
    elif page == "Patient Demographics":
        patient_demographics(filtered)
    elif page == "Admission & Hospital Utilization":
        admission_utilization(filtered)
    elif page == "Treatment & Medication":
        treatment_medication(filtered)
    elif page == "Clinical Profile":
        clinical_profile(filtered)
    elif page == "Predictive Model":
        predictive_model(filtered)
    elif page == "Readmission Risk Calculator":
        risk_calculator(filtered)
    else:
        data_dictionary_methodology(filtered)


if __name__ == "__main__":
    main()
