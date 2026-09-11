# -*- coding: utf-8 -*-
"""
Title     : PRF to NIT Mapping and Fitment of Resources to Project Requirements
Version   : 1.0
File Name : prf_nit_fuzzy_main.py 
Author    : Adityan Rajendran (adityanyogesh@gmail.com)
Comments
-------
For the uploaded dataset, the Technical category
alone requires thousands of repeated fuzzy simulation calls. Since there are only
five skill levels and five project requirement levels, the same fuzzy results are
repeated many times.

Fixes from beta version: 
-----------
This script replaces repeated fuzzy simulation calls with a cached lookup table
based on the same original fuzzy rules:
    None / Not Applicable / Basic / Capable / Proficient / Expert
    -> No Fittment / Can Contribute / Good Fit / Best Fit

The count logic, scoring thresholds, SQLite output tables and Power BI transpose
outputs are retained. Decision Tree refinement is optional and is applied only
if historical_allocations.csv is present.

Expected files in the same folder:
    NIT_PRF.xlsx

Optional file:
    historical_allocations.csv

Install requirements:
    pip install pandas numpy openpyxl scikit-learn joblib

Run:
    python prf_nit_fuzzy_decision_tree_debug_fast.py
"""

from __future__ import annotations

import sqlite3
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier, export_text
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix,
    )
    from sklearn.preprocessing import LabelEncoder
    import joblib

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

INPUT_EXCEL_FILE = "NIT_PRF.xlsx"
OUTPUT_DATABASE = "prf_nit_tech_func.db"
HISTORICAL_ALLOCATION_FILE = "historical_allocations.csv"

# Legacy spelling retained to avoid breaking old Power BI filters.
NO_FITMENT_LABEL = "No Fittment"

PRF_LEVEL_MAP = {
    "Not Applicable": 0,
    "Basic": 2,
    "Capable": 5,
    "Proficient": 8,
    "Expert": 9,
}

NIT_LEVEL_MAP = {
    "None": 0,
    "Basic": 2,
    "Capable": 5,
    "Proficient": 8,
    "Expert": 9,
}

# Original fuzzy output codes after rounding suitability output.
OUTPUT_CODE = {
    "Not Applicable": 0,
    NO_FITMENT_LABEL: 1,
    "Can Contribute": 2,
    "Good Fit": 5,
    "Best Fit": 8,
}

FITMENT_PRIORITY = {
    NO_FITMENT_LABEL: 0,
    "No Fitment": 0,
    "Can Contribute": 1,
    "Good Fit": 2,
    "Best Fit": 3,
}

CATEGORY_CODE = {
    "Technical": 1,
    "Functional": 2,
    "Others": 3,
}

REQUIRED_SHEETS = {
    "PRF_table_tech": "PRF",
    "PRF_table_func": "PRF",
    "PRF_table_oth": "PRF",
    "NIT_table_tech": "NIT",
    "NIT_table_func": "NIT",
    "NIT_table_oth": "NIT",
}

SQL_INPUT_TABLES = {
    "PRF_table_tech": "PRF_Table_tech_in",
    "PRF_table_func": "PRF_Table_func_in",
    "PRF_table_oth": "PRF_Table_oth_in",
    "NIT_table_tech": "NIT_Table_tech_in",
    "NIT_table_func": "NIT_Table_func_in",
    "NIT_table_oth": "NIT_Table_oth_in",
}


# ---------------------------------------------------------------------
# Section 1: Original fuzzy logic as fast lookup
# ---------------------------------------------------------------------

def original_fuzzy_label(resource_level: int, requirement_level: int) -> str:
    """
    Return the same category produced by the original fuzzy rule base.

    resource_level values:
        0=None, 2=Basic, 5=Capable, 8=Proficient, 9=Expert

    requirement_level values:
        0=Not Applicable, 2=Basic, 5=Capable, 8=Proficient, 9=Expert
    """

    # Not Applicable requirement always remains not applicable.
    if requirement_level == 0:
        return "Not Applicable"

    # Resource has no skill but project requires a skill.
    if resource_level == 0:
        return NO_FITMENT_LABEL

    # Original rules for Basic resource.
    if resource_level == 2:
        if requirement_level in (2, 5):
            return "Can Contribute"
        return NO_FITMENT_LABEL

    # Original rules for Capable resource.
    if resource_level == 5:
        if requirement_level == 2:
            return "Can Contribute"
        if requirement_level in (5, 8):
            return "Good Fit"
        if requirement_level == 9:
            return NO_FITMENT_LABEL

    # Original rules for Proficient resource.
    if resource_level == 8:
        if requirement_level in (2, 5):
            return "Good Fit"
        if requirement_level in (8, 9):
            return "Best Fit"

    # Original rules for Expert resource.
    if resource_level == 9:
        if requirement_level in (2, 5):
            return "Good Fit"
        if requirement_level in (8, 9):
            return "Best Fit"

    return NO_FITMENT_LABEL

def build_fuzzy_lookup() -> Dict[Tuple[int, int], int]:
    """
    Build lookup for all valid skill and requirement combinations.
    This is the main performance fix.
    """
    lookup = {}
    for resource_level in NIT_LEVEL_MAP.values():
        for requirement_level in PRF_LEVEL_MAP.values():
            label = original_fuzzy_label(int(resource_level), int(requirement_level))
            lookup[(int(resource_level), int(requirement_level))] = OUTPUT_CODE[label]
    return lookup


FUZZY_LOOKUP = build_fuzzy_lookup()


def apply_fuzzy_function_fast(skill_value: float, requirement_value: float) -> int:
    """Fast fuzzy result using original rule lookup."""
    return FUZZY_LOOKUP.get((int(skill_value), int(requirement_value)), OUTPUT_CODE[NO_FITMENT_LABEL])


# ---------------------------------------------------------------------
# Section 2: Data extraction and cleaning
# ---------------------------------------------------------------------

def is_missing_value(value) -> bool:
    """Treat NaN and blank strings as missing."""
    if pd.isna(value):
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


def missing_value_summary(df: pd.DataFrame, table_name: str, stage: str) -> List[dict]:
    """Compute missing value count and percentage for every column."""
    rows = []
    total_rows = len(df)

    for col in df.columns:
        missing_count = int(df[col].apply(is_missing_value).sum())
        missing_pct = round((missing_count / total_rows * 100), 2) if total_rows else 0.0
        rows.append({
            "Stage": stage,
            "Table_Name": table_name,
            "Column_Name": col,
            "Missing_Count": missing_count,
            "Total_Rows": total_rows,
            "Missing_Percentage": missing_pct,
        })

    return rows


def clean_input_table(
    df: pd.DataFrame,
    table_name: str,
    table_type: str,
) -> Tuple[pd.DataFrame, List[dict], List[dict]]:
    """Clean one PRF or NIT table and return cleaning logs."""

    cleaning_rows = []
    missing_rows = []
    missing_rows.extend(missing_value_summary(df, table_name, "Before Cleaning"))

    cleaned = df.copy()

    old_columns = list(cleaned.columns)
    cleaned.columns = [str(col).strip() for col in cleaned.columns]
    renamed_count = sum(1 for old, new in zip(old_columns, cleaned.columns) if str(old) != str(new))

    cleaning_rows.append({
        "Table_Name": table_name,
        "Cleaning_Action": "Column name trimming",
        "Records_Affected": renamed_count,
        "Remarks": "Leading/trailing spaces removed from column names",
    })

    object_cols = cleaned.select_dtypes(include=["object"]).columns
    for col in object_cols:
        cleaned[col] = cleaned[col].map(lambda x: x.strip() if isinstance(x, str) else x)

    cleaning_rows.append({
        "Table_Name": table_name,
        "Cleaning_Action": "Text value trimming",
        "Records_Affected": len(object_cols),
        "Remarks": "Leading/trailing spaces removed from text cells",
    })

    duplicate_count = int(cleaned.duplicated().sum())
    if duplicate_count:
        cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    cleaning_rows.append({
        "Table_Name": table_name,
        "Cleaning_Action": "Duplicate row removal",
        "Records_Affected": duplicate_count,
        "Remarks": "Exact duplicate rows removed",
    })

    key_col = cleaned.columns[0]
    missing_key_count = int(cleaned[key_col].apply(is_missing_value).sum())
    if missing_key_count:
        cleaned = cleaned[~cleaned[key_col].apply(is_missing_value)].reset_index(drop=True)

    cleaning_rows.append({
        "Table_Name": table_name,
        "Cleaning_Action": "Missing key identifier removal",
        "Records_Affected": missing_key_count,
        "Remarks": f"Rows without key identifier '{key_col}' removed",
    })

    skill_cols = list(cleaned.columns[1:])

    if table_type.upper() == "PRF":
        valid_values = set(PRF_LEVEL_MAP.keys())
        default_value = "Not Applicable"
    else:
        valid_values = set(NIT_LEVEL_MAP.keys())
        default_value = "None"

    missing_skill_count = 0
    invalid_skill_count = 0

    for col in skill_cols:
        missing_mask = cleaned[col].apply(is_missing_value)
        missing_skill_count += int(missing_mask.sum())
        cleaned.loc[missing_mask, col] = default_value

        invalid_mask = ~cleaned[col].isin(valid_values)
        invalid_skill_count += int(invalid_mask.sum())
        cleaned.loc[invalid_mask, col] = default_value

    cleaning_rows.append({
        "Table_Name": table_name,
        "Cleaning_Action": "Missing skill value replacement",
        "Records_Affected": missing_skill_count,
        "Remarks": f"Missing skill values replaced with '{default_value}'",
    })

    cleaning_rows.append({
        "Table_Name": table_name,
        "Cleaning_Action": "Invalid skill value replacement",
        "Records_Affected": invalid_skill_count,
        "Remarks": f"Invalid skill values replaced with '{default_value}'",
    })

    missing_rows.extend(missing_value_summary(cleaned, table_name, "After Cleaning"))

    return cleaned, cleaning_rows, missing_rows


def read_and_clean_workbook(file_path: str) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    """Read all required Excel sheets and clean them."""
    if not Path(file_path).exists():
        raise FileNotFoundError(f"Input workbook not found: {file_path}")

    workbook = pd.ExcelFile(file_path)
    missing_sheets = [sheet for sheet in REQUIRED_SHEETS if sheet not in workbook.sheet_names]
    if missing_sheets:
        raise ValueError(f"Missing required sheets: {missing_sheets}")

    cleaned_tables = {}
    cleaning_log_rows = []
    missing_summary_rows = []

    for sheet_name, table_type in REQUIRED_SHEETS.items():
        raw_df = pd.read_excel(file_path, sheet_name=sheet_name, keep_default_na=False)
        cleaned_df, clean_rows, missing_rows = clean_input_table(raw_df, sheet_name, table_type)

        cleaned_tables[sheet_name] = cleaned_df
        cleaning_log_rows.extend(clean_rows)
        missing_summary_rows.extend(missing_rows)

    return cleaned_tables, pd.DataFrame(cleaning_log_rows), pd.DataFrame(missing_summary_rows)


# ---------------------------------------------------------------------
# Section 3: Transformation
# ---------------------------------------------------------------------

def encode_skill_table(df: pd.DataFrame, table_type: str) -> pd.DataFrame:
    """Convert text skill levels into numeric values."""
    encoded = df.iloc[:, 1:].copy()
    mapping = PRF_LEVEL_MAP if table_type.upper() == "PRF" else NIT_LEVEL_MAP

    for col in encoded.columns:
        encoded[col] = encoded[col].map(mapping)

    if encoded.isna().any().any():
        bad_columns = encoded.columns[encoded.isna().any()].tolist()
        raise ValueError(f"Encoding failed for table type {table_type}. Bad columns: {bad_columns}")

    return encoded.astype(float)


def validate_category_columns(prf_df: pd.DataFrame, nit_df: pd.DataFrame, category_name: str):
    """Ensure PRF and NIT skill columns match for the category."""
    prf_cols = list(prf_df.columns[1:])
    nit_cols = list(nit_df.columns[1:])

    if prf_cols != nit_cols:
        raise ValueError(
            f"Skill columns do not match for {category_name}.\n"
            f"PRF columns: {prf_cols}\n"
            f"NIT columns: {nit_cols}"
        )


# ---------------------------------------------------------------------
# Section 4: Original scoring and count logic
# ---------------------------------------------------------------------

def classify_technical_fitment(skill_score: float, best_count: int, good_count: int, capable_count: int) -> str:
    """Original technical fitment classification logic."""
    if (
        skill_score >= 85
        or best_count >= 3
        or good_count >= 4
        or (best_count >= 2 and good_count >= 3)
    ):
        return "Best Fit"
    if (
        skill_score >= 65
        or (best_count >= 1 and good_count >= 2)
        or good_count >= 3
        or capable_count >= 4
    ):
        return "Good Fit"
    if (
        skill_score >= 50
        or (good_count >= 1 and capable_count >= 2)
        or capable_count >= 3
        or good_count >= 2
    ):
        return "Can Contribute"

    return NO_FITMENT_LABEL

def classify_functional_or_other_fitment(best_count: int, good_count: int, capable_count: int) -> str:
    """Original functional/other count-based fitment classification logic."""
    if best_count >= 3 or good_count >= 3 or (best_count >= 2 and good_count >= 2):
        return "Best Fit"

    if (best_count >= 1 and good_count >= 1) or good_count >= 2:
        return "Good Fit"

    if (good_count >= 1 and capable_count >= 1) or capable_count >= 2:
        return "Can Contribute"

    return NO_FITMENT_LABEL


def compute_category_fitment(
    category_name: str,
    prf_df: pd.DataFrame,
    nit_df: pd.DataFrame,
    prf_encoded: pd.DataFrame,
    nit_encoded: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute fuzzy fitment for Technical, Functional or Others category."""

    validate_category_columns(prf_df, nit_df, category_name)

    prf_id_col = prf_df.columns[0]
    nit_id_col = nit_df.columns[0]
    prf_ids = prf_df[prf_id_col].astype(str).tolist()
    nit_ids = nit_df[nit_id_col].astype(str).tolist()
    output_df = pd.DataFrame(index=nit_ids, columns=prf_ids)
    output_df.index.name = "Team Member"

    long_rows = []
    prf_project_scores = prf_encoded.sum(axis=1).tolist()
    number_of_resources = len(nit_encoded)

    for i in range(number_of_resources):
        resource_id = nit_ids[i]

        if (i + 1) % 10 == 0 or (i + 1) == number_of_resources:
            print(f"  {category_name}: processed {i + 1}/{number_of_resources} resources")

        nit_row = nit_encoded.iloc[i].to_numpy(dtype=int)

        for j in range(len(prf_encoded)):
            prf_id = prf_ids[j]
            prf_row = prf_encoded.iloc[j].to_numpy(dtype=int)

            fuzzy_outputs = tuple(
                apply_fuzzy_function_fast(nit_row[k], prf_row[k])
                for k in range(len(nit_row))
            )

            best_count = fuzzy_outputs.count(OUTPUT_CODE["Best Fit"])
            good_count = fuzzy_outputs.count(OUTPUT_CODE["Good Fit"])
            capable_count = fuzzy_outputs.count(OUTPUT_CODE["Can Contribute"])
            no_fitment_count = fuzzy_outputs.count(OUTPUT_CODE[NO_FITMENT_LABEL])
            not_applicable_count = fuzzy_outputs.count(OUTPUT_CODE["Not Applicable"])

            best_score = best_count * 8
            good_score = good_count * 5
            capable_score = capable_count * 2
            weighted_score = best_score + good_score + capable_score

            project_score = prf_project_scores[j]
            skill_score_pct = (weighted_score / project_score * 100) if project_score else 0

            if category_name == "Technical":
                final_fitment = classify_technical_fitment(skill_score_pct, best_count, good_count, capable_count)
            else:
                final_fitment = classify_functional_or_other_fitment(best_count, good_count, capable_count)

            output_df.loc[resource_id, prf_id] = final_fitment

            long_rows.append({
                "Category": category_name,
                "Category_Code": CATEGORY_CODE[category_name],
                "Team_Member": resource_id,
                "PRF": prf_id,
                "Best_Count": best_count,
                "Good_Count": good_count,
                "Capable_Count": capable_count,
                "No_Fitment_Count": no_fitment_count,
                "Not_Applicable_Count": not_applicable_count,
                "Project_Score": project_score,
                "Weighted_Score": weighted_score,
                "Skill_Score_Pct": round(skill_score_pct, 2),
                "Fuzzy_Fitment": final_fitment,
                "Fuzzy_Fitment_Code": FITMENT_PRIORITY.get(final_fitment, 0),
            })

    return output_df.reset_index(), pd.DataFrame(long_rows)


def prepare_power_bi_transpose(wide_df: pd.DataFrame) -> pd.DataFrame:
    """Prepare transposed output table for Power BI filters."""
    transposed = wide_df.set_index("Team Member").T
    transposed = transposed.reset_index().rename(columns={"index": "PRF"})
    return transposed


# ---------------------------------------------------------------------
# Section 5: Decision Tree refinement
# ---------------------------------------------------------------------

FEATURE_COLUMNS = [
    "Category_Code",
    "Best_Count",
    "Good_Count",
    "Capable_Count",
    "No_Fitment_Count",
    "Not_Applicable_Count",
    "Project_Score",
    "Weighted_Score",
    "Skill_Score_Pct",
    "Fuzzy_Fitment_Code",
]


def standardize_target_label(value) -> str:
    """Standardize fitment labels from historical allocation file."""
    if pd.isna(value):
        return NO_FITMENT_LABEL
    text = str(value).strip()
    text_lower = text.lower()
    if text_lower in ["no fitment", "no fittment", "nofitment", "no_fitment"]:
        return NO_FITMENT_LABEL
    if text_lower in ["can contribute", "contribute", "capable"]:
        return "Can Contribute"
    if text_lower in ["good fit", "good"]:
        return "Good Fit"
    if text_lower in ["best fit", "best"]:
        return "Best Fit"
    return text

def prepare_historical_training_data(path: str) -> pd.DataFrame:
    """
    Prepare historical allocation data for Decision Tree training.

    Preferred columns:
        Category_Code, Best_Count, Good_Count, Capable_Count,
        No_Fitment_Count, Not_Applicable_Count, Project_Score,
        Weighted_Score, Skill_Score_Pct, Fuzzy_Fitment_Code, Final_Fitment
    """
    historical = pd.read_csv(path)
    historical.columns = [col.strip() for col in historical.columns]

    if "Final_Fitment" in historical.columns and set(FEATURE_COLUMNS).issubset(historical.columns):
        training_df = historical[FEATURE_COLUMNS + ["Final_Fitment"]].copy()
        training_df["Final_Fitment"] = training_df["Final_Fitment"].map(standardize_target_label)
        return training_df

    if "final_fitment" in historical.columns and set(FEATURE_COLUMNS).issubset(historical.columns):
        training_df = historical[FEATURE_COLUMNS + ["final_fitment"]].copy()
        training_df = training_df.rename(columns={"final_fitment": "Final_Fitment"})
        training_df["Final_Fitment"] = training_df["Final_Fitment"].map(standardize_target_label)
        return training_df

    raise ValueError(
        "historical_allocations.csv does not contain the expected Decision Tree columns. "
        "Use the preferred columns mentioned in prepare_historical_training_data()."
    )


def run_decision_tree_refinement(long_df: pd.DataFrame, historical_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Apply Decision Tree refinement when historical data is available."""
    refined_df = long_df.copy()
    metric_rows = []

    if not SKLEARN_AVAILABLE:
        refined_df["Decision_Tree_Fitment"] = refined_df["Fuzzy_Fitment"]
        refined_df["Final_Refined_Fitment"] = refined_df["Fuzzy_Fitment"]
        refined_df["Decision_Tree_Status"] = "SKIPPED_SKLEARN_NOT_AVAILABLE"
        metric_rows.append({"Metric": "Status", "Value": "Skipped - scikit-learn not installed"})
        return refined_df, pd.DataFrame(metric_rows)

    if not Path(historical_path).exists():
        refined_df["Decision_Tree_Fitment"] = refined_df["Fuzzy_Fitment"]
        refined_df["Final_Refined_Fitment"] = refined_df["Fuzzy_Fitment"]
        refined_df["Decision_Tree_Status"] = "SKIPPED_NO_HISTORICAL_DATA"
        metric_rows.append({"Metric": "Status", "Value": "Skipped - historical_allocations.csv not found"})
        return refined_df, pd.DataFrame(metric_rows)

    training_df = prepare_historical_training_data(historical_path)

    X = training_df[FEATURE_COLUMNS].fillna(0)
    y = training_df["Final_Fitment"].map(standardize_target_label)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    class_counts = pd.Series(y_encoded).value_counts()
    can_use_test_split = len(class_counts) > 1 and class_counts.min() >= 2 and len(training_df) >= 8

    model = DecisionTreeClassifier(
        criterion="gini",
        max_depth=5,
        min_samples_split=2,
        random_state=42,
    )

    if can_use_test_split:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y_encoded,
            test_size=0.2,
            random_state=42,
            stratify=y_encoded,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metric_rows.extend([
            {"Metric": "Status", "Value": "TRAINED_WITH_TEST_SPLIT"},
            {"Metric": "Accuracy", "Value": round(accuracy_score(y_test, y_pred), 4)},
            {"Metric": "Precision_Weighted", "Value": round(precision_score(y_test, y_pred, average="weighted", zero_division=0), 4)},
            {"Metric": "Recall_Weighted", "Value": round(recall_score(y_test, y_pred, average="weighted", zero_division=0), 4)},
            {"Metric": "F1_Weighted", "Value": round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 4)},
            {"Metric": "Confusion_Matrix", "Value": str(confusion_matrix(y_test, y_pred).tolist())},
        ])
    else:
        model.fit(X, y_encoded)
        metric_rows.append({"Metric": "Status", "Value": "TRAINED_WITH_FULL_DATA_NO_TEST_SPLIT"})

    model_bundle = {
        "model": model,
        "label_encoder": label_encoder,
        "feature_columns": FEATURE_COLUMNS,
    }
    joblib.dump(model_bundle, "decision_tree_fitment_model.pkl")

    tree_rules = export_text(model, feature_names=FEATURE_COLUMNS)
    with open("decision_tree_rules.txt", "w", encoding="utf-8") as file:
        file.write(tree_rules)

    prediction_features = refined_df[FEATURE_COLUMNS].fillna(0)
    prediction_encoded = model.predict(prediction_features)
    predictions = label_encoder.inverse_transform(prediction_encoded)

    refined_df["Decision_Tree_Fitment"] = predictions
    refined_df["Final_Refined_Fitment"] = predictions
    refined_df["Decision_Tree_Status"] = "APPLIED"

    return refined_df, pd.DataFrame(metric_rows)


# ---------------------------------------------------------------------
# Section 6: SQLite output
# ---------------------------------------------------------------------

def write_dataframe(conn: sqlite3.Connection, df: pd.DataFrame, table_name: str):
    """Write dataframe to SQLite."""
    df.to_sql(table_name, conn, if_exists="replace", index=False)


def build_category_summary(long_df: pd.DataFrame) -> pd.DataFrame:
    """Build category-level fitment summary."""
    return (
        long_df.groupby(["Category", "Fuzzy_Fitment"])
        .size()
        .reset_index(name="Record_Count")
        .sort_values(["Category", "Fuzzy_Fitment"])
    )


def write_all_outputs_to_sql(
    db_path: str,
    cleaned_tables: Dict[str, pd.DataFrame],
    wide_outputs: Dict[str, pd.DataFrame],
    pbi_outputs: Dict[str, pd.DataFrame],
    long_df: pd.DataFrame,
    refined_df: pd.DataFrame,
    cleaning_log_df: pd.DataFrame,
    missing_summary_df: pd.DataFrame,
    dt_metrics_df: pd.DataFrame,
):
    """Write all outputs to SQLite database."""
    conn = sqlite3.connect(db_path)

    try:
        for sheet_name, table_name in SQL_INPUT_TABLES.items():
            write_dataframe(conn, cleaned_tables[sheet_name], table_name)

        write_dataframe(conn, wide_outputs["Technical"], "PRF_NIT_Table_out_tech")
        write_dataframe(conn, wide_outputs["Functional"], "PRF_NIT_Table_out_func")
        write_dataframe(conn, wide_outputs["Others"], "PRF_NIT_Table_out_oth")

        write_dataframe(conn, pbi_outputs["Technical"], "PRF_NIT_Table_out_tech_PBi")
        write_dataframe(conn, pbi_outputs["Functional"], "PRF_NIT_Table_out_func_PBi")
        write_dataframe(conn, pbi_outputs["Others"], "PRF_NIT_Table_out_oth_PBi")

        write_dataframe(conn, long_df, "PRF_NIT_Fitment_Long")
        write_dataframe(conn, refined_df, "PRF_NIT_DT_Refined_Long")
        write_dataframe(conn, build_category_summary(long_df), "PRF_NIT_Category_Summary")
        write_dataframe(conn, cleaning_log_df, "PRF_NIT_Data_Cleaning_Summary")
        write_dataframe(conn, missing_summary_df, "PRF_NIT_Missing_Value_Summary")
        write_dataframe(conn, dt_metrics_df, "PRF_NIT_Decision_Tree_Metrics")

    finally:
        conn.close()


# ---------------------------------------------------------------------
# Section 7: Main execution
# ---------------------------------------------------------------------

def main():
    warnings.filterwarnings("ignore", category=FutureWarning)

    print("Starting PRF-NIT Fuzzy + Decision Tree mapping process...")

    print("Reading and cleaning workbook...")
    cleaned_tables, cleaning_log_df, missing_summary_df = read_and_clean_workbook(INPUT_EXCEL_FILE)

    print("Encoding skill levels...")
    prf_tech_num = encode_skill_table(cleaned_tables["PRF_table_tech"], "PRF")
    prf_func_num = encode_skill_table(cleaned_tables["PRF_table_func"], "PRF")
    prf_oth_num = encode_skill_table(cleaned_tables["PRF_table_oth"], "PRF")

    nit_tech_num = encode_skill_table(cleaned_tables["NIT_table_tech"], "NIT")
    nit_func_num = encode_skill_table(cleaned_tables["NIT_table_func"], "NIT")
    nit_oth_num = encode_skill_table(cleaned_tables["NIT_table_oth"], "NIT")

    wide_outputs = {}
    long_outputs = []

    print("Computing Technical fitment...")
    wide_outputs["Technical"], long_tech = compute_category_fitment(
        "Technical",
        cleaned_tables["PRF_table_tech"],
        cleaned_tables["NIT_table_tech"],
        prf_tech_num,
        nit_tech_num,
    )
    long_outputs.append(long_tech)

    print("Computing Functional fitment...")
    wide_outputs["Functional"], long_func = compute_category_fitment(
        "Functional",
        cleaned_tables["PRF_table_func"],
        cleaned_tables["NIT_table_func"],
        prf_func_num,
        nit_func_num,
    )
    long_outputs.append(long_func)

    print("Computing Others fitment...")
    wide_outputs["Others"], long_oth = compute_category_fitment(
        "Others",
        cleaned_tables["PRF_table_oth"],
        cleaned_tables["NIT_table_oth"],
        prf_oth_num,
        nit_oth_num,
    )
    long_outputs.append(long_oth)

    print("Combining long-form output...")
    long_df = pd.concat(long_outputs, ignore_index=True)

    print("Preparing Power BI transposed tables...")
    pbi_outputs = {
        category: prepare_power_bi_transpose(output_df)
        for category, output_df in wide_outputs.items()
    }

    print("Running optional Decision Tree refinement...")
    refined_df, dt_metrics_df = run_decision_tree_refinement(long_df, HISTORICAL_ALLOCATION_FILE)

    print("Writing outputs to SQLite database...")
    write_all_outputs_to_sql(
        OUTPUT_DATABASE,
        cleaned_tables,
        wide_outputs,
        pbi_outputs,
        long_df,
        refined_df,
        cleaning_log_df,
        missing_summary_df,
        dt_metrics_df,
    )

    print("Process completed successfully.")
    print(f"Output database: {OUTPUT_DATABASE}")

    if Path(HISTORICAL_ALLOCATION_FILE).exists():
        print("Decision Tree model saved: decision_tree_fitment_model.pkl")
        print("Decision Tree rules saved: decision_tree_rules.txt")
    else:
        print("Decision Tree skipped safely because historical_allocations.csv was not found.")


if __name__ == "__main__":
    main()
