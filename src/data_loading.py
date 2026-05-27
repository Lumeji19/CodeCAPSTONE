from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from scipy.io import arff



PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_UCI = PROJECT_ROOT / "data" / "RawData" / "RawUCI"
RAW_KEEL = PROJECT_ROOT / "data" / "RawData" / "RawKEEL"



DATASETS: Dict[str, Dict[str, Any]] = {
    "bank_marketing": {
        "path": RAW_UCI / "bank+marketing" / "bank" / "bank-full.csv",
        "sep": ";",
        "target_col": "y",
        "source": "UCI",
        "header": 0,
    },
    "hcv_data": {
        "path": RAW_UCI / "hcv+data" / "hcvdat0.csv",
        "sep": ",",
        "target_col": 1,
        "source": "UCI",
        "header": 0,
    },
    "online_shoppers": {
        "path": RAW_UCI / "online+shoppers+intentions" / "online_shoppers_intention.csv",
        "sep": ",",
        "target_col": "Revenue",
        "source": "UCI",
        "header": 0,
    },
    "blood_transfusion": {
        "path": RAW_UCI / "blood+transfusion+service+center" / "transfusion.data",
        "sep": ",",
        "target_col": -1,
        "source": "UCI",
        "header": 0,
    },
    "ionosphere": {
        "path": RAW_UCI / "ionosphere" / "ionosphere.data",
        "sep": ",",
        "target_col": -1,
        "source": "UCI",
        "header": None,
    },
    "statlog_heart": {
        "path": RAW_UCI / "statlog+heart" / "heart.dat",
        "sep": r"\s+",
        "target_col": -1,
        "source": "UCI",
        "header": None,
    },
    "breast_cancer_wisconsin_diagnostic": {
        "path": RAW_UCI / "breast+cancer+wisconsin+diagnostic" / "wdbc.data",
        "sep": ",",
        "target_col": 1,
        "source": "UCI",
        "header": None,
    },
    "early_stage_diabetes_risk": {
        "path": RAW_UCI / "early+stage+diabetes+risk+prediction+dataset" / "diabetes_data_upload.csv",
        "sep": ",",
        "target_col": "class",
        "source": "UCI",
        "header": 0,
    },
    "parkinsons": {
        "path": RAW_UCI / "parkinsons" / "parkinsons.data",
        "sep": ",",
        "target_col": "status",
        "source": "UCI",
        "header": 0,
    },
    "heart_failure_clinical_records": {
        "path": RAW_UCI / "heart+failure+clinical+records" / "heart_failure_clinical_records_dataset.csv",
        "sep": ",",
        "target_col": "DEATH_EVENT",
        "source": "UCI",
        "header": 0,
    },
    "polish_bankruptcy_1year": {
        "path": RAW_UCI / "polish+companies+bankruptcy+data" / "1year.arff",
        "target_col": "class",
        "source": "UCI",
    },
    "polish_bankruptcy_2year": {
        "path": RAW_UCI / "polish+companies+bankruptcy+data" / "2year.arff",
        "target_col": "class",
        "source": "UCI",
    },
    "polish_bankruptcy_3year": {
        "path": RAW_UCI / "polish+companies+bankruptcy+data" / "3year.arff",
        "target_col": "class",
        "source": "UCI",
    },
    "polish_bankruptcy_4year": {
        "path": RAW_UCI / "polish+companies+bankruptcy+data" / "4year.arff",
        "target_col": "class",
        "source": "UCI",
    },
    "polish_bankruptcy_5year": {
        "path": RAW_UCI / "polish+companies+bankruptcy+data" / "5year.arff",
        "target_col": "class",
        "source": "UCI",
    },
    "abalone9_18": {
        "path": RAW_KEEL / "abalone9-18.dat",
        "target_col": "Class",
        "source": "KEEL",
    },
    "kr_vs_k_one_vs_fifteen": {
        "path": RAW_KEEL / "kr-vs-k-one_vs_fifteen.dat",
        "target_col": "Class",
        "source": "KEEL",
    },
    "car_good": {
        "path": RAW_KEEL / "car-good.dat",
        "target_col": "Class",
        "source": "KEEL",
    },
    "page_blocks0": {
        "path": RAW_KEEL / "page-blocks0.dat",
        "target_col": "Class",
        "source": "KEEL",
    },
    "flare_F": {
        "path": RAW_KEEL / "flare-F.dat",
        "target_col": "Class",
        "source": "KEEL",
    },
    "shuttle_c0_vs_c4": {
        "path": RAW_KEEL / "shuttle-c0-vs-c4.dat",
        "target_col": "Class",
        "source": "KEEL",
    },
    "vowel0": {
        "path": RAW_KEEL / "vowel0.dat",
        "target_col": "Class",
        "source": "KEEL",
    },
    "winequality_red_4": {
        "path": RAW_KEEL / "winequality-red-4.dat",
        "target_col": "Class",
        "source": "KEEL",
    },
    "yeast_0_2_5_7_9_vs_3_6_8": {
        "path": RAW_KEEL / "yeast-0-2-5-7-9_vs_3-6-8.dat",
        "target_col": "Class",
        "source": "KEEL",
    },
    "yeast3": {
        "path": RAW_KEEL / "yeast3.dat",
        "target_col": "Class",
        "source": "KEEL",
    },
    "kr_vs_k_three_vs_eleven": {
        "path": RAW_KEEL / "kr-vs-k-three_vs_eleven.dat",
        "target_col": "Class",
        "source": "KEEL",
    },
    "abalone_21_vs_8": {
        "path": RAW_KEEL / "abalone-21_vs_8.dat",
        "target_col": "Class",
        "source": "KEEL",
    },
    "yeast5": {
        "path": RAW_KEEL / "yeast5.dat",
        "target_col": "Class",
        "source": "KEEL",
    },
    "led7digit_0_2_4_5_6_7_8_9_vs_1": {
        "path": RAW_KEEL / "led7digit-0-2-4-5-6-7-8-9_vs_1.dat",
        "target_col": "Class",
        "source": "KEEL",
    },
    "dermatology_6": {
        "path": RAW_KEEL / "dermatology-6.dat",
        "target_col": "Class",
        "source": "KEEL",
    },
    "abalone_3_vs_11": {
        "path": RAW_KEEL / "abalone-3_vs_11.dat",
        "target_col": "Class",
        "source": "KEEL",
    },
    "glass4": {
        "path": RAW_KEEL / "glass4.dat",
        "target_col": "Class",
        "source": "KEEL",
    },
    "page_blocks_1_3_vs_4": {
        "path": RAW_KEEL / "page-blocks-1-3_vs_4.dat",
        "target_col": "Class",
        "source": "KEEL",
    },
    "shuttle_2_vs_5": {
        "path": RAW_KEEL / "shuttle-2_vs_5.dat",
        "target_col": "Class",
        "source": "KEEL",
    },
    "poker_9_vs_7": {
        "path": RAW_KEEL / "poker-9_vs_7.dat",
        "target_col": "Class",
        "source": "KEEL",
    }, #new datasets
    "autos": {
        "path": RAW_KEEL / "autos.dat",
        "target_col": "Class",
        "source": "KEEL",
    },
    "cleveland_0_vs_4": {
        "path": RAW_KEEL / "cleveland-0_vs_4.dat",
        "target_col": "Class",
        "source": "KEEL",
    },
    "glass2": {
        "path": RAW_KEEL / "glass2.dat",
        "target_col": "Class",
        "source": "KEEL",
    },
    "glass5": {
        "path": RAW_KEEL / "glass5.dat",
        "target_col": "Class",
        "source": "KEEL",
    },
    "kddcup_guess_passwd_vs_satan": {
        "path": RAW_KEEL / "kddcup-guess_passwd_vs_satan.dat",
        "target_col": "Class",
        "source": "KEEL",
    },
    "kddcup_land_vs_satan": {
        "path": RAW_KEEL / "kddcup-land_vs_satan.dat",
        "target_col": "Class",
        "source": "KEEL",
    },
    "kr_vs_k_zero_vs_fifteen": {
        "path": RAW_KEEL / "kr-vs-k-zero_vs_fifteen.dat",
        "target_col": "Class",
        "source": "KEEL",
    },
    "lymphography_normal_fibrosis": {
        "path": RAW_KEEL / "lymphography-normal-fibrosis.dat",
        "target_col": "Class",
        "source": "KEEL",
    },
    "thyroid": {
        "path": RAW_KEEL / "thyroid.dat",
        "target_col": "Class",
        "source": "KEEL",
    },
    "winequality_white_9_vs_4": {
        "path": RAW_KEEL / "winequality-white-9_vs_4.dat",
        "target_col": "Class",
        "source": "KEEL",
    },
    "zoo_3": {
        "path": RAW_KEEL / "zoo-3.dat",
        "target_col": "Class",
        "source": "KEEL",
    }, #new datasets
    "abalone_19_vs_10_11_12_13": {
        "path": RAW_KEEL / "abalone-19_vs_10-11-12-13.dat",
        "target_col": "Class",
        "source": "KEEL",
    },
    "ecoli_0_1_3_7_vs_2_6": {
        "path": RAW_KEEL / "ecoli-0-1-3-7_vs_2-6.dat",
        "target_col": "Class",
        "source": "KEEL",
    },
    "kddcup_land_vs_portsweep": {
        "path": RAW_KEEL / "kddcup-land_vs_portsweep.dat",
        "target_col": "Class",
        "source": "KEEL",
    },
    "kr_vs_k_zero_vs_eight": {
        "path": RAW_KEEL / "kr-vs-k-zero_vs_eight.dat",
        "target_col": "Class",
        "source": "KEEL",
    },
    "poker_8_vs_6": {
        "path": RAW_KEEL / "poker-8_vs_6.dat",
        "target_col": "Class",
        "source": "KEEL",
    },
    "yeast6": {
        "path": RAW_KEEL / "yeast6.dat",
        "target_col": "Class",
        "source": "KEEL",
    },
    "winequality_white_3_vs_7": {
        "path": RAW_KEEL / "winequality-white-3_vs_7.dat",
        "target_col": "Class",
        "source": "KEEL",
    },
    "winequality_red_8_vs_6_7": {
        "path": RAW_KEEL / "winequality-red-8_vs_6-7.dat",
        "target_col": "Class",
        "source": "KEEL",
    },
    "winequality_red_3_vs_5": {
        "path": RAW_KEEL / "winequality-red-3_vs_5.dat",
        "target_col": "Class",
        "source": "KEEL",
    },
    "poker_8_9_vs_6": {
        "path": RAW_KEEL / "poker-8-9_vs_6.dat",
        "target_col": "Class",
        "source": "KEEL",
    },
}


# loading helpers

def load_arff_dataset(path: Path) -> pd.DataFrame:

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    data, _meta = arff.loadarff(path)
    df = pd.DataFrame(data)

    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(
                lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
            )

    return df


def load_csv_dataset(path: Path, sep: str, header=0) -> pd.DataFrame:

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    return pd.read_csv(
        path,
        sep=sep,
        header=header,
        na_values=["?", "NA", "N/A", ""],
    )


def load_keel_dat_dataset(path: Path) -> pd.DataFrame:
   
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    attribute_names = []
    rows = []
    data_started = False

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("%"):
                continue

            lower_line = line.lower()

            if lower_line.startswith("@attribute"):
                parts = line.split()
                if len(parts) >= 2:
                    attribute_names.append(parts[1])

            elif lower_line.startswith("@data"):
                data_started = True
                continue

            elif data_started:
                row = [x.strip() for x in line.split(",")]
                rows.append(row)

    df = pd.DataFrame(rows, columns=attribute_names)
    df = df.replace("?", np.nan)

    return df


def load_raw_dataframe(dataset_name: str) -> pd.DataFrame:
   
    if dataset_name not in DATASETS:
        raise KeyError(f"Unknown dataset: {dataset_name}")

    config = DATASETS[dataset_name]
    path = config["path"]

    if str(path).endswith(".arff"):
        return load_arff_dataset(path)

    if str(path).endswith(".dat") and config["source"] == "KEEL":
        return load_keel_dat_dataset(path)

    return load_csv_dataset(
        path=path,
        sep=config["sep"],
        header=config.get("header", 0),
    )




#def convert_numeric_like_columns(df: pd.DataFrame) -> pd.DataFrame:
    
#    df = df.copy()

#    for col in df.columns:
#        df[col] = pd.to_numeric(df[col], errors="ignore")

#    return df

def convert_numeric_like_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert columns that look numeric into numeric type.
    Non-numeric columns are left alone.
    """
    df = df.copy()
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col], errors="raise")
        except (ValueError, TypeError):
            # Column has values that can't be converted to numeric
            # leave  for one-hot encoding
            pass
    return df


def impute_missing_numeric(X: pd.DataFrame) -> pd.DataFrame:

    X = X.copy()
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].median())
    return X


def one_hot_encode_categoricals(X: pd.DataFrame) -> pd.DataFrame:
    """


    For categorical columns, missing values are preserved as their own
    category (maybe change?) !!!
    """
    categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    # If there are no categorical columns, return unchanged
    if not categorical_cols:
        return X.copy()

    return pd.get_dummies(
        X,
        columns=categorical_cols,
        drop_first=True,
        dummy_na=True,      # preserve NaN as its own "missing" category
        dtype=float,        
    )


def preprocess_features(X: pd.DataFrame) -> pd.DataFrame:

    X = impute_missing_numeric(X)
    X = one_hot_encode_categoricals(X)
    return X


def get_target_series(df: pd.DataFrame, target_col) -> Tuple[pd.Series, pd.DataFrame, str]:

    if isinstance(target_col, str):
        y = df[target_col].copy()
        X = df.drop(columns=[target_col]).copy()
        target_name = target_col
    elif isinstance(target_col, int):
        target_name = df.columns[target_col]
        y = df.iloc[:, target_col].copy()
        X = df.drop(columns=[target_name]).copy()
    else:
        raise ValueError("error.")

    return y, X, target_name


def standardize_binary_target(y: pd.Series) -> pd.Series:

    y = y.copy()

    if y.isna().any():
        raise ValueError("Target column contains missing values.")

    class_counts = y.value_counts()

    if len(class_counts) != 2:
        raise ValueError("Target is not binary") # in case

    majority_label = class_counts.idxmax()
    minority_label = class_counts.idxmin()

    mapping = {
        majority_label: 0,
        minority_label: 1,
    }

    y = y.map(mapping)

    return y.astype(int)


def load_dataset(dataset_name: str) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:

    if dataset_name not in DATASETS:
        raise KeyError(f"Unknown dataset: {dataset_name}")

    config = DATASETS[dataset_name]
    
    df = load_raw_dataframe(dataset_name)
    df = convert_numeric_like_columns(df)

    y, X, target_name = get_target_series(df, config["target_col"])
    class_counts = y.value_counts()
    n_classes = int(len(class_counts))

    majority_count = int(class_counts.max()) if n_classes > 0 else np.nan
    minority_count = int(class_counts.min()) if n_classes > 0 else np.nan
    imbalance_ratio = float(majority_count / minority_count) if minority_count not in [0, np.nan] else np.nan


    
    n_features_before = X.shape[1]
    n_categorical_original = int(X.select_dtypes(exclude=[np.number]).shape[1])

    
    X = preprocess_features(X)

    
    y = standardize_binary_target(y)

    info = {
        "dataset_name": dataset_name,
        "source": config["source"],
        "target_name": target_name,
        "n_rows": len(df),
        "n_features": X.shape[1],
        "n_features_before_encoding": n_features_before,
        "n_categorical_original": n_categorical_original,
        "imbalance_ratio": round(imbalance_ratio, 2) if n_classes == 2 else np.nan,
        "n_minority": minority_count,
    }

    return X, y, info


def list_datasets() -> list:
    """
    Return the list of available dataset names.
    """
    return list(DATASETS.keys())