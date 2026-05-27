from pathlib import Path
import pandas as pd
import numpy as np
from scipy.io import arff


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_UCI = PROJECT_ROOT / "data" / "RawData" / "RawUCI"
RAW_KEEL = PROJECT_ROOT / "data" / "RawData" / "RawKEEL"
METADATA_DIR = PROJECT_ROOT / "metadata"
METADATA_DIR.mkdir(exist_ok=True)

OUTPUT_PATH = METADATA_DIR / "dataset_inventory_uci.csv"


DATASETS = {
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
    },    
    #new
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


#helper functions for arff, csv and dat

def load_arff_dataset(path: Path) -> pd.DataFrame:
    data, meta = arff.loadarff(path)
    df = pd.DataFrame(data)

    for col in df.select_dtypes([object]):
        df[col] = df[col].str.decode("utf-8")

    return df

def load_csv_dataset(path: Path, sep: str) -> pd.DataFrame:
 
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    return pd.read_csv(path, sep=sep, na_values=["?", "NA", "N/A", ""])
def load_keel_dat_dataset(path: Path) -> pd.DataFrame:

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    attribute_names = []
    data_started = False
    rows = []

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

def get_target_series(df: pd.DataFrame, target_col):

    if isinstance(target_col, str):
        y = df[target_col]
        X = df.drop(columns=[target_col])
        target_name = target_col
    elif isinstance(target_col, int):
        y = df.iloc[:, target_col]
        X = df.drop(columns=[df.columns[target_col]])
        target_name = df.columns[target_col]
    else:
        raise ValueError("target_col must be a string or integer column index.")

    return y, X, target_name


def summarize_dataset(dataset_name: str, config: dict) -> dict:
   
    if str(config["path"]).endswith(".arff"):
        df = load_arff_dataset(config["path"])
    elif str(config["path"]).endswith(".dat") and config["source"] == "KEEL":
        df = load_keel_dat_dataset(config["path"])
    else:
        df = load_csv_dataset(config["path"], config["sep"])
    print("\n", dataset_name)
    print(df.head())
    y, X, target_name = get_target_series(df, config["target_col"])

    class_counts = y.value_counts(dropna=False)
    print("Class counts:", class_counts.to_dict())
    n_classes = int(len(class_counts))

    majority_count = int(class_counts.max()) if n_classes > 0 else np.nan
    minority_count = int(class_counts.min()) if n_classes > 0 else np.nan
    imbalance_ratio = float(majority_count / minority_count) if minority_count not in [0, np.nan] else np.nan

    missing_values_total = int(df.isna().sum().sum())

    n_numeric_features = int(X.select_dtypes(include=[np.number]).shape[1])
    n_categorical_features = int(X.shape[1] - n_numeric_features)  

    return {
        "dataset_name": dataset_name,
        "source": config["source"],
        "n_rows": int(df.shape[0]),
        "n_features": int(X.shape[1]),
        "binary_or_multiclass": "binary" if n_classes == 2 else "multiclass",
        "IR": round(imbalance_ratio, 2) if n_classes == 2 else np.nan,
        "missing_values_total": missing_values_total,
        "n_numeric_features": n_numeric_features,
        "n_categorical_features": n_categorical_features,
        "notes": "",
        "keep": "",
    }



def main():
    rows = []

    for dataset_name, config in DATASETS.items():
        try:
            summary = summarize_dataset(dataset_name, config)
            rows.append(summary)
            print(f"Processed: {dataset_name}")
        except Exception as e:
            print(f"Failed: {dataset_name} -> {e}")

    inventory_df = pd.DataFrame(rows)
    inventory_df = inventory_df.sort_values("IR", ascending=True).reset_index(drop=True)
    inventory_df.to_csv(OUTPUT_PATH, index=False)


if __name__ == "__main__":
    main()