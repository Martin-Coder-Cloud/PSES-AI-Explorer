# metadata_loader.py

import pandas as pd
import os
import gdown

# Required metadata files
REQUIRED_METADATA_FILES = {
    "layout": "metadata/filelayout.xlsx",
    "questions": "metadata/Survey Questions.xlsx",
    "themes": "metadata/Survey Themes.xlsx",
    "scales": "metadata/Survey Scales.xlsx",
    "demographics": "metadata/Demographics.xlsx"
}

# Dataset config
DATASET_URL_ID = "1Se-MqDGvcZWNftv7pstp1y-G_4N_LDMN"  # <- Your shared file ID
DATASET_LOCAL_PATH = "data/Main_Subset_2022_2023.csv.gz"

def download_csv_from_drive():
    url = f"https://drive.google.com/uc?id={DATASET_URL_ID}"
    os.makedirs("data", exist_ok=True)
    if not os.path.exists(DATASET_LOCAL_PATH):
        gdown.download(url, DATASET_LOCAL_PATH, quiet=False)
    return DATASET_LOCAL_PATH

def load_required_metadata():
    metadata = {}
    for key, path in REQUIRED_METADATA_FILES.items():
        if not os.path.exists(path):
            raise RuntimeError(f"Missing required metadata file: {path}")
        try:
            df = pd.read_excel(path)
            df.columns = [c.upper().strip() for c in df.columns]
            metadata[key] = df
        except Exception as e:
            raise RuntimeError(f"Error loading metadata file `{path}`: {e}")
    return metadata

def validate_dataset(layout_df):
    dataset_path = download_csv_from_drive()
    try:
        sample = pd.read_csv(dataset_path, compression='gzip', nrows=5)
        normalized_headers = [col.upper().strip() for col in sample.columns]
    except Exception as e:
        raise RuntimeError(f"Failed to read dataset headers: {e}")

    required_fields = ["QUESTION", "SURVEYR", "DEMCODE", "ANSWER1"]
    layout_column_names = layout_df["COLUMN_NAME"].str.upper().unique()

    for field in required_fields:
        if field not in layout_column_names:
            raise RuntimeError(f"`filelayout.xlsx` is missing required column: {field}")
        if field not in normalized_headers:
            raise RuntimeError(f"Dataset is missing required column: {field}")

    return dataset_path
