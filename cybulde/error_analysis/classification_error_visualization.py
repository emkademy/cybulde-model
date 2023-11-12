from collections import defaultdict

import pandas as pd
import streamlit as st

from cybulde.models.common.exporter import TarModelLoader
from cybulde.models.models import Model
from cybulde.models.transformations import HuggingFaceTokenizationTransformation


@st.cache_data
def load_dataframe(dataframe_parquet_path: str) -> pd.DataFrame:
    return pd.read_parquet(dataframe_parquet_path)


@st.cache_resource
def load_transformation(tokenizer_path: str, max_sequence_length: int) -> HuggingFaceTokenizationTransformation:
    return HuggingFaceTokenizationTransformation(tokenizer_path, max_sequence_length=max_sequence_length)


@st.cache_resource
def load_model(tar_model_path: str) -> Model:
    return TarModelLoader(tar_model_path).load()


@st.cache_data
def get_error_df(
    dataframe_parquet_path: str, tokenizer_path: str, max_sequence_length: int, tar_model_path: str
) -> pd.DataFrame:
    df = load_dataframe(dataframe_parquet_path)
    transformation = load_transformation(tokenizer_path, max_sequence_length)
    model = load_model(tar_model_path)

    errors_dict = defaultdict(list)
    for _, row in df.iterrows():
        text = row["text"]
        cleaned_text = row["cleaned_text"]
        label = row["label"]
        dataset_name = row["dataset_name"]

        encodings = transformation([cleaned_text])
        logits = model(encodings)[0]
        predicted_label = logits >= 0.5

        if predicted_label != label:
            errors_dict["text"].append(text)
            errors_dict["cleaned_text"].append(cleaned_text)
            errors_dict["gt_label"].append(label)
            errors_dict["dataset_name"].append(dataset_name)

    return pd.DataFrame(errors_dict)


@st.cache_data
def classification_error_analysis(
    dataframe_parquet_path: str, tokenizer_path: str, max_sequence_length: int, tar_model_path: str
) -> None:
    error_df = get_error_df(dataframe_parquet_path, tokenizer_path, max_sequence_length, tar_model_path)
    st.dataframe(error_df)


if __name__ == "__main__":
    dataframe_parquet_path = "gs://emkademy/cybulde/data/processed/rebalanced_splits/test.parquet"
    tokenizer_path = "gs://emkademy/cybulde/data/processed/rebalanced_splits/trained_tokenizer"
    max_sequence_length = 100
    tar_model_path = "/mlflow-artifact-store/5/78254570f7ee45f29e6e3f5593633384/artifacts/exported_model.tar.gz"

    classification_error_analysis(dataframe_parquet_path, tokenizer_path, max_sequence_length, tar_model_path)
