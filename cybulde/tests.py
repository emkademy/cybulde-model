from cybulde.models.common.exporter import TarModelLoader
from cybulde.models.transformations import HuggingFaceTokenizationTransformation

tar_model_path = "/mlflow-artifact-store/36/6c798daf6a6c4616a62a2f85fc029dc0/artifacts/exported_model.tar.gz"
model = TarModelLoader(tar_model_path).load()


pretrained_tokenizer_name_or_path: str = "gs://emkademy/cybulde/data/processed/rebalanced_splits/trained_tokenizer"
max_sequence_length: int = 100
transformation = HuggingFaceTokenizationTransformation(pretrained_tokenizer_name_or_path, max_sequence_length)

text = ["fuck you"]
encodings = transformation(text)
logits = model(encodings)

print(logits)
