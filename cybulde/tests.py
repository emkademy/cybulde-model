from cybulde.data_modules.transformations import HuggingFaceTokenizationTransformation
from cybulde.models.backbones import HuggingFaceBackbone



pretrained_tokenizer_name_or_path = "gs://emkademy/cybulde/data/processed/rebalanced_splits/trained_tokenizer"
max_sequence_length = 72

tokenizer = HuggingFaceTokenizationTransformation(pretrained_tokenizer_name_or_path, max_sequence_length)

texts = ["hi! how are you?"]

encodings = tokenizer(texts)

backbone = HuggingFaceBackbone(pretrained_model_name_or_path="bert-base-uncased", pretrained=False)


output = backbone(encodings).pooler_output

print(output)
print(output.shape)
