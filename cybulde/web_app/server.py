import os

from fastapi import FastAPI
from hydra.utils import instantiate
from torch.nn.functional import sigmoid

from cybulde.models.common.exporter import TarModelLoader
from cybulde.utils.config_utils import load_config
from cybulde.utils.mlflow_utils import get_client

config = load_config(config_path="../configs/automatically_generated", config_name="config")
tokenizer = instantiate(config.tasks.binary_text_classification_task.data_module.transformation)

model_name = "bert_tiny"
model_version = "1"
mlflow_client = get_client()

mlflow_model = mlflow_client.get_model_version(name=model_name, version=model_version)
model_path = os.path.join(mlflow_model.source, "exported_model.tar.gz")  # type: ignore
model = TarModelLoader(exported_model_path=model_path).load()
model.eval()

app = FastAPI()


@app.get("/predict")
def predict_cyberbullying(text: str) -> dict[str, int]:
    print(f"{text=}")
    tokens = tokenizer([text])
    print(f"{tokens=}")
    probs = model(tokens)
    print(f"{probs=}")
    classes = (probs > 0.5).item()
    print(f"{classes=}")
    return {"is_cyberbullying": int(classes)}
