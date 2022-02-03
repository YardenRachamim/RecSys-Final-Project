import torch
from configuration import config


def load_trained_model(model_name: str, eval=True):
    model_path = config['models_path'] / model_name

    checkpoint = torch.load(model_path)
    model = checkpoint["model"]
    if eval:
        model.eval()
    else:
        model.train()

    return model
