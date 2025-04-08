import torch
from .MST_Plus_Plus import MST_Plus_Plus

def model_generator(method, pretrained_model_path=None):
    model = MST_Plus_Plus()  # Removed .cuda() for CPU compatibility
    if pretrained_model_path is not None:
        print(f'load model from {pretrained_model_path}')
        checkpoint = torch.load(pretrained_model_path, map_location=torch.device('cpu'))  # Ensure loaded to CPU
        model.load_state_dict(
            {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()},
            strict=True
        )
    return model
