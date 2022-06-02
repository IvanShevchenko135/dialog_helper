"""Freezes and optimize the model. Use after training."""
from collections import OrderedDict

import torch

from model import SpeechRecognition
from train import model_hyper_parameters


def trace(model):
    model.eval()

    sample = torch.rand(1, 81, 300)
    hidden = model.init_hidden(1)
    traced_model = torch.jit.trace(model, (sample, hidden))

    return traced_model


# removing 'model.' from state dictionary keys
def remove_unnecessary_params_from_dict(state_dict):
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        new_key = key.replace('model.', '')
        new_state_dict[new_key] = value

    return new_state_dict


def save_model(checkpoint_path, save_to):
    print('Loading model from', checkpoint_path, '...')
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

    model = SpeechRecognition(**model_hyper_parameters)

    print('Removing unnecessary params...')
    model_state_dict = checkpoint['state_dict']
    model_state_dict = remove_unnecessary_params_from_dict(model_state_dict)

    model.load_state_dict(model_state_dict)

    print('Tracing model...')
    traced_model = trace(model)

    print('Saving to', save_to, '...')
    traced_model.save(save_to)

    print('Model successfully saved!')


if __name__ == "__main__":
    save_model('./best_models/asr_model-epoch=50.ckpt', '../recognition/model.zip')
