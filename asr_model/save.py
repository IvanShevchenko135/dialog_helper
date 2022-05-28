"""Freezes and optimize the model. Use after training."""
import argparse
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
    parser = argparse.ArgumentParser(description='Saving model')
    parser.add_argument('--model_path', type=str, help='Path to best checkpoint', required=True)
    parser.add_argument('--save_to', type=str, help='Path to save the model', required=True)
    args = parser.parse_args()

    save_model(args.model_path, args.save_to)
