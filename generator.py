import pretty_midi
import numpy as np

import torch
import torch.nn as nn

import model as vaeModel

from utils import *
from config import *
from model import ModelConfig


def drumSeqIdx_to_note(note_idx):
    """
    Randomly select drum note from inferenced drum note index which is assigned during preprocess
    """

    idx_to_rolandNotes = dict()
    for value, key in params["roland_to_idx"].items():
        if key in idx_to_rolandNotes.keys():
            idx_to_rolandNotes[key] = idx_to_rolandNotes[key] + [value]
        else:
            idx_to_rolandNotes[key] = [value]

    return np.random.choice(idx_to_rolandNotes[note_idx])


def latent_generator(x):
    return (torch.randn(1, ModelConfig.latent_dim), None, None)


def make_gererating_model():
    """
    Changes total combined model to inferece model by replacing encoder part with latent generator
    """
    device = get_device()
    model = vaeModel.Model(device)
    checkpoint = torch.load(params["path_model_trained"])
    model.load_state_dict(checkpoint["model_state_dict"])

    for param in model.parameters():
        param.requires_grad = False

    model._modules["encoder"] = latent_generator
    return model


if __name__ == "__main__":
    beat_time = 1 / 4
    velocity = 60
    output_filename = "./generated/generated.midi"

    model = make_gererating_model()

    # generate onehot output from generated random latent vec
    output, _, _ = model(None)
    softmax = nn.Softmax(dim=2)(output)
    output = output.cpu().detach().numpy()

    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=32, is_drum=True)

    # argmax iteration from onehot
    for idx, onehot_idx in enumerate(np.argmax(np.squeeze(output), axis=1)):
        start_time = beat_time * idx
        end_time = beat_time * (idx + 1)

        # onehot_idx to drum_seq
        drum_seq = np.array(list(map(int, list(np.binary_repr(onehot_idx)))))

        # get drum channel classes from drum_seq
        for onehot_idx in np.where(drum_seq == 1)[0]:
            pitch = drumSeqIdx_to_note(onehot_idx)
            inst.notes.append(pretty_midi.Note(velocity, pitch, start_time, end_time))

    pm.instruments.append(inst)
    pm.write(output_filename)
    print("생성 완료")
