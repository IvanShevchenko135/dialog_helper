import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torchaudio
from deep_translator import GoogleTranslator
from next_word_prediction import GPT2


@st.cache(hash_funcs={GPT2: lambda _: None})
def load_prediction_model():
    return GPT2()


@st.cache(hash_funcs={GoogleTranslator: lambda _: None})
def translator(to_translate):
    return GoogleTranslator(source='en', target='ru').translate(to_translate)


class SpecAugment(nn.Module):

    def __init__(self, rate=0.5, freq_mask=15, time_mask=35):
        super(SpecAugment, self).__init__()

        self.rate = rate

        self.specaugment = nn.Sequential(
            torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask),
            torchaudio.transforms.TimeMasking(time_mask_param=time_mask)
        )

        self.specaugment2 = nn.Sequential(
            torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask),
            torchaudio.transforms.TimeMasking(time_mask_param=time_mask),
            torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask),
            torchaudio.transforms.TimeMasking(time_mask_param=time_mask)
        )

    def forward(self, x):
        return self.rand_policy(x)

    def policy1(self, x):
        probability = torch.rand(1, 1).item()

        if self.rate > probability:
            return self.specaugment(x)

        return x

    def policy2(self, x):
        probability = torch.rand(1, 1).item()

        if self.rate > probability:
            return self.specaugment2(x)

        return x

    def rand_policy(self, x):
        probability = torch.rand(1, 1).item()

        if probability > 0.5:
            return self.policy1(x)

        return self.policy2(x)


class LogMelSpectrogram(nn.Module):

    def __init__(self, sample_rate=8000, n_mels=81, win_length=160, hop_length=80):
        super(LogMelSpectrogram, self).__init__()

        self.transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
            win_length=win_length,
            hop_length=hop_length
        )

    def forward(self, x):
        x = self.transform(x)
        x = np.log(x + 1e-14)

        return x


def get_feature_extractor(sample_rate=8000, n_feats=81):
    return LogMelSpectrogram(sample_rate=sample_rate, n_mels=n_feats, win_length=160, hop_length=80)


if __name__ == "__main__":
    print('1')

    for i in range(1, 1000):
        print(translator('good'))

    print('2')
