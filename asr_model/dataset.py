import pandas as pd
import torch
import torch.nn as nn
import torchaudio
from recognition.decoder import TextProcess
from recognition.utils import LogMelSpectrogram, SpecAugment
from torch.utils.data import Dataset

VALID_PATH = './dataset/valid.json'
TRAIN_PATH = './dataset/train.json'


class CommonVoiceDataset(Dataset):

    def __init__(
            self,
            sample_rate,
            n_feats=81,
            specaugment_rate=0.5,
            time_mask=70,
            frequency_mask=15,
            valid=False,
            log_ex=True
    ):
        self.json_path = VALID_PATH if valid else TRAIN_PATH
        self.data = pd.read_json(self.json_path, lines=True)

        self.log_ex = log_ex
        self.text_process = TextProcess()

        if valid:
            self.audio_transforms = torch.nn.Sequential(
                LogMelSpectrogram(sample_rate=sample_rate, n_mels=n_feats, win_length=160, hop_length=80)
            )
        else:
            self.audio_transforms = torch.nn.Sequential(
                LogMelSpectrogram(sample_rate=sample_rate, n_mels=n_feats, win_length=160, hop_length=80),
                SpecAugment(specaugment_rate, frequency_mask, time_mask)
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        try:
            file_path = self.data.iloc[index]['key']
            transcript = self.data.iloc[index]['text'].lower()

            waveform, _ = torchaudio.load(file_path)
            labels_array = self.text_process.text_to_int_sequence(transcript)
            spectrogram = self.audio_transforms(waveform)

            time_steps = spectrogram.shape[-1] // 2
            label_len = len(labels_array)

            if time_steps < label_len:
                raise Exception('Target length is bigger then input for file: %s' % file_path)
            if spectrogram.shape[0] > 1:
                raise Exception('Dual channel, skipping audio file: %s' % file_path)
            if label_len == 0:
                raise Exception('No text in file: %s' % file_path)
        except Exception as e:
            if self.log_ex:
                print('EXCEPTION: ', str(e))

            return self.__getitem__(index - 1 if index != 0 else index + 1)

        return {
            'spectrogram': spectrogram,
            'labels_array': labels_array,
            'time_steps': time_steps,
            'label_len': label_len
        }


def collate_fn(data):
    spectrograms = []
    items_labels = []
    input_lengths = []
    label_lengths = []

    for item in data:
        spectrograms.append(item['spectrogram'].squeeze(0).transpose(0, 1))
        items_labels.append(torch.Tensor(item['labels_array']))
        input_lengths.append(item['time_steps'])
        label_lengths.append(item['label_len'])

    spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).unsqueeze(1).transpose(2, 3)
    labels = nn.utils.rnn.pad_sequence(items_labels, batch_first=True)

    return spectrograms, labels, input_lengths, label_lengths
