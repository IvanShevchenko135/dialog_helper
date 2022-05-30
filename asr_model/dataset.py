import pandas as pd
import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import Dataset

from dialog_helper.recognition.decoder import TextProcess
from dialog_helper.recognition.utils import LogMelSpectrogram, SpecAugment


class CommonVoiceDataset(Dataset):

    def __init__(self,
                 json_path,
                 sample_rate,
                 n_feats=81,
                 specaugment_rate=0.5,
                 time_mask=70,
                 frequency_mask=15,
                 valid=False,
                 shuffle=True,
                 text_to_int=True,
                 log_ex=True
                 ):
        self.data = pd.read_json(json_path, lines=True)

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
        # if torch.is_tensor(index):
        #     index = index.item()

        try:
            file_path = self.data.iloc[index]['key']
            transcript = self.data.iloc[index]['text'].lower()

            waveform, _ = torchaudio.load(file_path)
            labels_array = self.text_process.text_to_int_sequence(transcript)
            spectrogram = self.audio_transforms(waveform)  # (channel, feature, time)

            time_steps = spectrogram.shape[-1] // 2
            label_len = len(labels_array)

            if spectrogram.shape[0] > 1:
                raise Exception('Dual channel, skipping audio file %s' % file_path)
            if spectrogram.shape[2] > 1650:
                raise Exception('Spectrogram is too big. Size: %s' % spectrogram.shape[2])
            if label_len == 0:
                raise Exception('No text in file: %s' % file_path)
        except Exception as e:
            if self.log_ex:
                print('EXCEPTION: ', str(e))

            return self.__getitem__(index - 1 if index != 0 else index + 1)

        return spectrogram, labels_array, time_steps, label_len


def collate_fn_padd(data):
    '''
    padds batch of variable length

    note: it converts things ToTensor manually here since the ToTensor transform
    assume it takes in images rather than arbitrary tensors.
    '''
    # print(data)
    spectrograms = []
    labels = []
    input_lengths = []
    label_lengths = []
    for (spectrogram, label, input_length, label_length) in data:
        if spectrogram is None:
            continue
        # print(spectrogram.shape)
        spectrograms.append(spectrogram.squeeze(0).transpose(0, 1))
        labels.append(torch.Tensor(label))
        input_lengths.append(input_length)
        label_lengths.append(label_length)

    spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).unsqueeze(1).transpose(2, 3)
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)
    input_lengths = input_lengths
    # print(spectrograms.shape)
    label_lengths = label_lengths
    # ## compute mask
    # mask = (batch != 0).cuda(gpu)
    # return batch, lengths, mask
    return spectrograms, labels, input_lengths, label_lengths


if __name__ == "__main__":
    # tp = TextProcess()
    # tp.text_to_int_sequence('The field hockey team is trying to score a goal.')

    cvd = CommonVoiceDataset('./dataset/train.json', 8000)
    one, two, three, four = cvd.__getitem__(0)
    print(three, four)

    # loader = DataLoader(cvd, 1)
# [21, 9, 6, 1, 7, 10, 6, 13, 5, 1, 9, 16, 4, 12, 6, 26, 1, 21, 6, 2, 14, 1, 10, 20, 1, 21, 19, 26, 10, 15, 8, 1, 21, 16, 1, 20, 4, 16, 19, 6, 1, 2, 1, 8, 16, 2, 13]?
