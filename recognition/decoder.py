import torch

from fast_ctc_decode import beam_search

BLANK = '_'

labels = [
    "'",  # 0
    " ",  # 1
    "a",  # 2
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "h",
    "i",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    "p",
    "q",
    "r",
    "s",
    "t",
    "u",
    "v",
    "w",
    "x",
    "y",
    "z",  # 27
    BLANK,  # 28, blank
]


class TextProcess:
    def __init__(self):
        self.char_map = {}
        self.index_map = {}

        for index, char in enumerate(labels):
            self.char_map[char] = index
            self.index_map[index] = char

    def text_to_int_sequence(self, text):
        int_sequence = []

        for char in text:
            corresponding_int = self.char_map.get(char, None)
            if corresponding_int: int_sequence.append(corresponding_int)

        return int_sequence

    def int_to_text_sequence(self, labels_array):
        transcript = []

        for index in labels_array:
            corresponding_char = self.index_map[index]
            transcript.append(corresponding_char)

        return ''.join(transcript)


textprocess = TextProcess()


class GreedyDecoder:

    def __init__(self, blank_label=28):
        self.blank_label = blank_label

    def __call__(self, output):
        arg_maxes = torch.argmax(output, dim=2).squeeze(1)[0]
        decode = []
        for i, index in enumerate(arg_maxes):
            if index != self.blank_label:
                if i != 0 and index == arg_maxes[i - 1]:
                    continue
                decode.append(index.item())
        return textprocess.int_to_text_sequence(decode)


class CTCBeamDecoder:

    def __init__(self, beam_size=100):
        self.beam_size = beam_size
        self.chars = ''.join(labels)

    def __call__(self, output):
        transcription, _ = beam_search(output[0].numpy(), self.chars, beam_size=self.beam_size)
        return self.delete_blank(transcription)

    def delete_blank(self, str):
        return ''.join(str.split(BLANK))
