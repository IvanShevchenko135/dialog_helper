import torch

from fast_ctc_decode import beam_search


class TextProcess:
    def __init__(self):
        char_map_str = """
		' 0
		<SPACE> 1
		a 2
		b 3
		c 4
		d 5
		e 6
		f 7
		g 8
		h 9
		i 10
		j 11
		k 12
		l 13
		m 14
		n 15
		o 16
		p 17
		q 18
		r 19
		s 20
		t 21
		u 22
		v 23
		w 24
		x 25
		y 26
		z 27
		"""
        self.char_map = {}
        self.index_map = {}
        for line in char_map_str.strip().split('\n'):
            ch, index = line.split()
            self.char_map[ch] = int(index)
            self.index_map[int(index)] = ch
        self.index_map[1] = ' '

    def text_to_int_sequence(self, text):
        """ Use a character map and convert text to an integer sequence """
        int_sequence = []
        for c in text:
            if c == ' ':
                ch = self.char_map['<SPACE>']
            else:
                ch = self.char_map[c]
            int_sequence.append(ch)
        return int_sequence

    def int_to_text_sequence(self, labels):
        """ Use a character map and convert integer labels to an text sequence """
        string = []
        for i in labels:
            string.append(self.index_map[i])
        return ''.join(string).replace('<SPACE>', ' ')


textprocess = TextProcess()

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
