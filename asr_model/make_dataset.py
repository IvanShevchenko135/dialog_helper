import csv
import json
import random
import warnings

import librosa
import soundfile as sf
from progress.bar import IncrementalBar

warnings.filterwarnings('ignore', 'PySoundFile failed. Trying audioread instead.')


def make_dataset(file_path, save_path, json_name, downsample_to=None, limit=None):
    data = []
    directory = file_path.rpartition('/')[0]

    with open(file_path) as f:
        length = sum(1 for line in f)

    if not limit:
        limit = length

    bar_files = IncrementalBar('Files', max=min(limit, length))
    bar_jsons = IncrementalBar('Jsons', max=min(limit, length))

    with open(file_path, newline='') as f:
        reader = csv.DictReader(f, delimiter='\t')
        index = 1
        for row in reader:
            filename_old = row['path'] + '.mp3'
            filename_new = row['path'] + '.wav'

            text = row['sentence']

            data.append({
                'key': save_path + '/clips/' + filename_new,
                'text': text
            })

            src = directory + '/clips/' + filename_old
            dst = save_path + '/clips/' + filename_new

            sound, sample_rate = librosa.load(src, sr=downsample_to)
            sf.write(dst, sound, sample_rate)

            bar_files.next()

            index = index + 1

            if index > limit:
                break

    bar_files.finish()
    random.shuffle(data)

    with open(save_path + '/' + json_name + '.json', 'w+') as f:
        i = 0
        while i <= limit and i < len(data):
            current = data[i]
            line = json.dumps(current)
            f.write(line + '\n')
            bar_jsons.next()
            i = i + 1

    bar_jsons.finish()


if __name__ == '__main__':
    make_dataset('../../../../Downloads/en/test.tsv', 'dataset', 'valid', downsample_to=8000, limit=5)
