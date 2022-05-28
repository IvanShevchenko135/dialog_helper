import threading
import time
import wave

import pyaudio
import torch
import torchaudio
from textblob import TextBlob

from .decoder import CTCBeamDecoder
from .utils import get_feature_extractor


class TranscriptWriter:

    def __init__(self, file_name):
        self.all_beams = ''
        self.current_beam = ''

        self.file_name = file_name

        with open(self.file_name, 'w') as f:
            f.write('')

    def __call__(self, x):
        current_beam_results, current_beam_duration = x
        self.current_beam = current_beam_results

        transcript = ' '.join(self.all_beams.split() + self.current_beam.split())
        self.save_transcript(transcript)

        if current_beam_duration > 10:
            self.all_beams = transcript

    def save_transcript(self, transcript):
        with open(self.file_name, 'w+') as f:
            print(transcript)
            f.write(self.correct(transcript))

    def correct(self, text):
        blob = TextBlob(text)
        corrected_text = blob.correct()

        # blob to str
        return ' '.join(corrected_text.split())


class Listener:

    def __init__(self, sample_rate=8000):
        self.chunk = 1024
        self.sample_rate = sample_rate
        self.pyaudio = pyaudio.PyAudio()
        self.stream = self.pyaudio.open(format=pyaudio.paInt16,
                                        channels=1,
                                        rate=self.sample_rate,
                                        input=True,
                                        output=True,
                                        frames_per_buffer=self.chunk)

    def listen(self, audio_queue):
        while True:
            data = self.stream.read(self.chunk, exception_on_overflow=False)
            audio_queue.append(data)
            time.sleep(0.01)

    def run(self, audio_queue):
        thread = threading.Thread(target=self.listen, args=(audio_queue,), daemon=True)
        thread.start()


class SpeechRecognitionEngine:

    def __init__(self, asr_model_file, context_length=10):
        self.listener = Listener(sample_rate=8000)
        self.feature_extractor = get_feature_extractor(sample_rate=8000)

        self.model = torch.jit.load(asr_model_file)
        self.model.eval().to('cpu')

        self.audio_queue = list()
        self.hidden = (torch.zeros(1, 1, 1024), torch.zeros(1, 1, 1024))
        self.out_args = None

        self.decoder = CTCBeamDecoder(beam_size=100)

        self.context_length = context_length * 50

    def save_audio(self, waveforms, file_name='audio/audio_tmp.wav'):
        wf = wave.open(file_name, 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(self.listener.pyaudio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(8000)
        wf.writeframes(b''.join(waveforms))
        wf.close()

        return file_name

    def predict(self, audio):
        with torch.no_grad():
            file_name = self.save_audio(audio)
            waveform, _ = torchaudio.load(file_name)
            log_mel_spec = self.feature_extractor(waveform).unsqueeze(1)

            out, self.hidden = self.model(log_mel_spec, self.hidden)
            out = torch.nn.functional.softmax(out, dim=2)
            out = out.transpose(0, 1)

            self.out_args = out if self.out_args is None else torch.cat((self.out_args, out), dim=1)
            results = self.decoder(self.out_args)

            current_context_length = self.out_args.shape[1] / 50

            if self.out_args.shape[1] > self.context_length:
                self.out_args = None

            return results, current_context_length

    def prediction_loop(self, writer):
        while True:
            if len(self.audio_queue) < 5:
                continue
            else:
                prediction_queue = self.audio_queue.copy()
                self.audio_queue.clear()
                writer(self.predict(prediction_queue))
            time.sleep(0.05)

    def run(self, writer):
        self.listener.run(self.audio_queue)
        thread = threading.Thread(target=self.prediction_loop, args=(writer,), daemon=True)
        thread.start()
