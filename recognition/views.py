from nltk.corpus import stopwords
from rest_framework import views, response

from .engine import SpeechRecognitionEngine, TranscriptWriter
from .utils import load_prediction_model, translator

TRANSCRIPT_FILE_NAME = 'audio/transcript.txt'

gpt2 = load_prediction_model()
stop_words = set(stopwords.words('english'))


def get_transcript():
    with open(TRANSCRIPT_FILE_NAME, 'r') as f:
        transcript = f.read()

    return transcript


def get_predictions(text, word_count=3):
    if not text:
        return ['-' for x in range(word_count)]

    predictions = gpt2.predict_next(text, 100)
    predictions = [word for word in predictions if word not in stop_words and word.isalpha()]

    return list(map(
        lambda word: {'word': word, 'translation': translator(word).lower()},
        predictions[:word_count]),
    )


class Transript(views.APIView):
    def get(self, request):
        return response.Response({
            'transcript': get_transcript(),
        })


class Prediction(views.APIView):
    def get(self, request):
        transcript = get_transcript()
        predictions = get_predictions(transcript)

        return response.Response({
            'predictions': predictions,
        })


class StartRecording(views.APIView):
    def __init__(self):
        super(StartRecording, self).__init__()
        self.asr_engine = SpeechRecognitionEngine('recognition/model.zip')
        self.is_running = False

    def get(self, request):
        if self.is_running:
            return response.Response('already running', status=400)

        with open(TRANSCRIPT_FILE_NAME, 'w') as f:
            f.write('')

        writer = TranscriptWriter(TRANSCRIPT_FILE_NAME)
        self.asr_engine.run(writer)
        self.is_running = True

        return response.Response('success')
