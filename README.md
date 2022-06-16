# Dialog Helper

### Приложение-помощник для ведения диалога на иностранном языке

На данный момент релизованы подсказки на английском языке. Перевод осуществляется на русский язык.

## Запуск приложения

Для запуска приложения локально необходимо:

### Создать и активировать виртуальное окружение

1. `python3 -m venv dialog_helper.venv`
2. `source dialog_helper.venv/bin/activate`

### Установить зависимости

3. `pip install .`

### Запустиить Django проект

4. `python3 manage.py runserver`

### Открыть приложение на локальном хосте

5. В строке браузера введите `http://127.0.0.1:8000` или `http://localhost:8000/`

## Обучение модели

Для обучения модели распознавания речи самостоятельно необходимо:

### Создать и активировать виртуальное окружение

1. `python3 -m venv dialog_helper.venv`
2. `source dialog_helper.venv/bin/activate`

### Установить зависимости

3. `pip install .`

### Скачать и подготовить к работе датасет

4. Датасет Common Voice можно скачать по ссылке `https://commonvoice.mozilla.org/datasets`
5. `cd asr_model`
6. `python3 make_dataset.py --file_path 'file/path/to/dataset/train.tsv' --save_path 'dataset' --json_name 'train' --downsample_to 8000`
7. `python3 make_dataset.py --file_path 'file/path/to/dataset/test.tsv' --save_path 'dataset' --json_name 'valid' --downsample_to 8000`

**<u>ЗАМЕЧАНИЕ</u>: Скрипт `make_dataset.py` предназначен для работы с датасетом Common Voice от Mozilla. Для того,
чтобы обучать модель на другом датасете необходим следующий формат датасета: аудиофайлы храняться по пути
asr_model/dataset/clips в формате WAV, транскрипции храняться в двух файлах asr_model/dataset/valid.json и
asr_model/dataset/train.json в виде JSON-ов
формата `{"key": "dataset/clips/audio_file_name.wav", "text": "full transcript of audio_file_name.wav"}`.**

### Запустить обучающий скрипт

8. `python3 train.py`

**<u>ЗАМЕЧАНИЕ</u>: После каждой эпохи обучения все параметры модели будут сохраняться в папке asr_model/best_models.
Чтобы запустить обучение с чекпоинта, необходимо указать путь к чекпоинту в параметре
--model_path `python3 train.py --model_path 'best_models/asr_model-epoch={epoch_number}.ckpt'`**

### Оптимизировать чекпоинт и сохранить его, как модель с помощью скрипта save.py

9. `python3 save.py --model_path 'best_models/asr_model-epoch={epoch_number}.ckpt' --save_to '../recognition/model2.zip'`