import nltk
import urllib
import os

import argostranslate.package
import argostranslate.translate
import argostranslate.argospm

from setuptools import setup, find_packages


def create_dataset_dir():
    if not os.path.exists('asr_model/dataset/'):
        os.makedirs('asr_model/dataset/')

    if not os.path.exists('asr_model/dataset/clips'):
        os.makedirs('asr_model/dataset/clips')


def download_model(url, path):
    if not os.path.isfile(path):
        urllib.request.urlretrieve(url, path)


def download_langs_for_translator():
    argostranslate.argospm.update_index('')

    from_code = "en"
    to_code = "ru"

    # Download and install Argos Translate package
    available_packages = argostranslate.package.get_available_packages()

    available_package = list(
        filter(
            lambda x: x.from_code == from_code and x.to_code == to_code, available_packages
        )
    )[0]
    download_path = available_package.download()
    argostranslate.package.install_from_path(download_path)


create_dataset_dir()

nltk.download('stopwords')

download_model('https://github.com/IvanShevchenko135/dialog_helper/files/8935583/model.zip', 'recognition/model2.zip')

download_langs_for_translator()

with open('requirements.txt') as f:
    required_packages = f.read().splitlines()

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='dialog_helper',
    version='0.0.1',
    author='Ivan Shevchenko',
    author_email='ivanshevchenko135@gmail.com',
    description='Assistant app for conducting a dialogue in a foreign language',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/IvanShevchenko135/dialog_helper',
    license='MIT',
    packages=find_packages(),
    install_requires=required_packages,
)
