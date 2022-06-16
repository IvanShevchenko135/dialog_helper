from setuptools import setup, find_packages

with open("requirements.txt") as f:
    required_packages = f.read().splitlines()

with open("README.md") as f:
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
