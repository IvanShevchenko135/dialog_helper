from argostranslate import translate


class Translator:

    def __init__(self, name_from, name_to):
        self.lang_from = None
        self.lang_to = None

        available_languages = translate.get_installed_languages()

        for lang in available_languages:
            if lang.name == name_from:
                self.lang_from = lang
            if lang.name == name_to:
                self.lang_to = lang

        if not self.lang_from:
            raise Exception('Language %s is not installed!' % name_from)
        if not self.lang_to:
            raise Exception('Language %s is not installed!' % name_to)

        self.translator = self.lang_from.get_translation(self.lang_to)

    def __call__(self, text):
        return self.translator.translate(text)
