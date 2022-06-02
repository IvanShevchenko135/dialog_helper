if __name__ == '__main__':
    import os

    os.environ['DJANGO_SETTINGS_MODULE'] = 'dialog_helper.settings'

    import django

    django.setup()
