import os


def check_dir(_dir, make=True):
    if os.path.exists(_dir):
        return True
    else:
        if make:
            os.makedirs(_dir)
            return True
        return False
