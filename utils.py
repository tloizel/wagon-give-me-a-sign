import os
import shutil
import glob


def delete_local_enchantillon():
    """
    Suppression de l'échantillon stocké en local
    """
    files = glob.glob('./raw_data')
    for f in files:
        if os.path.isfile(f):
            os.remove(f)
        elif os.path.isdir(f):
            shutil.rmtree(f)
    pass
