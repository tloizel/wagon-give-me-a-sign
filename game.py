import random
import time

def random_letter():
    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

    random_letter = random.choice(alphabet)
    return random_letter


def translate_words(pred):
    if pred == "love":
        return 'Thank you @lewagon'
    if pred == "fuck":
        return "Let's not be rude"
    if pred == "space":
        return "Space"
    if pred == "back":
        return "Delete"
    else:
        return pred



if __name__ == "__main__":
    random_letter()
