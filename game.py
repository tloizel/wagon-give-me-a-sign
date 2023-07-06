import random
import time

def random_letter():
    alphabet = 'ABC'
    # alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

    random_letter = random.choice(alphabet)
    return random_letter


def translate_words(pred):
    if pred == "love":
        return 'Thank you LeWagon'
    if pred == "fuck":
        return "Let's not be rude"
    if pred == "space":
        return "No letter"
    if pred == "back":
        return "No letter"
    else:
        return pred



if __name__ == "__main__":
    random_letter()
