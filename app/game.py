import random

def random_letter():
    alphabet = 'BC'
    # alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

    random_letter = random.choice(alphabet)
    return random_letter


if __name__ == "__main__":
    random_letter()
