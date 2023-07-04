import random
import time

def random_letter():
    alphabet = 'ABC'
    # alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

    random_letter = random.choice(alphabet)
    return random_letter

def countdown_timer(countdown_text, duration):
    start_time = time.time()
    end_time = start_time + duration

    while time.time() < end_time:
        remaining_time = int(end_time - time.time())
        countdown_text.text(f"Countdown: {remaining_time} seconds")
        time.sleep(1)



if __name__ == "__main__":
    random_letter()
