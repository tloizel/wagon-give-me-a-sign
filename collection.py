import os
import cv2
import string
import uuid


def collection():
    """
    Collecter de la donnée depuis la caméra de l'ordinateur
    """
    DATA_DIR = './raw_data'
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    alphabet = list(string.ascii_lowercase)
    alphabet.extend(['space', 'back', 'love', 'fuck'])

    dataset_size = 1000 #each of us should do 100, 400 in total per letter

    cap = cv2.VideoCapture(0)
    for j in alphabet:
        if not os.path.exists(os.path.join(DATA_DIR, str(j))):
            os.makedirs(os.path.join(DATA_DIR, str(j)))

        print('Collecting data for class {}'.format(j))

        while True:
            ret, frame = cap.read()
            cv2.putText(frame, f'Ready for << {j} >> ? Press "N" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                        cv2.LINE_AA)
            cv2.imshow('frame', frame)
            if cv2.waitKey(25) == ord('n'):
                break

        counter = 0
        while counter < dataset_size:
            ret, frame = cap.read()
            cv2.imshow('frame', frame)
            cv2.waitKey(25)
            cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(str(uuid.uuid1()))), frame)

            counter += 1

    cap.release()
    cv2.destroyAllWindows()
    pass

if __name__ == "__main__":
    collection()
