import os
import cv2
import uuid


DATA_DIR = './raw_data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 3 #change to 24 for initial alphabet
dataset_size = 100 #each of us should do 100, 400 in total per letter
delay_snapshot = 100


cap = cv2.VideoCapture(0)
for j in range(number_of_classes):
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))

    print('Collecting data for class {}'.format(j))

    done = False
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, f'Ready? Press "c" and sign letter {j}! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(100) == ord('c'):
            break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(delay_snapshot)
        #cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(counter)), frame)
        cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(str(uuid.uuid1()))), frame)

        counter += 1

cap.release()
cv2.destroyAllWindows()
