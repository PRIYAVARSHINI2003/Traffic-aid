import numpy as np
import cv2
import pickle
import pyttsx3

# Initialize the text-to-speech engine
engine = pyttsx3.init()

frameWidth = 32  # Resize directly to the model input size
frameHeight = 32
brightness = 180
threshold = 0.90  # PROBABILITY THRESHOLD
font = cv2.FONT_HERSHEY_SIMPLEX

# SETUP THE VIDEO CAMERA
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, brightness)

# IMPORT THE TRAINED MODEL
with open("model_trained.p", "rb") as f:
    model = pickle.load(f)


def getClassName(classNo):
    class_names = {
        0: 'Speed Limit 20 km/h',
        1: 'Speed Limit 20 km/h',
        2: 'Speed Limit 20 km/h',
        13: 'Stop',
        14: 'Stop Sign',
        28: 'Children Crossing',
        29: 'No Crossing'  # Assuming classNo 29 is for 'No Crossing'
        # Add more class names as needed
    }
    class_name = class_names.get(classNo, 'Stop')
    engine.say(class_name)
    engine.runAndWait()
    return class_name


while True:
    # READ IMAGE
    ret, imgOriginal = cap.read()
    if not ret:
        break

    # PROCESS IMAGE
    img = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255.0  # Normalize
    img = cv2.resize(img, (frameWidth, frameHeight))
    img = np.expand_dims(img, axis=0)

    # PREDICT IMAGE
    predictions = model.predict(img)
    classIndex = np.argmax(predictions)
    probabilityValue = np.amax(predictions)

    if probabilityValue > threshold:
        className = getClassName(classIndex)
        cv2.putText(imgOriginal, "CLASS: " + str(className), (20, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(imgOriginal, "PROBABILITY: " + str(round(probabilityValue * 100, 2)) + "%", (20, 75), font, 0.75,
                    (0, 0, 255), 2, cv2.LINE_AA)

    # Display the processed frame
    cv2.imshow("Result", imgOriginal)

    # Wait for 'q' key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
