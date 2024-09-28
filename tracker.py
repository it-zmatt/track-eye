
import cv2
print(cv2.__version__)


# Load the Haar cascades for face and eye detection these are OpenCV trained models to detect face and eyes
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# Start video capture from the webcam
cap = cv2.VideoCapture(0)

while True:

    # Capture frame-by-frame from the webcam
    # ret:  is a boolean to check whether capturing a video is working or no
    # frame: a numpy array representation of the frame
    ret, frame = cap.read()

    frame_roi = frame[150:900, 400:1500]

    # Convert the frame to grayscale (Haar cascades work better on grayscale images)
    # To make it easier detecting face and do operations
    gray = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Loop through each detected face
    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame_roi, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Define the region of interest (ROI) for eyes (within the face)
        roi_gray = gray[y:y+h//2, x:x+w]
        roi_color = frame_roi[y:y+h, x:x+w]

        # Detect eyes within the face ROI
        eyes = eye_cascade.detectMultiScale(roi_gray)

        # Loop through each detected eye
        for (ex, ey, ew, eh) in eyes:
            # Draw a rectangle around the eye
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

    # Display the frame with the detected face and eyes
    cv2.imshow('Eye Tracking', frame_roi)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
