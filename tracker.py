import os
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
import seaborn as sns  # Import Seaborn for styling

# Apply Seaborn theme with minimal style and fall colors
sns.set(style="whitegrid", palette="muted")


def plot_pie_chart(on_screen_time, off_screen_time):
    # Prepare data for the pie chart
    labels = ['On Screen', 'Off Screen']
    sizes = [on_screen_time, off_screen_time]
    colors = ['#F6D0C9', '#DEDDE6']

    # Create a pie chart
    plt.figure(figsize=(8, 6))
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140,
            wedgeprops={'linewidth': 1, 'edgecolor': 'white'})
    plt.axis('equal')  # Equal aspect ratio ensures the pie chart is circular.
    plt.title('Face Detection Time Distribution', fontsize=16, fontweight='bold')

    # Remove grid and make the chart minimal
    plt.gca().set_facecolor('white')
    plt.gca().spines['top'].set_color('none')
    plt.gca().spines['right'].set_color('none')

    # Show the pie chart
    plt.show()


def main():
    # Open the capture
    directory = os.path.dirname(__file__)
    capture = cv2.VideoCapture(0)  # Camera
    if not capture.isOpened():
        print("Camera not opened.")
        return

    Value = 3

    # Load the model
    weights = os.path.join(directory, "face_detection_yunet_2023mar.onnx")
    face_detector = cv2.FaceDetectorYN_create(weights, "", (0, 0))

    # Initialize timers for on-screen and off-screen states
    on_screen_time = 0  # Time in seconds the face was on screen
    off_screen_time = 0  # Time in seconds the face was off screen

    # Initialize previous state to None
    previous_state = None

    # Time counter for the last update
    last_update_time = time.time()

    try:
        while True:
            start_time = time.time()
            duration = 5  # Duration to check gaze direction in seconds

            # To track the last states for S and A
            last_states = []

            while True:
                # Capture the frame and load the image
                result, image = capture.read()
                if not result:
                    print("Failed to capture image.")
                    break

                # Convert image to 3-channel if it is not already
                channels = 1 if len(image.shape) == 2 else image.shape[2]
                if channels == 1:
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                if channels == 4:
                    image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

                # Set the input size
                height, width, _ = image.shape
                face_detector.setInputSize((width, height))

                # Detect faces
                _, faces = face_detector.detect(image)
                faces = faces if faces is not None else []

                closest_face = None
                closest_y = float('inf')  # Start with a large value

                # Find the closest face
                for face in faces:
                    box = list(map(int, face[:4]))
                    y_coordinate = box[1]  # Get the y-coordinate of the bounding box

                    if y_coordinate < closest_y:
                        closest_y = y_coordinate
                        closest_face = face

                if closest_face is not None:
                    box = list(map(int, closest_face[:4]))
                    color = (0, 0, 255)
                    thickness = 2
                    cv2.rectangle(image, box, color, thickness, cv2.LINE_AA)

                    # Landmarks (right eye, left eye, nose)
                    landmarks = list(map(int, closest_face[4:len(closest_face) - 1]))
                    landmarks = np.array_split(landmarks, len(landmarks) / 2)
                    right_eye = landmarks[0]
                    left_eye = landmarks[1]
                    nose = landmarks[2]  # Assuming the nose is the third landmark

                    # Draw eyes and nose
                    cv2.circle(image, tuple(right_eye), 5, (0, 0, 255), thickness, cv2.LINE_AA)
                    cv2.circle(image, tuple(left_eye), 5, (0, 0, 255), thickness, cv2.LINE_AA)
                    cv2.circle(image, tuple(nose), 5, (255, 0, 0), thickness, cv2.LINE_AA)

                    # Check if user is looking straight or away
                    if abs(right_eye[1] - left_eye[1]) < 10 and abs(nose[0] - ((right_eye[0] + left_eye[0]) // 2)) < 10:
                        current_state = 'S'  # Look straight
                    else:
                        current_state = 'A'  # Look away

                    last_states.append(current_state)

                    if len(last_states) > 5:
                        last_states.pop(0)

                    # Update the timer based on the current state
                    current_time = time.time()
                    time_elapsed = current_time - last_update_time

                    if current_state == 'S':
                        on_screen_time += time_elapsed
                    elif current_state == 'A':
                        off_screen_time += time_elapsed

                    last_update_time = current_time
                    previous_state = current_state

                elapsed_time = time.time() - start_time
                if elapsed_time >= duration:
                    if len(last_states) == 5 and all(state == 'S' for state in last_states):
                        Value = 1
                        print("Look straight", Value)

                    elif len(last_states) == 5 and all(state == 'A' for state in last_states):
                        Value = 0
                        print("Look away", Value)

                    last_states = []
                    start_time = time.time()

                cv2.imshow("face detection", image)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

    except KeyboardInterrupt:
        plot_pie_chart(on_screen_time, off_screen_time)
        print(f"Face has been off screen for: {time.strftime('%H:%M:%S', time.gmtime(off_screen_time))}")
        print(f"Face has been on screen for: {time.strftime('%H:%M:%S', time.gmtime(on_screen_time))}")
        print("Program interrupted.")
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
