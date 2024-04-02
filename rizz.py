import cv2 as cv
import random

# Open the default webcam
cap = cv.VideoCapture(0)

# Load the pre-trained face detection model
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

frame_count = 0
max_training_frames = 200

# Flag to indicate if training is completed
training_completed = False

# Flag to indicate if face recognition model is loaded
model_loaded = False

# Random number to display on the rectangle
random_number = random.randint(50, 100)

while True:
    # Read the frame from the webcam
    ret, frame = cap.read()

    # Increment frame count
    frame_count += 1

    if not training_completed:
        # Convert the frame to grayscale for face detection
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        # Train facial recognition model using the first 200 frames
        if frame_count <= max_training_frames:
            for (x, y, w, h) in faces:
                # Crop the face region
                face_roi = gray[y:y+h, x:x+w]

                # Draw rectangle around the face for training visualization
                cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), thickness=2)

                # Display the frame with training rectangles
                cv.imshow('Training Faces', frame)

        # After 200 frames, mark training as completed and load the face recognition model
        if frame_count == max_training_frames:
            training_completed = True
            print("Training completed.")
            # Load the face recognition model (this step is usually done here, but in this example, we're skipping it)
            # model = load_model()

    else:
        # Use the trained facial recognition model after training is completed
        # Convert the frame to grayscale for face detection
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            # Draw rectangle around the detected face
            cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)
            
            # Draw the random number inside the rectangle
            cv.putText(frame, 'Rizz Level: ' + str(random_number), (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        # Display the frame with rectangles and text
        cv.imshow('Detected Faces', frame)

    # Wait for 'q' key to exit the loop and close the window
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture object and close all windows
cap.release()
cv.destroyAllWindows()