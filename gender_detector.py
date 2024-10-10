import cv2
import dlib
import os

# Load Gender Detection Model
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

# Load gender model
gender_Net = cv2.dnn.readNet(genderModel, genderProto)

# Gender categories
genderList = ['Male', 'Female']

# Model mean values required for preprocessing
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

# Initialize webcam
cap = cv2.VideoCapture(0)  # 0 is the default webcam, change if you have multiple cameras

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Model for face detection
face_detector = dlib.get_frontal_face_detector()

# Function for face alignment (optional, can improve results)
def align_face(face_img):
    predictor_path = "shape_predictor_68_face_landmarks.dat"

    if not os.path.isfile(predictor_path):
        print(f"Error: {predictor_path} not found.")
        return face_img  # Return unaligned face if loading fails

    try:
        predictor = dlib.shape_predictor(predictor_path)
    except Exception as e:
        print(f"Error loading shape predictor: {e}")
        return face_img  # Return unaligned face if loading fails

    gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    rects = face_detector(gray_face, 1)

    if len(rects) > 0:
        landmarks = predictor(gray_face, rects[0])
        aligned_face = face_img  # Placeholder for aligned face
        return aligned_face
    return face_img

# Start video stream
while True:
    ret, frame = cap.read()  # Capture frame from webcam
    if not ret:
        print("Failed to grab frame")
        break

    # Convert image to grayscale for face detection
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect Faces
    faces = face_detector(img_gray)

    # If faces are detected
    if faces:
        for face in faces:
            # Get face coordinates
            x, y, x2, y2 = face.left(), face.top(), face.right(), face.bottom()

            # Store the face bounding box
            box = [x, y, x2, y2]

            # Crop the face region
            face_img = frame[box[1]:box[3], box[0]:box[2]]

            # Skip empty faces
            if face_img.size == 0:
                continue

            # Align face (optional)
            aligned_face = align_face(face_img)

            # Preprocess the face for gender prediction
            blob_gender = cv2.dnn.blobFromImage(aligned_face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

            # Gender Prediction
            gender_Net.setInput(blob_gender)
            gender_preds = gender_Net.forward()
            gender = genderList[gender_preds[0].argmax()]

            # Display the predicted gender
            label = f'Gender: {gender}'
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

            # Draw rectangle around the face
            cv2.rectangle(frame, (x, y), (x2, y2), (0, 200, 200), 2)

    # Show the frame with face detection and gender label
    cv2.imshow("Gender Detection", frame)

    # Press 'q' to quit the video stream
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
