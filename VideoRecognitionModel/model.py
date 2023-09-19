import cv2
import face_recognition
import os

known_faces_path = 'Training_images'

known_face_encodings = []
known_face_names = []

# Iterate through each image in the directory
for filename in os.listdir(known_faces_path):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        # Load the image and encode the face if a face is detected
        image = face_recognition.load_image_file(os.path.join(known_faces_path, filename))
        face_encodings = face_recognition.face_encodings(image)
        
        if len(face_encodings) > 0:
            known_face_encodings.append(face_encodings[0])  # Assuming only one face per image
            known_face_names.append(os.path.splitext(filename)[0])
        else:
            print(f"No face found in {filename}. Skipping.")

video_capture = cv2.VideoCapture('Classroom_Clip.mp4')

while True:
    # Read a frame from the video
    ret, frame = video_capture.read()

    # Find face locations and encodings in the current frame
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    # Loop through the detected faces in the current frame
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Compare the face encoding with the known face encodings
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        name = "Unknown"

        if True in matches:
            match_index = matches.index(True)
            name = known_face_names[match_index]

        # Draw a box around the face and label it
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close OpenCV windows
video_capture.release()
cv2.destroyAllWindows()
