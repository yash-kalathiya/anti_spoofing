# import cv2
# import numpy as np
# from tensorflow.keras.models import load_model


# # anti_spoofing_data = np.load('D:\\New folder\\anti_spoofing_data.npz')
# # X, y = anti_spoofing_data['arr_0'], anti_spoofing_data['arr_1']
# # temp = set(y)
# # check_live_label = 0
# # check_spoof_label = 0
# # for i in y: 
# #     if i == 1:
# #         check_live_label += 1
# #     elif i == 0:
# #         check_spoof_label += 1
# # print(f"There are 2 classes including number of live is {check_live_label} and number of spoof is {check_spoof_label}")

# # from sklearn.model_selection import train_test_split
# # import numpy as np

# # X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=42)
# # X_valid, X_test, y_valid, y_test = train_test_split(X_valid, y_valid, test_size=0.5, random_state=42)

# # import tensorflow as tf

# # from tensorflow.keras import datasets, layers, models
# # import matplotlib.pyplot as plt


# # model = models.Sequential()
# # model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
# # model.add(layers.MaxPooling2D((2, 2)))
# # model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# # model.add(layers.MaxPooling2D((2, 2)))
# # model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# # model.add(layers.Flatten())
# # model.add(layers.Dense(64, activation='relu'))
# # model.add(layers.Dense(2,activation='softmax'))

# # model.summary()


# # model.compile(optimizer='adam',
# #               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
# #               metrics=['accuracy'])
# # # X_train, X_test, y_train, y_test
# # history = model.fit(X_train, y_train, epochs=20, 
# #                     validation_data=(X_valid, y_valid))
# # model.save("./antiSpoof.h5") 

# # test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=2)



# import cv2
# import numpy as np
# from tensorflow.keras.models import load_model
# import warnings

# # Ignore all warnings
# warnings.filterwarnings("ignore")

# # Load your model
# model = load_model("/Users/anshdobariya/Downloads/my_model_3_0.h5")

# # Load the pre-trained face detection cascade
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# # Set up a video capture object
# cap = cv2.VideoCapture(0)  # 0 is for the default camera, change it if you have multiple cameras

# while True:
#     # Capture frame-by-frame
#     ret, frame = cap.read()
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#     # Scale down the V (Value) channel to reduce brightness
#     hsv[:, :, 2] = hsv[:, :, 2] * 1
#     # Convert back to BGR color space
#     frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
#     # Convert frame to grayscale for face detection
#     # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     # print(gray.shape)
#     # Detect faces in the grayscale frame
#     faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

#     # Process each detected face
#     for (x, y, w, h) in faces:
#         # Extract the face region from the frame
#         face_roi = frame[y:y+h, x:x+w]

#         # Preprocess the face region (resize, normalize, etc.)
#         processed_face = cv2.resize(face_roi, (128,128))  # Example resizing to match model input size
#         # print(processed_face.shape)
#         # processed_face = processed_face / 255.0  # Example normalization

#         # Make prediction on the face region
#         prediction = model.predict(np.expand_dims(processed_face, axis=0))

#         # Determine label based on prediction
#         labels=['Fake','Real']
#         label=labels[np.argmax(prediction)]
#         # label=labels[0 if prediction > 0.8 else 1]

#         # Display the label on the frame
#         cv2.putText(frame, f"Label: {label} Prediction: {prediction}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Draw rectangle around the face

#     # Display the frame
#     cv2.imshow('Live Prediction', frame)

#     # Break the loop if 'q' is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release the capture
# cap.release()
# cv2.destroyAllWindows()

# import cv2
# import face_recognition
# import os
# import numpy as np

# # Load the face and eye cascade
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# # Directory containing images of known faces
# known_faces_dir = "/Users/anshdobariya/Downloads/d/"

# # Load faces
# known_face_encodings = []
# known_face_names = []
# for file_name in os.listdir(known_faces_dir):
#     img_path = os.path.join(known_faces_dir, file_name)
#     known_image = face_recognition.load_image_file(img_path)
#     known_encoding = face_recognition.face_encodings(known_image)[0]
#     known_face_encodings.append(known_encoding)
#     known_face_names.append(file_name.split('.')[0])

# # webcam
# cap = cv2.VideoCapture(0)

# # eye size changes


# while True:
#     ret, frame = cap.read()

#     if not ret:
#         break

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # Detect faces in the frame
#     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

#     for (x, y, w, h) in faces:
#         roi_gray = gray[y:y + h, x:x + w]        # Cheack if eyes are detected


#         # Use face recognition to check if the face is a known face
#         face_encoding = face_recognition.face_encodings(frame, [(y, x + w, y + h, x)])[0]
#         matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

#         name = "Unknown"
#         if True in matches:
#             first_match_index = matches.index(True)
#             name = known_face_names[first_match_index]

#         # Draw rectangles around the face and eyes
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

#         font = cv2.FONT_HERSHEY_DUPLEX
#         cv2.putText(frame, name, (x + 6, y - 6), font, 0.5, (255, 255, 255), 1)

#     # Display the frame
#     cv2.imshow('Face Recognition', frame)

#     # Break the loop when 'q' is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release the webcam and close all windows
# cap.release()
# cv2.destroyAllWindows()

import cv2
import numpy as np
from tensorflow.keras.models import load_model
import face_recognition
import os

# Load the pre-trained face detection cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load your anti-spoofing model
anti_spoofing_model = load_model("/Users/anshdobariya/Downloads/my_model_3_0.h5")
# anti_spoofing_model = load_model("/Users/anshdobariya/Downloads/my_model_gray_scale_all_data.h5")

# Load faces for face recognition
known_faces_dir = "/Users/anshdobariya/Downloads/images/"
known_face_encodings = []
known_face_names = []

for file_name in os.listdir(known_faces_dir):
    img_path = os.path.join(known_faces_dir, file_name)
    known_image = face_recognition.load_image_file(img_path)
    known_encoding = face_recognition.face_encodings(known_image)[0]
    known_face_encodings.append(known_encoding)
    known_face_names.append(file_name.split('.')[0])

# Set up a video capture object
cap = cv2.VideoCapture(1)  # 0 is for the deif you have multiple cameras

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        break

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Extract the face region from the frame
        face_roi = frame[y:y+h, x:x+w]

        # Preprocess the face region (resize, normalize, etc.) for anti-spoofing model
        processed_face = cv2.resize(face_roi, (128,128))  # Example resizing to match model input size
        prediction = anti_spoofing_model.predict(np.expand_dims(processed_face, axis=0))

        # Determine label based on prediction
        labels=['Fake','Real']
        label=labels[np.argmax(prediction)]

        # Draw rectangle around the face and display anti-spoofing label


        # Use face recognition to check if the face is a known face
        face_encoding = face_recognition.face_encodings(frame, [(y, x + w, y + h, x)])[0]
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        # Draw rectangle around the face and display the recognized name
        font = cv2.FONT_HERSHEY_DUPLEX
        # cv2.putText(frame, name, (x + 6, y - 6), font, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"{label,name}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Draw rectangle around the face
    # Display the frame
    cv2.imshow('Combined Functionality', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()
