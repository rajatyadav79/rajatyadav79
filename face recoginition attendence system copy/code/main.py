import csv
import numpy as np
import cv2
import face_recognition
from datetime import datetime

from face_recognition import face_encodings

video_capture = cv2.VideoCapture(0)

# load knonw faces
rajat_image = face_recognition.load_image_file("Images/001.png")
rajat_encoding = face_recognition.face_encodings(rajat_image)[0]

Navya_image = face_recognition.load_image_file("Images/004.jpeg")
Navya_encoding = face_recognition.face_encodings(Navya_image)[0]

modi_image = face_recognition.load_image_file("Images/002.png")
modi_encoding = face_recognition.face_encodings(modi_image)[0]

elon_image = face_recognition.load_image_file("Images/003.png")
elon_encoding = face_recognition.face_encodings(elon_image)[0]

Known_face_encodings = [rajat_encoding, Navya_encoding, modi_encoding, elon_encoding ]
Known_face_names = ["Rajat", "Navya", "Modi", "Musk"]

# list of expected students
students = Known_face_names.copy()

face_location = []
face_encoding = []

# Get the current date and time

now = datetime.now()
current_date = now.strftime("%y-%m-%d")

f= open(f"{current_date}.csv", "w+", newline="")
lnwriter = csv.writer(f)

while True:
    _, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Recgonize faces
    face_location = face_recognition.face_locations(rgb_small_frame)
    face_encoding = face_recognition.face_encodings(rgb_small_frame, face_location)

    for face_encoding in face_encoding:
        matches = face_recognition.compare_faces(Known_face_encodings , face_encoding)
        face_distance = face_recognition.face_distance(Known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distance)

        if(matches[best_match_index]):
            name = Known_face_names[best_match_index]

       # add the text if a person is present
        if name in Known_face_names:
            font = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (10,100)
            fontscale = 1.5
            fontcolor = (255,0,0)
            thickness = 3
            lineType = 2
            cv2.putText(frame, name + "Present", bottomLeftCornerOfText, font, fontscale, fontcolor, thickness, lineType)

            if name in students:
                students.remove(name)
                current_time = now.strftime("%H-%M%S")
                lnwriter.writerow([name, current_time])

    cv2.imshow("Attendance", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
f.close()







