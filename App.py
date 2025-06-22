import cv2
import os
import csv
import smtplib
import mimetypes
from email.message import EmailMessage
import numpy as np
from datetime import datetime
from ultralytics import YOLO
from twilio.rest import Client

# Load the YOLOv8 model
model = YOLO("./models/gear.pt")  # Replace with your trained YOLO model

# Create a face recognizer using LBPH
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + './models/haarcascade_frontalface_default.xml')
# Load employee images from a folder and train the recognizer
def train_face_recognizer(face_folder_path):
    faces = []
    labels = []
    label_map = {}
    current_label = 0

    for filename in os.listdir(face_folder_path):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image = cv2.imread(f'{face_folder_path}/{filename}')
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces_detected = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces_detected:
                face = gray[y:y + h, x:x + w]
                faces.append(face)
                labels.append(current_label)
                label_map[current_label] = filename.split('.')[0]  # Name of the employee

            current_label += 1

    # Train the recognizer with the collected faces and labels
    recognizer.train(faces, np.array(labels))
    return label_map

# Twilio configuration
TWILIO_ACCOUNT_SID = "Account_Sid"  # Replace with your Twilio Account SID
TWILIO_AUTH_TOKEN = "ACC_Auth_token"    # Replace with your Twilio Auth Token
TWILIO_PHONE_NUMBER = "Twilio_num"  # Replace with your Twilio phone number

# Email configuration
EMAIL_ADDRESS = "<Sender_Mail_id>"  # Replace with your email address
EMAIL_PASSWORD = "<Sender_Mail_Password>"    # Replace with your email password
HR_EMAIL = "<Receiver_Mail_id>"       # Replace with HR's email address

# Load employee contacts
def load_employee_contacts():
    contacts = {}
    contacts_file = "employee_contacts.csv"  # CSV file with "Name,PhoneNumber"

    if os.path.exists(contacts_file):
        with open(contacts_file, mode='r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip header
            for row in reader:
                name, phone = row
                contacts[name] = phone
    return contacts

employee_contacts = load_employee_contacts()  # Load employee phone numbers

# Function to send an SMS notification to the employee
def send_sms(name):
    if name in employee_contacts:
        employee_phone = employee_contacts[name]
        try:
            client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
            message = client.messages.create(
                body=f"Safety Violation Alert: {name}, you were detected without safety gear. Please adhere to safety protocols.",
                from_=TWILIO_PHONE_NUMBER,
                to=employee_phone
            )
            print(f"SMS sent to {name} ({employee_phone}). Message SID: {message.sid}")
        except Exception as e:
            print(f"Error sending SMS to {name}: {e}")
    else:
        print(f"Phone number not found for {name}. Cannot send SMS.")

# Function to send an email notification to HR with a screenshot
def send_email(name, screenshot_path):
    try:
        msg = EmailMessage()
        msg['Subject'] = f"Safety Violation Alert: {name}"
        msg['From'] = EMAIL_ADDRESS
        msg['To'] = HR_EMAIL
        msg.set_content(f"A safety violation was detected for {name}.\n\nPlease find the attached screenshot.")

        with open(screenshot_path, 'rb') as f:
            file_data = f.read()
            file_name = os.path.basename(screenshot_path)
            mime_type, _ = mimetypes.guess_type(screenshot_path)
            mime_main, mime_sub = mime_type.split('/')
            msg.add_attachment(file_data, maintype=mime_main, subtype=mime_sub, filename=file_name)

        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            smtp.send_message(msg)

        print(f"Email sent to HR for {name}.")
    except Exception as e:
        print(f"Error sending email: {e}")

# Initialize webcam or CCTV feed
cap = cv2.VideoCapture(0)  # Use '0' for default webcam or use CCTV IP URL (e.g., 'rtsp://your-cctv-url')

# Load the trained recognizer and label map
label_map = train_face_recognizer('employee_faces')

# Prepare the CSV file for logging
csv_file = "violations_log.csv"
if not os.path.exists(csv_file):
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Name", "Timestamp", "Screenshot"])  # Write header if file doesn't exist

# Function to log violations
def log_violation(name, screenshot_path):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([name, timestamp, screenshot_path])

# Safety gear and face recognition
def detect_safety_gear_and_faces():
    logged_names_today = set()

    if not cap.isOpened():
        print("Error: Unable to access the camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to capture frame.")
            break

        # Perform detection using YOLO
        results = model.predict(source=frame, save=False, conf=0.5)

        safety_gear_detected = False

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                confidence = box.conf[0]
                class_id = int(box.cls[0])
                label = model.names[class_id]

                if label in ["helmet", "gloves", "vest"]:
                    safety_gear_detected = True

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {confidence:.2f}",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if not safety_gear_detected:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                face = gray[y:y + h, x:x + w]
                label, confidence = recognizer.predict(face)
                name = label_map.get(label, "Unknown")

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, f"NO GEAR: {name}",
                            (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                if name != "Unknown" and name not in logged_names_today:
                    screenshot_path = f"screenshots/{name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.jpg"
                    cv2.imwrite(screenshot_path, frame)
                    log_violation(name, screenshot_path)
                    send_sms(name)
                    send_email(name, screenshot_path)
                    logged_names_today.add(name)

        cv2.imshow("Safety Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

detect_safety_gear_and_faces()
