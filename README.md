Here’s the updated `README.md` content with the new title:

---

# Safety-Gear-Violation-Detection-using-YOLOv8-And-Haarcascade

## 📁 Steps to Run the Program

### Step 1: Extract the Zip File

Extract the downloaded zip file to a location of your choice.

### Step 2: Add Employee Images

Inside the extracted folder, create a new folder named `employee_faces` and store the **images of the employees** in it.

> 📷 Use clear frontal photos for better facial recognition accuracy.

### Step 3: Setup Screenshots and Employee Data

* Create a **new folder** to store screenshots.
* Create a **CSV file** that contains employee names and their mobile numbers in the format:

  ```
  Name,MobileNumber
  John Doe,+919876543210
  ```

### Step 4: Configure Mail and Twilio Credentials

Open the code file and **edit the following variables** with your details:

```python
TWILIO_ACCOUNT_SID = "your_twilio_account_sid"
TWILIO_AUTH_TOKEN = "your_twilio_auth_token"
TWILIO_PHONE_NUMBER = "+your_twilio_phone_number"

EMAIL_ADDRESS = "your_email@gmail.com"
EMAIL_PASSWORD = "your_app_password"
HR_EMAIL = "hr_email@gmail.com"
```

> 🔐 Make sure your email allows **app passwords** or **less secure app access**.

### Step 5: Open Command Prompt

Open **Command Prompt** and navigate to the extracted folder using the `cd` command:

```bash
cd path\to\your\extracted\folder
```

### Step 6: Install Required Python Packages

Run the following command to install all dependencies:

```bash
pip install -r requirements.txt
```

### Step 7: Run the Application

Once all dependencies are installed, start the application using:

```bash
python App.py
```

---

## ✅ You're All Set!

Make sure you have a webcam connected and internet access for email and SMS features to work properly.

---

Let me know if you'd like to add more sections like **project description**, **tech stack**, or **sample outputs**.
