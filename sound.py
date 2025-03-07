import speech_recognition as sr
import pyttsx3
from langdetect import detect, DetectorFactory
import time

# Ensure consistent language detection results
DetectorFactory.seed = 0

# Initialize the recognizer
r = sr.Recognizer()

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Function to log suspicious activity
def log_violation(message):
    with open("proctoring_log.txt", "a") as log_file:
        log_file.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")

# Function to convert text to speech (optional alerts)
def SpeakText(command):
    engine.say(command)
    engine.runAndWait()

# Function to monitor speech during assessment
def monitor_speech(duration=60):  # Set duration as needed
    print("🔍 Proctoring started. Listening for speech...")
    start_time = time.time()

    while (time.time() - start_time) < duration:
        try:
            with sr.Microphone() as source2:
                r.adjust_for_ambient_noise(source2, duration=0.5)
                audio2 = r.listen(source2)

                # Convert speech to text
                MyText = r.recognize_google(audio2)
                MyText = MyText.lower()

                # Detect language
                detected_lang = detect(MyText)

                if detected_lang != "en":
                    warning_message = f"⚠️ Non-English speech detected: {MyText} (Language: {detected_lang})"
                    print(warning_message)
                    log_violation(warning_message)
                else:
                    print(f"🎤 Student said: {MyText}")

        except sr.RequestError as e:
            print(f"⚠️ API Error: {e}")
            log_violation(f"API Error: {e}")

        except sr.UnknownValueError:
            print("🔇 No speech detected or unclear audio.")

        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            log_violation(f"Unexpected error: {e}")

    print("📌 Proctoring session ended.")

# Run proctoring for a fixed duration (e.g., 5 minutes = 300 seconds)
monitor_speech(duration=300)

