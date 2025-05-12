import cv2
import numpy as np
import os
import json
from datetime import datetime
import time
import winsound  # For alert sounds
import shutil    # For directory operations

# Try to import optional dependencies
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    print("Ultralytics/YOLO not available - object detection disabled")
    YOLO_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    TF_AVAILABLE = True
except ImportError:
    print("TensorFlow not available - emotion detection disabled")
    TF_AVAILABLE = False

class AISurveillanceSystem:
    def __init__(self, known_faces_dir="data/known_faces", database_path="database/records.json",
                 models_dir="models", alert_threshold=80):
        # Face recognition
        self.known_faces_dir = known_faces_dir
        self.database_path = database_path
        self.alert_threshold = alert_threshold
        self.models_dir = models_dir
        self.known_face_encodings = []
        self.known_face_names = []
        
        # Create necessary directories
        os.makedirs(os.path.dirname(self.database_path), exist_ok=True)
        os.makedirs(self.known_faces_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Use LBPH recognizer for face recognition
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create(
            radius=1,
            neighbors=8,
            grid_x=8,
            grid_y=8
        )
        
        # Face detector - try both Haar cascade and DNN
        print("Loading face detection models...")
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        if self.face_cascade.empty():
            print("Warning: Could not load face cascade classifier")
            
        # Initialize emotion detection
        self.emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
        self.emotion_model = None
        self.load_emotion_model()
        
        # Initialize object detection
        self.object_model = None
        if YOLO_AVAILABLE:
            self.load_object_model()
        
        # Load database records
        self.records = self.load_records()
        
        # Check if we have any known faces and load them
        self.model_trained = False
        if os.path.exists(self.known_faces_dir):
            # Check if directory has any content
            has_content = False
            for _, _, files in os.walk(self.known_faces_dir):
                if files:
                    has_content = True
                    break
            
            if has_content:
                success = self.load_team_photos()
                if success:
                    self.model_trained = True
                else:
                    print("Warning: Found face directories but couldn't train model.")
            else:
                print("No face data found. Add a person with --mode add")
    
    def load_emotion_model(self):
        """Load the emotion detection model if available."""
        if not TF_AVAILABLE:
            return
            
        try:
            emotion_model_path = os.path.join(self.models_dir, "emotion_model.h5")
            if os.path.exists(emotion_model_path):
                self.emotion_model = load_model(emotion_model_path, compile=False)
                self.emotion_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
                print("Emotion detection model loaded successfully")
            else:
                print("Emotion model not found at:", emotion_model_path)
        except Exception as e:
            print(f"Error loading emotion model: {e}")
            self.emotion_model = None
    
    def load_object_model(self):
        """Load the YOLO object detection model if available."""
        if not YOLO_AVAILABLE:
            return
            
        try:
            yolo_model_path = os.path.join(self.models_dir, "yolov8n.pt")
            if os.path.exists(yolo_model_path):
                self.object_model = YOLO(yolo_model_path)
                print("Object detection model loaded successfully")
            else:
                print("YOLO model not found at:", yolo_model_path)
                # Try to load the default YOLO model
                try:
                    self.object_model = YOLO("yolov8n.pt")
                    print("Loaded default YOLO model")
                except:
                    print("Could not load default YOLO model - object detection disabled")
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            self.object_model = None
    
    def load_records(self):
        """Load the existing recognition records from the database."""
        try:
            with open(self.database_path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            # If file doesn't exist or is invalid, start with empty records
            return {"recognitions": [], "alerts": []}
    
    def save_records(self):
        """Save the current records to the database file."""
        os.makedirs(os.path.dirname(self.database_path), exist_ok=True)
        with open(self.database_path, 'w') as f:
            json.dump(self.records, f, indent=4)
    
    def detect_faces(self, image):
        """Detect faces in the image."""
        if image is None:
            return [], None
            
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Use more sensitive detection settings
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # If no faces found, try more aggressive parameters
        if len(faces) == 0:
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.05,
                minNeighbors=2,
                minSize=(20, 20),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
        
        return faces, gray
    
    def load_team_photos(self):
        """Load all known face images from the specified directory."""
        if not os.path.exists(self.known_faces_dir):
            print(f"Directory '{self.known_faces_dir}' does not exist.")
            os.makedirs(self.known_faces_dir, exist_ok=True)
            print(f"Created directory '{self.known_faces_dir}'")
            return False
            
        print(f"Loading team photos from {self.known_faces_dir}...")
        
        # Clear existing data
        self.known_face_encodings = []
        self.known_face_names = []
        
        # Process all person subdirectories
        for person_name in os.listdir(self.known_faces_dir):
            person_dir = os.path.join(self.known_faces_dir, person_name)
            if os.path.isdir(person_dir):
                # Process each image file for this person
                face_count = 0
                for img_file in os.listdir(person_dir):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_path = os.path.join(person_dir, img_file)
                        if self._process_known_face(image_path, person_name):
                            face_count += 1
                if face_count > 0:
                    print(f"Loaded {face_count} images for {person_name}")
                else:
                    print(f"Warning: No valid face images found for {person_name}")
        
        # Train the face recognizer with collected data
        if len(self.known_face_encodings) > 0:
            try:
                # Prepare face recognition data
                faces = []
                labels = []
                label_map = {}
                
                for i, (encoding, name) in enumerate(zip(self.known_face_encodings, self.known_face_names)):
                    if name not in label_map:
                        label_map[name] = len(label_map)
                    faces.append(encoding)
                    labels.append(label_map[name])
                
                # Convert labels to numpy array
                label_array = np.array(labels)
                
                # Train the recognizer
                print(f"Training the face recognizer with {len(faces)} faces of {len(set(self.known_face_names))} people")
                self.face_recognizer.train(faces, label_array)
                self.label_to_name = {v: k for k, v in label_map.items()}
                print("Face recognizer trained successfully")
                return True
            except Exception as e:
                print(f"Error training face recognizer: {e}")
                return False
        else:
            print("No face images found in the specified directory.")
            return False
    
    def _process_known_face(self, image_path, person_name):
        """Process a single known face image for training."""
        try:
            image = cv2.imread(image_path)
            if image is None:
                print(f"Warning: Could not read image {image_path}")
                return False
                
            # Detect faces
            faces, gray = self.detect_faces(image)
            
            if len(faces) > 0:
                # Take the largest face found
                largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
                x, y, w, h = largest_face
                
                face = gray[y:y+h, x:x+w]
                
                # Apply histogram equalization
                face = cv2.equalizeHist(face)
                
                # Resize to a standard size
                face = cv2.resize(face, (100, 100))
                
                # Add to our collection
                self.known_face_encodings.append(face)
                self.known_face_names.append(person_name)
                return True
            else:
                print(f"No face detected in {image_path}")
                return False
                
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return False
    
    def recognize_face(self, face_image):
        """Recognize a single face from a grayscale image."""
        try:
            # Check if model is trained
            if not self.model_trained:
                return "Unknown (model not trained)", 0
                
            # Apply histogram equalization
            face = cv2.equalizeHist(face_image)
            
            # Resize to match our training data
            face = cv2.resize(face, (100, 100))
            
            # Predict who this person is
            label, confidence = self.face_recognizer.predict(face)
            
            # Lower confidence is better in this algorithm
            if confidence < self.alert_threshold:
                return self.label_to_name[label], confidence
            else:
                return "Unknown", confidence
                
        except Exception as e:
            print(f"Error during recognition: {e}")
            return "Unknown", 0
    
    def detect_emotion(self, face_image):
        """Detect emotion in a face image."""
        if self.emotion_model is None:
            return "Unknown"
        
        try:
            # Prepare the image
            gray_face = cv2.resize(face_image, (64, 64))
            
            # Add missing dimensions required by the model
            gray_face = gray_face.reshape(1, 64, 64, 1) / 255.0
            
            # Make prediction
            emotion_prediction = self.emotion_model.predict(gray_face, verbose=0)
            emotion_label = self.emotions[np.argmax(emotion_prediction)]
            
            return emotion_label
        except Exception as e:
            print(f"Error detecting emotion: {e}")
            return "Unknown"
    
    def detect_objects(self, frame):
        """Detect objects like weapons and masks in the frame."""
        dangerous_objects = []
        
        if self.object_model is None:
            return dangerous_objects
        
        try:
            # Run object detection
            results = self.object_model.predict(frame, verbose=False)
            
            # Process results
            for result in results:
                for box in result.boxes.data:
                    confidence = float(box[4])
                    cls = int(box[5])
                    label = result.names[cls]
                    
                    # Check for dangerous objects
                    if confidence > 0.5 and label.lower() in ["knife", "gun", "scissors", "mask"]:
                        x1, y1, x2, y2 = map(int, box[:4])
                        dangerous_objects.append({
                            "label": label,
                            "confidence": confidence,
                            "bbox": (x1, y1, x2, y2)
                        })
            
            return dangerous_objects
        except Exception as e:
            print(f"Error detecting objects: {e}")
            return []
    
    def trigger_alert(self, reason):
        """Trigger an alert sound and log the event."""
        print(f"\n⚠️ ALERT TRIGGERED: {reason}")
        
        # Play alert sound
        try:
            winsound.Beep(1000, 500)  # 1000 Hz for 500 ms
        except:
            print("(Alert sound failed - winsound may not be available on this system)")
        
        # Log to database
        timestamp = datetime.now().isoformat()
        if "alerts" not in self.records:
            self.records["alerts"] = []
        
        self.records["alerts"].append({
            "reason": reason,
            "timestamp": timestamp
        })
        
        print("(Alert logged and notification would be sent)")
    
    def process_frame(self, frame):
        """Process a single frame for surveillance."""
        display_frame = frame.copy()
        
        # Step 1: Detect faces
        faces, gray = self.detect_faces(frame)
        
        # Step 2: Process each face
        for (x, y, w, h) in faces:
            # Make detection box slightly larger
            expanded_w = int(w * 1.1)
            expanded_h = int(h * 1.1)
            center_x = x + w // 2
            center_y = y + h // 2
            x1 = max(0, center_x - expanded_w // 2)
            y1 = max(0, center_y - expanded_h // 2)
            x2 = min(frame.shape[1], center_x + expanded_w // 2)
            y2 = min(frame.shape[0], center_y + expanded_h // 2)
            
            # Get face region
            face_roi = frame[y1:y2, x1:x2]
            face_gray = gray[y1:y2, x1:x2]
            
            if face_gray.size == 0:
                continue
            
            # Step 3: Recognize face if model is trained
            name, confidence = self.recognize_face(face_gray)
            
            # Record this recognition
            timestamp = datetime.now().isoformat()
            self.records["recognitions"].append({
                "name": name,
                "confidence": float(confidence),
                "timestamp": timestamp
            })
            
            # Step 4: Draw rectangle and name
            if "not trained" in name:
                color = (255, 165, 0)  # Orange for model not trained
            elif name != "Unknown":
                color = (0, 255, 0)    # Green for known person
            else:
                color = (0, 0, 255)    # Red for unknown
                
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            
            label_text = f"{name}" if "not trained" in name else f"{name} ({confidence:.1f})"
            cv2.putText(display_frame, label_text, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Step 5: Check if unknown face
            if self.model_trained and name == "Unknown":
                self.trigger_alert(f"Unknown person detected")
            
            # Step 6: Detect emotion if model is available
            emotion = self.detect_emotion(face_gray)
            if emotion != "Unknown":
                cv2.putText(display_frame, emotion, (x1, y2 + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # Check for suspicious emotions
                if emotion in ["Angry", "Fear", "Disgust"]:
                    self.trigger_alert(f"Suspicious emotion detected: {emotion}")
        
        # Step 7: Detect objects
        dangerous_objects = self.detect_objects(frame)
        
        # Step 8: Process dangerous objects
        for obj in dangerous_objects:
            label = obj["label"]
            x1, y1, x2, y2 = obj["bbox"]
            
            # Draw bounding box
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(display_frame, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Trigger alert
            self.trigger_alert(f"{label} detected")
        
        # Display status and instructions
        cv2.putText(display_frame, "AI Surveillance Active", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Display model status
        if not self.model_trained:
            cv2.putText(display_frame, "Face recognition model not trained - add faces first", 
                       (10, display_frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        return display_frame
    
    def start_surveillance(self):
        """Start the surveillance system with the webcam."""
        print("\n🚀 Starting AI Surveillance System...")
        
        # Try multiple camera indices to find a working camera
        camera_found = False
        for camera_index in range(3):  # Try indices 0, 1, 2
            cap = cv2.VideoCapture(camera_index)
            if cap.isOpened():
                print(f"Camera found at index {camera_index}")
                camera_found = True
                break
        
        if not camera_found:
            print("Error: Could not open any camera. Tried indices 0-2.")
            return
        
        try:
            frame_count = 0
            last_save_time = time.time()
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame, retrying...")
                    # Try to reconnect
                    cap.release()
                    time.sleep(1)
                    cap = cv2.VideoCapture(camera_index)
                    continue
                
                frame_count += 1
                current_time = time.time()
                
                # Process frames (skip some for performance)
                if frame_count % 2 == 0:
                    display = self.process_frame(frame)
                    
                    # Display the result
                    cv2.imshow('AI Surveillance System', display)
                
                # Save records periodically (every 30 seconds)
                if current_time - last_save_time > 30:
                    self.save_records()
                    last_save_time = current_time
                
                # Break loop on 'q' key press
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                    
        except Exception as e:
            print(f"Error in surveillance: {e}")
        finally:
            # Save records and clean up
            self.save_records()
            cap.release()
            cv2.destroyAllWindows()
            print("Surveillance stopped.")
    
    def add_new_person(self, person_name, capture_count=5):
        """Add a new person by capturing their face from the camera."""
        person_name = person_name.strip()
        if not person_name:
            print("Error: Person name cannot be empty")
            return False
            
        # Create directory for this person
        person_dir = os.path.join(self.known_faces_dir, person_name)
        if os.path.exists(person_dir):
            choice = input(f"Person '{person_name}' already exists. Replace existing images? (y/n): ")
            if choice.lower() == 'y':
                shutil.rmtree(person_dir)
            else:
                print("Operation cancelled")
                return False
                
        os.makedirs(person_dir, exist_ok=True)
        
        print(f"Adding new person: {person_name}")
        print(f"Will capture {capture_count} images.")
        print("Instructions:")
        print("  - Position your face in front of the camera")
        print("  - Press SPACE to capture an image when ready")
        print("  - Press ESC to cancel")
        print("  - We'll capture multiple images to improve recognition")
        
        # Try multiple camera indices
        camera_found = False
        for camera_index in range(3):  # Try indices 0, 1, 2
            cap = cv2.VideoCapture(camera_index)
            if cap.isOpened():
                print(f"Camera found at index {camera_index}")
                camera_found = True
                break
        
        if not camera_found:
            print("Error: Could not open any camera.")
            return False
            
        captured = 0
        try:
            while captured < capture_count:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame, retrying...")
                    cap.release()
                    time.sleep(1)
                    cap = cv2.VideoCapture(camera_index)
                    continue
                    
                # Get a copy for display
                display = frame.copy()
                
                # Detect faces
                faces, _ = self.detect_faces(frame)
                
                # Draw rectangles around faces
                if len(faces) > 0:
                    face_found = True
                    for (x, y, w, h) in faces:
                        cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 0), 2)
                else:
                    face_found = False
                    # Show warning
                    cv2.putText(display, "No face detected! Move closer to camera", 
                               (10, display.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                # Show instructions and status
                cv2.putText(display, f"Captured: {captured}/{capture_count}", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display, "Press SPACE to capture", 
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display, "Press ESC to cancel", 
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            
                # Display
                cv2.imshow('Capture Face: ' + person_name, display)
                
                # Capture on SPACE, quit on ESC
                key = cv2.waitKey(1) & 0xFF
                if key == ord(' ') and face_found:
                    # Find the largest face
                    largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
                    x, y, w, h = largest_face
                    
                    # Add a bit of margin around the face (20%)
                    margin_x = int(w * 0.2)
                    margin_y = int(h * 0.2)
                    x1 = max(0, x - margin_x)
                    y1 = max(0, y - margin_y)
                    x2 = min(frame.shape[1], x + w + margin_x)
                    y2 = min(frame.shape[0], y + h + margin_y)
                    
                    face_img = frame[y1:y2, x1:x2]
                    
                    # Save the image
                    img_path = os.path.join(person_dir, f"{person_name}_{captured+1}.jpg")
                    cv2.imwrite(img_path, face_img)
                    
                    captured += 1
                    print(f"Captured image {captured}/{capture_count}")
                    
                    # Show a brief "captured" indicator
                    cap_display = frame.copy()
                    cv2.putText(cap_display, "Image Captured!", (center_x-100, center_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                    cv2.imshow('Capture Face: ' + person_name, cap_display)
                    cv2.waitKey(500)  # Show for 500ms
                    
                elif key == 27:  # ESC key
                    print("Face capture cancelled")
                    break
                    
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
        # If we captured any images, reload the known faces
        if captured > 0:
            print(f"\nSuccessfully added {captured} images for {person_name}")
            print("Retraining face recognition model...")
            if self.load_team_photos():
                self.model_trained = True
                print("Model successfully trained with the new face!")
                return True
            else:
                print("Warning: Failed to train model with the new face.")
                return False
        else:
            print("No images captured.")
            return False
    
    def list_known_people(self):
        """List all known people in the database."""
        if not os.path.exists(self.known_faces_dir):
            print("No face database exists yet.")
            return set()
            
        # Get list of people by directory names
        people = []
        for item in os.listdir(self.known_faces_dir):
            person_dir = os.path.join(self.known_faces_dir, item)
            if os.path.isdir(person_dir):
                # Check if directory has face images
                has_images = False
                for file in os.listdir(person_dir):
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        has_images = True
                        break
                
                if has_images:
                    people.append(item)
        
        if people:
            print(f"Known people ({len(people)}):")
            for person in sorted(people):
                print(f"- {person}")
        else:
            print("No people in the database yet. Add someone with --mode add")
            
        return set(people)
    
    def search_records(self, name=None, date=None):
        """Search recognition records by name or date."""
        results = self.records["recognitions"]
        
        if name:
            results = [r for r in results if r["name"].lower() == name.lower()]
            
        if date:
            # Assume date is in format YYYY-MM-DD
            results = [r for r in results if r["timestamp"].startswith(date)]
            
        print(f"Found {len(results)} records:")
        for record in results:
            print(f"{record['timestamp']}: {record['name']} (confidence: {record['confidence']:.1f})")
            
        return results

# Helper function to guide first-time setup
def setup_wizard():
    """Guide the user through setting up the system."""
    print("\n======================================")
    print("🔧 AI Surveillance System Setup Wizard 🔧")
    print("======================================\n")
    
    print("This wizard will help you set up the AI surveillance system.")
    print("First, let's add your face to the system for recognition.\n")
    
    name = input("Enter your name: ")
    if not name.strip():
        print("Error: Name cannot be empty")
        return False
        
    system = AISurveillanceSystem()
    
    # Add the person
    print(f"\nGreat! Now let's capture some images of {name}'s face.")
    success = system.add_new_person(name)
    
    if success:
        print("\n✅ Setup complete! Your face has been added to the system.")
        print("You can now run the surveillance system with:")
        print("  python surveillance.py --mode surveillance")
        return True
    else:
        print("\n❌ Setup failed. Please try again.")
        return False

# Main entry point
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='AI Surveillance System')
    parser.add_argument('--mode', choices=['surveillance', 'add', 'list', 'search', 'setup'], 
                        default='surveillance', help='Operation mode')
    parser.add_argument('--name', help='Person name for add or search mode')
    parser.add_argument('--date', help='Date for search (YYYY-MM-DD)')
    parser.add_argument('--known-faces-dir', default='data/known_faces', 
                       help='Directory containing known face photos')
    parser.add_argument('--models-dir', default='models',
                       help='Directory containing AI models')
    parser.add_argument('--capture-count', type=int, default=5,
                       help='Number of images to capture when adding a face')
    parser.add_argument('--threshold', type=float, default=80.0,
                       help='Recognition confidence threshold')
    
    args = parser.parse_args()
    
    # Setup wizard for first-time users
    if args.mode == 'setup':
        setup_wizard()
    else:
        # Create the system with specified parameters
        system = AISurveillanceSystem(
            known_faces_dir=args.known_faces_dir,
            models_dir=args.models_dir,
            alert_threshold=args.threshold
        )
        
        # Run in selected mode
        if args.mode == 'surveillance':
            if system.model_trained:
                system.start_surveillance()
            else:
                print("\nNo trained face recognition model found.")
                choice = input("Would you like to add your face first? (y/n): ")
                if choice.lower() == 'y':
                    name = input("Enter your name: ")
                    if system.add_new_person(name, args.capture_count):
                        system.start_surveillance()
                else:
                    print("Running surveillance with face recognition disabled.")