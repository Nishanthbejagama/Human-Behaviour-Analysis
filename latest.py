import cv2
import dlib
import mediapipe as mp
import numpy as np
from scipy.spatial import distance
import os

# Initialize MediaPipe Face Mesh (for yawning & face tracking)
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Load Dlib's face detector & shape predictor
print("Loading dlib model...")
predictor_path = "shape_predictor_68_face_landmarks.dat"

if not os.path.exists(predictor_path):
    print(f"Error: {predictor_path} not found. Download it from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
    exit()

try:
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    print("Dlib predictor loaded successfully.")
except Exception as e:
    print(f"Error loading Dlib model: {e}")
    exit()

# Function to calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Function to detect yawn (Mouth Aspect Ratio - MAR)
def is_yawning(mouth_points, landmarks):
    try:
        top_lip = landmarks[mouth_points[0]]
        bottom_lip = landmarks[mouth_points[1]]
        mouth_width_left = landmarks[mouth_points[2]]
        mouth_width_right = landmarks[mouth_points[3]]
        
        mouth_height = np.linalg.norm(np.array([top_lip.x - bottom_lip.x, top_lip.y - bottom_lip.y]))
        mouth_width = np.linalg.norm(np.array([mouth_width_left.x - mouth_width_right.x, mouth_width_left.y - mouth_width_right.y]))
        
        mar = mouth_height / mouth_width
        return mar > 0.5  
    except:
        return False  

# Indices for left and right eye in dlib's 68-landmark model
LEFT_EYE = list(range(42, 48))
RIGHT_EYE = list(range(36, 42))

# EAR thresholds and counters
EAR_THRESHOLD = 0.25  
CONSEC_FRAMES = 3      
blink_counter = 0
closed_frames = 0
eye_closed = False

# Load video file
video_path = "input.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video file: {video_path}")
    exit()
else:
    print("Video file loaded successfully.")

# Initialize MediaPipe Face Mesh
with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
    yawn_count = 0
    face_angle_change_count = 0
    prev_face_angle = None

    print("Starting video processing...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame. Stopping video processing.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Dlib face detection
        faces = detector(gray)

        for face in faces:
            landmarks = predictor(gray, face)
            
            # Extract eye landmarks
            left_eye = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in LEFT_EYE])
            right_eye = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in RIGHT_EYE])

            # Compute EAR
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            avg_ear = (left_ear + right_ear) / 2.0

            # Detect eye closure
            if avg_ear < EAR_THRESHOLD:
                closed_frames += 1
                if not eye_closed and closed_frames >= CONSEC_FRAMES:
                    blink_counter += 1
                    eye_closed = True
            else:
                closed_frames = 0
                eye_closed = False

            # Draw eye landmarks
            for (x, y) in np.concatenate((left_eye, right_eye)):
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        # MediaPipe Face Mesh for yawning & face movement tracking
        results = face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
            for landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(frame, landmarks, mp_face_mesh.FACEMESH_CONTOURS)

                # Get mouth landmarks for yawning detection
                mouth_points = [13, 14, 78, 308]
                if is_yawning(mouth_points, landmarks.landmark):
                    yawn_count += 1

                # Face angle tracking using nose, chin, and eye positions
                nose = landmarks.landmark[1]
                chin = landmarks.landmark[152]

                # Calculate face angle (simplified as a 2D angle between nose and chin)
                current_face_angle = np.arctan2(chin.y - nose.y, chin.x - nose.x)

                # Check for significant face angle change
                if prev_face_angle is not None and abs(current_face_angle - prev_face_angle) > 0.1:
                    face_angle_change_count += 1

                prev_face_angle = current_face_angle

        # Display information
        cv2.putText(frame, f"Eye Closures: {blink_counter}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Yawns: {yawn_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Face Angle Changes: {face_angle_change_count}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display the frame
        cv2.imshow("Fatigue Detection", frame)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Display final counts
    print(f"Total Eye Closures: {blink_counter}")
    print(f"Total Yawns: {yawn_count}")
    print(f"Total Face Angle Changes: {face_angle_change_count}")

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("Processing complete.")