import cv2
import dlib
import mediapipe as mp
import numpy as np
import sys
import os
import subprocess
from scipy.spatial import distance

# Initialize MediaPipe Face Mesh
drawing_utils = mp.solutions.drawing_utils
face_mesh = mp.solutions.face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Load Dlib's face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

LEFT_EYE = list(range(42, 48))
RIGHT_EYE = list(range(36, 42))
EAR_THRESHOLD = 0.25  
CONSEC_FRAMES = 3

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def is_yawning(landmarks):
    try:
        top_lip = landmarks[13]
        bottom_lip = landmarks[14]
        left_corner = landmarks[78]
        right_corner = landmarks[308]
        mouth_height = np.linalg.norm([top_lip.y - bottom_lip.y])
        mouth_width = np.linalg.norm([left_corner.x - right_corner.x])
        mar = mouth_height / mouth_width
        return mar > 0.5
    except:
        return False  

def analyze_fatigue(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file", file=sys.stderr)
        return

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.join("static", "uploads", "outputs", video_name)
    os.makedirs(output_dir, exist_ok=True)

    blink_counter, closed_frames, yawn_count, face_angle_change_count = 0, 0, 0, 0
    eye_closed, prev_face_angle = False, None
    frame_index = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = detector(gray)

        for face in faces:
            landmarks = predictor(gray, face)
            left_eye = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in LEFT_EYE])
            right_eye = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in RIGHT_EYE])
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            avg_ear = (left_ear + right_ear) / 2.0

            if avg_ear < EAR_THRESHOLD:
                closed_frames += 1
                if not eye_closed and closed_frames >= CONSEC_FRAMES:
                    blink_counter += 1
                    eye_closed = True
            else:
                closed_frames = 0
                eye_closed = False

        results = face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                drawing_utils.draw_landmarks(frame, face_landmarks, mp.solutions.face_mesh.FACEMESH_TESSELATION,
                                             drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                                             drawing_utils.DrawingSpec(color=(0, 0, 255), thickness=1))
                if is_yawning(face_landmarks.landmark):
                    yawn_count += 1
                nose, chin = face_landmarks.landmark[1], face_landmarks.landmark[152]
                current_face_angle = np.arctan2(chin.y - nose.y, chin.x - nose.x)
                if prev_face_angle is not None and abs(current_face_angle - prev_face_angle) > 0.1:
                    face_angle_change_count += 1
                prev_face_angle = current_face_angle

        # Display information on the frame
        cv2.putText(frame, f"Blinks: {blink_counter}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Yawns: {yawn_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Face Angle Changes: {face_angle_change_count}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        frame_path = os.path.join(output_dir, f"frame_{frame_index:04d}.png")
        cv2.imwrite(frame_path, frame)
        frame_index += 1

    cap.release()

    output_video_path = os.path.join("static", "uploads", "outputs", f"{video_name}.mp4")
    ffmpeg_command = [
        "ffmpeg", "-y", "-framerate", "20", "-i", os.path.join(output_dir, "frame_%04d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", output_video_path
    ]
    subprocess.run(ffmpeg_command)

    print({
        "blinks": blink_counter,
        "yawns": yawn_count,
        "face_angle_changes": face_angle_change_count
    })

if __name__ == "__main__":
    video_path = sys.argv[1]
    analyze_fatigue(video_path)