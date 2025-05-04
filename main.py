import cv2
import mediapipe as mp
import time
import numpy as np

class FaceRecognizer:
    def __init__(self, static_image=False, max_faces=5, min_detection_confidence=0.5):
        """Initialize the FaceRecognizer with MediaPipe Face Mesh.
        
        Args:
            static_image: Whether to treat images as static (not video)
            max_faces: Maximum number of faces to detect
            min_detection_confidence: Minimum confidence for face detection
        """
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=static_image,
            max_num_faces=max_faces,
            min_detection_confidence=min_detection_confidence
        )
        
        # Initialize MediaPipe Drawing utilities
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize MediaPipe Face Detection (for bounding boxes)
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            min_detection_confidence=min_detection_confidence
        )
        
        # Important facial landmarks indices
        self.LEFT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.RIGHT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        self.LIPS_INDICES = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 61]
        
        # For FPS calculation
        self.prev_time = 0
        self.curr_time = 0

    def analyze_face(self, face_landmarks):
        """Analyze facial landmarks to extract features.
        
        Args:
            face_landmarks: MediaPipe face landmarks
            
        Returns:
            dict: Dictionary with facial features analysis
        """
        # Convert landmarks to numpy array for calculations
        landmarks = np.array([[lm.x, lm.y, lm.z] for lm in face_landmarks.landmark])
        
        # Calculate face orientation (simple approximation)
        # Using nose tip (landmark 1) and points on sides of face
        nose_tip = landmarks[1]
        left_temple = landmarks[234]
        right_temple = landmarks[454]
        
        # Face orientation based on nose position relative to face center
        face_center_x = (left_temple[0] + right_temple[0]) / 2
        face_orientation = "straight"
        if nose_tip[0] < face_center_x - 0.02:
            face_orientation = "looking left"
        elif nose_tip[0] > face_center_x + 0.02:
            face_orientation = "looking right"
            
       
            
    
        # Analyze results
        analysis = {
            "orientation": face_orientation,
        
        }
        
        return analysis

    def process_frame(self, frame, draw=True, show_fps=True):
        """Process a video frame for face detection and analysis.
        
        Args:
            frame: Input video frame
            draw: Whether to draw landmarks and bounding boxes
            show_fps: Whether to display FPS
            
        Returns:
            Processed frame with visualizations
            Dictionary with detected faces and analysis
        """
        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe Face Mesh
        mesh_results = self.face_mesh.process(rgb_frame)
        
        # Process with MediaPipe Face Detection
        detection_results = self.face_detection.process(rgb_frame)
        
        # Create output dictionary
        output = {
            "faces_detected": 0,
            "faces": []
        }
        
        # Height and width for display positioning
        h, w, _ = frame.shape
        
        # Process face detection results
        if detection_results.detections:
            output["faces_detected"] = len(detection_results.detections)
            
            for idx, detection in enumerate(detection_results.detections):
                # Get bounding box
                bbox = detection.location_data.relative_bounding_box
                x, y = int(bbox.xmin * w), int(bbox.ymin * h)
                width, height = int(bbox.width * w), int(bbox.height * h)
                
                # Draw bounding box if requested
                if draw:
                    cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
                    
                    # Display detection score
                    score = round(detection.score[0] * 100, 1)
                    cv2.putText(frame, f"Face {idx+1}: {score}%", 
                                (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Process face mesh results
        if mesh_results.multi_face_landmarks:
            # Update face count if mesh detected more faces
            faces_count = len(mesh_results.multi_face_landmarks)
            output["faces_detected"] = max(output["faces_detected"], faces_count)
            
            # Process each face
            for idx, face_landmarks in enumerate(mesh_results.multi_face_landmarks):
                # Analyze facial features
                analysis = self.analyze_face(face_landmarks)
                
                # Store results
                face_info = {
                    "id": idx,
                    "analysis": analysis
                }
                output["faces"].append(face_info)
                
                if draw:
                    # Draw face mesh with tesselation
                    self.mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
                    )
                    
                    # Draw eyes
                    self.mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=self.mp_face_mesh.FACEMESH_LEFT_EYE,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style()
                    )
                    self.mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=self.mp_face_mesh.FACEMESH_RIGHT_EYE,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style()
                    )
                    
                    # Draw lips
                    self.mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=self.mp_face_mesh.FACEMESH_LIPS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style()
                    )
                    
                    # Display facial analysis info
                    # Find position for text (use top of bounding box if available)
                    if idx < len(detection_results.detections or []):
                        bbox = detection_results.detections[idx].location_data.relative_bounding_box
                        y_pos = int(bbox.ymin * h) + height + 20
                        x_pos = int(bbox.xmin * w)
                    else:
                        # Fallback position if no detection bounding box
                        landmarks = np.array([[lm.x, lm.y] for lm in face_landmarks.landmark])
                        x_pos = int(np.min(landmarks[:, 0]) * w)
                        y_pos = int(np.max(landmarks[:, 1]) * h) + 20
                    
                    # Display analysis text
                    cv2.putText(frame, f"Orientation: {analysis['orientation']}", 
                                (x_pos, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)
                   
        
        # Calculate and display FPS
        if show_fps:
            self.curr_time = time.time()
            fps = 1 / (self.curr_time - self.prev_time) if self.prev_time > 0 else 0
            self.prev_time = self.curr_time
            cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        return frame, output

def main():
    """Main function to run the face recognizer with webcam."""
    # Initialize the face recognizer
    face_recognizer = FaceRecognizer(
        static_image=False,
        max_faces=5,
        min_detection_confidence=0.5
    )
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    # Check if webcam is opened successfully
    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return
    
    # Set webcam properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    try:
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to read frame.")
                break
            
            # Mirror image for more intuitive interaction
            frame = cv2.flip(frame, 1)
            
            # Process frame
            processed_frame, face_data = face_recognizer.process_frame(frame)
            
            # Display the resulting frame
            cv2.imshow("Face Recognizer", processed_frame)
            
            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except Exception as e:
        print(f"[ERROR] An error occurred: {e}")
    
    finally:
        # Release resources
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()