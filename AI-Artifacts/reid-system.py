import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import time
from collections import deque

class LightweightFeatureExtractor(nn.Module):
    """A feature extractor based on MobileNetV3 for person re-identification (Re-ID).
    
    This class is responsible for extracting meaningful features from input images
    of persons. It utilizes a pre-trained MobileNetV3-Small model as a backbone,
    which is modified to output feature vectors suitable for Re-ID tasks. If the
    pre-trained model fails to load, a simple CNN architecture is used as a fallback.
    The extracted features are L2 normalized for consistency in similarity comparisons.
    """
    def __init__(self, feature_dim=256):
        super(LightweightFeatureExtractor, self).__init__()
        # Load a pre-trained MobileNetV3-Small
        try:
            self.backbone = torch.hub.load('pytorch/vision', 'mobilenet_v3_small', pretrained=True)
            # Remove classifier
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
            # Add Re-ID projection head
            self.projection = nn.Sequential(
                nn.Linear(576, feature_dim),
                nn.BatchNorm1d(feature_dim)
            )
        except:
            print("Error loading pre-trained model. Using a simple CNN instead.")
            # Fallback to a simple CNN if torch.hub is not available
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 32, 3, 2, 1),
                nn.ReLU(),
                nn.Conv2d(32, 64, 3, 2, 1),
                nn.ReLU(),
                nn.Conv2d(64, 128, 3, 2, 1),
                nn.ReLU(),
                nn.Conv2d(128, 256, 3, 2, 1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.projection = nn.Linear(256, feature_dim)
        
    def forward(self, x):
        if hasattr(self, 'backbone') and isinstance(self.backbone, nn.Sequential):
            # MobileNetV3 path
            features = self.backbone(x)
            features = features.squeeze(-1).squeeze(-1)
            features = self.projection(features)
        else:
            # Fallback CNN path
            features = self.backbone(x)
            features = features.view(features.size(0), -1)
            features = self.projection(features)
        
        # L2 normalize features
        features = F.normalize(features, p=2, dim=1)
        return features

class PersonTracker:
    """A class for tracking persons using Re-ID capabilities.
    
    This class implements a person tracking system that utilizes a feature extractor
    to identify and track individuals across video frames. It employs YOLOv4-tiny for
    person detection and maintains a gallery of features for the selected target person.
    The tracker computes similarity scores to match detected persons with the target
    and handles cases where tracking is lost. It also provides methods for selecting
    a target person and processing video frames for tracking.
    """
    def __init__(self, feature_dim=256, max_gallery_size=10, similarity_threshold=0.6):
        # Initialize the person detector
        self.detector = cv2.dnn.readNetFromDarknet(
            "yolov4-tiny.cfg", "yolov4-tiny.weights")
        self.detector.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.detector.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
        # Load COCO class names
        with open("coco.names", "r") as f:
            self.class_names = [line.strip() for line in f.readlines()]
        
        # Initialize feature extractor
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.feature_extractor = LightweightFeatureExtractor(feature_dim=feature_dim)
        self.feature_extractor.to(self.device)
        self.feature_extractor.eval()
        
        # Initialize transformation for the feature extractor
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 64)),  # Standard size for Re-ID
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Gallery of target person features
        self.target_gallery = deque(maxlen=max_gallery_size)
        self.target_selected = False
        self.similarity_threshold = similarity_threshold
        
        # Frame processing parameters
        self.process_every_n_frames = 3  # Process every 3rd frame for Re-ID
        self.frame_count = 0
        
        # Tracking state
        self.last_seen_position = None
        self.tracking_lost_counter = 0
        self.max_tracking_lost = 30  # Number of frames to keep tracking without a match
        
    def detect_people(self, frame):
        """Detect people in the frame using YOLOv4-tiny"""
        height, width = frame.shape[:2]
        
        # Create a blob from the frame
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.detector.setInput(blob)
        
        # Get output layer names
        output_layers = self.detector.getUnconnectedOutLayersNames()
        layer_outputs = self.detector.forward(output_layers)
        
        # Process detections
        boxes = []
        confidences = []
        class_ids = []
        
        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                # Filter for people (class ID 0 in COCO)
                if class_id == 0 and confidence > 0.5:
                    # YOLO format to OpenCV format
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        # Apply non-max suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        person_boxes = []
        
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                # Ensure the box is within the frame boundaries
                x = max(0, x)
                y = max(0, y)
                w = min(width - x, w)
                h = min(height - y, h)
                person_boxes.append((x, y, w, h))
                
        return person_boxes
    
    def extract_features(self, frame, box):
        """Extract Re-ID features for a person in a bounding box"""
        x, y, w, h = box
        if w <= 0 or h <= 0:
            return None
            
        # Extract the person from the frame
        person_img = frame[y:y+h, x:x+w]
        if person_img.size == 0:
            return None
            
        # Transform the image for the feature extractor
        try:
            tensor = self.transform(person_img).unsqueeze(0).to(self.device)
            
            # Extract features
            with torch.no_grad():
                features = self.feature_extractor(tensor)
            
            return features.cpu().numpy()[0]
        except Exception as e:
            print(f"Error extracting features: {e}")
            return None
    
    def compute_similarity(self, features):
        """Compute similarity between input features and target gallery"""
        if not self.target_gallery:
            return 0.0
            
        # Compute cosine similarity with all gallery features
        similarities = []
        for gallery_features in self.target_gallery:
            similarity = np.dot(features, gallery_features)
            similarities.append(similarity)
            
        # Return maximum similarity
        return max(similarities)
    
    def update_gallery(self, features):
        """Update the target person's feature gallery"""
        self.target_gallery.append(features)
    
    def select_target(self, frame, selected_box):
        """Select a target person to track"""
        features = self.extract_features(frame, selected_box)
        if features is not None:
            self.target_gallery.clear()
            self.update_gallery(features)
            self.target_selected = True
            self.last_seen_position = selected_box
            print("Target selected and features extracted")
            return True
        else:
            print("Failed to extract features for the target")
            return False
    
    def find_target(self, frame):
        """Find the target person in the current frame"""
        self.frame_count += 1
        
        # Detect all people in the frame
        person_boxes = self.detect_people(frame)
        
        # If no target selected or no people detected, return
        if not self.target_selected or not person_boxes:
            return None
            
        # Skip feature extraction on some frames for efficiency
        if self.frame_count % self.process_every_n_frames != 0:
            return self.last_seen_position
            
        # For each detected person, compare with the target gallery
        best_match = None
        highest_similarity = 0
        
        for box in person_boxes:
            features = self.extract_features(frame, box)
            if features is not None:
                similarity = self.compute_similarity(features)
                
                if similarity > highest_similarity:
                    highest_similarity = similarity
                    best_match = box
        
        # If a good match is found
        if highest_similarity > self.similarity_threshold:
            # Update the gallery with the new features to adapt to appearance changes
            if highest_similarity > 0.8:  # Only update if very confident
                features = self.extract_features(frame, best_match)
                if features is not None:
                    self.update_gallery(features)
            
            self.tracking_lost_counter = 0
            self.last_seen_position = best_match
            return best_match
        elif self.last_seen_position is not None:
            # Increment lost counter if no match found
            self.tracking_lost_counter += 1
            
            # If lost for too many frames, consider the person gone
            if self.tracking_lost_counter > self.max_tracking_lost:
                self.last_seen_position = None
                
            return self.last_seen_position
            
        return None
    
    def process_frame(self, frame, selected_box=None):
        """Process a frame and return the tracking result"""
        # If a box is provided, select it as the target
        if selected_box is not None:
            self.select_target(frame, selected_box)
            return frame, selected_box
        
        # Otherwise, try to find and track the target
        target_box = self.find_target(frame)
        
        # Visualize the result
        result_frame = frame.copy()
        if target_box is not None:
            x, y, w, h = target_box
            cv2.rectangle(result_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(result_frame, "Target", (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        return result_frame, target_box

def main():
    # Main function to initialize and run the person Re-Identification (Re-ID) tracker,
    # which processes video frames to detect and track individuals based on their features.
    # The tracker can operate with a webcam or a video file, allowing for real-time 
    # person tracking and selection.
    # Initialize tracker
    tracker = PersonTracker(feature_dim=256, similarity_threshold=0.6)
    
    # Open webcam or video file
    cap = cv2.VideoCapture(0)  # 0 for webcam, or provide a video file path
    
    target_selected = False
    selected_box = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process the frame
        if target_selected:
            result_frame, tracked_box = tracker.process_frame(frame)
        else:
            # Detect people for selection
            person_boxes = tracker.detect_people(frame)
            result_frame = frame.copy()
            
            # Draw all detected people
            for i, box in enumerate(person_boxes):
                x, y, w, h = box
                cv2.rectangle(result_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(result_frame, f"Person {i+1}", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        
        # Display the result
        cv2.imshow("Person Tracker", result_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('s') and not target_selected:
            # Select the first person as target
            if person_boxes:
                selected_box = person_boxes[0]
                target_selected = tracker.select_target(frame, selected_box)
        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
