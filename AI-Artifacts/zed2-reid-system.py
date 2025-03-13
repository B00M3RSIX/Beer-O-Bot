import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import time
from collections import deque
import pyzed.sl as sl
import tensorrt as trt
import threading

class TRTFeatureExtractor:
    """TensorRT-optimized feature extractor for Re-ID"""
    def __init__(self, engine_path, feature_dim=256):
        # Load TensorRT engine
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        
        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
            
        self.context = self.engine.create_execution_context()
        self.input_shape = (1, 3, 256, 128)  # Standard Re-ID size (batch, channels, height, width)
        self.output_shape = (1, feature_dim)
        
        # Allocate device memory
        self.inputs = []
        self.outputs = []
        self.bindings = []
        
        for binding in range(self.engine.num_bindings):
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.engine.max_batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            self.bindings.append(int(device_mem))
            
            if self.engine.binding_is_input(binding):
                self.inputs.append({"host": host_mem, "device": device_mem})
            else:
                self.outputs.append({"host": host_mem, "device": device_mem})
        
        # Create CUDA stream
        self.stream = cuda.Stream()
        
        # Transformations
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def extract(self, img):
        """Extract features from an image"""
        # Preprocess image
        input_tensor = self.transform(img).unsqueeze(0).numpy()
        
        # Copy input data to device
        np.copyto(self.inputs[0]["host"], input_tensor.ravel())
        cuda.memcpy_htod_async(self.inputs[0]["device"], self.inputs[0]["host"], self.stream)
        
        # Execute inference
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        
        # Copy output back to host
        cuda.memcpy_dtoh_async(self.outputs[0]["host"], self.outputs[0]["device"], self.stream)
        self.stream.synchronize()
        
        # Process output
        output = self.outputs[0]["host"].reshape(self.output_shape)
        
        # Normalize features
        features = output / np.linalg.norm(output)
        
        return features

class ZED2ReIDTracker:
    """Person Re-ID tracker optimized for ZED2 camera and Jetson Orin NX"""
    def __init__(self, feature_dim=256, max_gallery_size=15, similarity_threshold=0.65):
        # Initialize ZED camera
        self.zed = sl.Camera()
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.HD720  # 720p resolution
        init_params.depth_mode = sl.DEPTH_MODE.ULTRA  # Best quality depth
        init_params.coordinate_units = sl.UNIT.METER
        init_params.sdk_verbose = True
        
        # Open the camera
        err = self.zed.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            print(f"Error opening ZED camera: {err}")
            exit(-1)
        
        # Enable positional tracking for better spatial awareness
        tracking_params = sl.PositionalTrackingParameters()
        self.zed.enable_positional_tracking(tracking_params)
        
        # Initialize object detection model
        self.obj_param = sl.ObjectDetectionParameters()
        self.obj_param.enable_tracking = True
        self.obj_param.enable_mask_output = True
        self.obj_param.detection_model = sl.OBJECT_DETECTION_MODEL.PERSON_HEAD_DETECTION_ACCURATE
        
        self.zed.enable_object_detection(self.obj_param)
        
        # Initialize runtime parameters
        self.obj_runtime_param = sl.ObjectDetectionRuntimeParameters()
        self.obj_runtime_param.detection_confidence_threshold = 40
        
        # Initialize feature extractor with TensorRT
        try:
            self.feature_extractor = TRTFeatureExtractor("reid_model.engine", feature_dim)
            print("TensorRT feature extractor initialized")
        except Exception as e:
            print(f"Failed to load TensorRT engine: {e}")
            print("Falling back to PyTorch model")
            self.feature_extractor = self._init_fallback_model(feature_dim)
        
        # Gallery of target person features
        self.target_gallery = deque(maxlen=max_gallery_size)
        self.target_selected = False
        self.similarity_threshold = similarity_threshold
        
        # Frame processing parameters
        self.process_every_n_frames = 2  # Process every 2nd frame for Re-ID (Orin can handle this)
        self.frame_count = 0
        
        # Tracking state
        self.last_seen_position = None
        self.tracking_lost_counter = 0
        self.max_tracking_lost = 45  # Increased for better persistence
        
        # 3D position tracking
        self.target_3d_position = None
        self.kalman_filter = self._init_kalman_filter()
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Image and point cloud containers
        self.image = sl.Mat()
        self.point_cloud = sl.Mat()
        self.depth = sl.Mat()
        
    def _init_fallback_model(self, feature_dim):
        """Initialize fallback PyTorch model if TensorRT fails"""
        model = torch.hub.load('pytorch/vision', 'mobilenet_v3_small', pretrained=True)
        # Remove classifier
        modules = list(model.children())[:-1]
        backbone = nn.Sequential(*modules)
        # Add Re-ID head
        model = nn.Sequential(
            backbone,
            nn.Flatten(),
            nn.Linear(576, feature_dim),
            nn.BatchNorm1d(feature_dim)
        )
        # Set to evaluation mode
        model.eval()
        # Move to CUDA if available
        if torch.cuda.is_available():
            model = model.cuda()
        
        return model
    
    def _init_kalman_filter(self):
        """Initialize Kalman filter for 3D position tracking"""
        kf = cv2.KalmanFilter(6, 3)  # 6 state variables (x,y,z,dx,dy,dz), 3 measurements (x,y,z)
        kf.measurementMatrix = np.array([[1, 0, 0, 0, 0, 0],
                                         [0, 1, 0, 0, 0, 0],
                                         [0, 0, 1, 0, 0, 0]], np.float32)
        kf.transitionMatrix = np.array([[1, 0, 0, 1, 0, 0],
                                        [0, 1, 0, 0, 1, 0],
                                        [0, 0, 1, 0, 0, 1],
                                        [0, 0, 0, 1, 0, 0],
                                        [0, 0, 0, 0, 1, 0],
                                        [0, 0, 0, 0, 0, 1]], np.float32)
        kf.processNoiseCov = np.eye(6, dtype=np.float32) * 0.03
        return kf
    
    def capture_frame(self):
        """Capture a new frame from ZED camera"""
        if self.zed.grab() == sl.ERROR_CODE.SUCCESS:
            # Retrieve image
            self.zed.retrieve_image(self.image, sl.VIEW.LEFT)
            # Retrieve depth map
            self.zed.retrieve_measure(self.depth, sl.MEASURE.DEPTH)
            # Retrieve point cloud
            self.zed.retrieve_measure(self.point_cloud, sl.MEASURE.XYZRGBA)
            
            return np.array(self.image.get_data())
        else:
            return None
    
    def detect_people(self):
        """Detect people using ZED SDK"""
        objects = sl.Objects()
        if self.zed.retrieve_objects(objects, self.obj_runtime_param) == sl.ERROR_CODE.SUCCESS:
            person_boxes = []
            
            for obj in objects.object_list:
                if obj.class_name == "Person":
                    # Get 2D bounding box
                    bbox = obj.bounding_box_2d
                    x = int(bbox[0][0])
                    y = int(bbox[0][1])
                    w = int(bbox[2][0] - bbox[0][0])
                    h = int(bbox[2][1] - bbox[0][1])
                    
                    # Get 3D position (average of head and torso position)
                    position_3d = obj.position
                    
                    person_boxes.append({
                        "box": (x, y, w, h),
                        "confidence": obj.confidence,
                        "position_3d": (position_3d[0], position_3d[1], position_3d[2]),
                        "tracking_id": obj.id  # ZED provides tracking IDs
                    })
            
            return person_boxes
        else:
            return []
    
    def extract_features(self, frame, box):
        """Extract Re-ID features for a person in a bounding box"""
        x, y, w, h = box
        if w <= 0 or h <= 0:
            return None
            
        # Extract the person from the frame
        person_img = frame[y:y+h, x:x+w]
        if person_img.size == 0:
            return None
            
        # Extract features
        try:
            if isinstance(self.feature_extractor, TRTFeatureExtractor):
                features = self.feature_extractor.extract(person_img)
            else:
                # Fallback to PyTorch
                tensor = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((256, 128)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])(person_img).unsqueeze(0)
                
                if torch.cuda.is_available():
                    tensor = tensor.cuda()
                
                with torch.no_grad():
                    features = self.feature_extractor(tensor)
                    features = F.normalize(features, p=2, dim=1)
                    features = features.cpu().numpy()[0]
            
            return features
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
        with self.lock:
            self.target_gallery.append(features)
    
    def select_target(self, frame, person_data):
        """Select a target person to track"""
        box = person_data["box"]
        features = self.extract_features(frame, box)
        
        if features is not None:
            with self.lock:
                self.target_gallery.clear()
                self.update_gallery(features)
                self.target_selected = True
                self.last_seen_position = box
                self.target_3d_position = person_data["position_3d"]
                
                # Initialize Kalman filter state
                self.kalman_filter.statePre = np.array([
                    [self.target_3d_position[0]],
                    [self.target_3d_position[1]],
                    [self.target_3d_position[2]],
                    [0], [0], [0]
                ], np.float32)
                
            print("Target selected and features extracted")
            return True
        else:
            print("Failed to extract features for the target")
            return False
    
    def find_target(self, frame, person_detections):
        """Find the target person in the current frame"""
        self.frame_count += 1
        
        # If no target selected or no people detected, return
        if not self.target_selected or not person_detections:
            return None
            
        # Skip feature extraction on some frames for efficiency
        if self.frame_count % self.process_every_n_frames != 0:
            return self.last_seen_position, self.target_3d_position
        
        # Kalman filter prediction
        predicted_state = self.kalman_filter.predict()
        predicted_position = (
            predicted_state[0][0],
            predicted_state[1][0],
            predicted_state[2][0]
        )
        
        # For each detected person, compare with the target gallery
        best_match = None
        highest_similarity = 0
        best_3d_position = None
        
        for person in person_detections:
            box = person["box"]
            person_3d = person["position_3d"]
            
            # Calculate 3D distance to predicted position
            distance_3d = np.sqrt(
                (predicted_position[0] - person_3d[0])**2 +
                (predicted_position[1] - person_3d[1])**2 +
                (predicted_position[2] - person_3d[2])**2
            )
            
            # Skip if too far (>5 meters) for efficiency
            if distance_3d > 5.0:
                continue
                
            # Proximity score (higher for closer)
            proximity_score = max(0, 1.0 - distance_3d / 5.0)
            
            features = self.extract_features(frame, box)
            if features is not None:
                # Get appearance similarity
                appearance_similarity = self.compute_similarity(features)
                
                # Combined score (70% appearance, 30% proximity)
                combined_score = 0.7 * appearance_similarity + 0.3 * proximity_score
                
                if combined_score > highest_similarity:
                    highest_similarity = combined_score
                    best_match = box
                    best_3d_position = person_3d
        
        # If a good match is found
        if highest_similarity > self.similarity_threshold:
            # Update the gallery with the new features to adapt to appearance changes
            if highest_similarity > 0.75:  # Only update if very confident
                features = self.extract_features(frame, best_match)
                if features is not None:
                    self.update_gallery(features)
            
            # Update Kalman filter
            measurement = np.array([[best_3d_position[0]], 
                                   [best_3d_position[1]], 
                                   [best_3d_position[2]]], np.float32)
            self.kalman_filter.correct(measurement)
            
            self.tracking_lost_counter = 0
            self.last_seen_position = best_match
            self.target_3d_position = best_3d_position
            return best_match, best_3d_position
            
        elif self.last_seen_position is not None:
            # Increment lost counter if no match found
            self.tracking_lost_counter += 1
            
            # If lost for too many frames, consider the person gone
            if self.tracking_lost_counter > self.max_tracking_lost:
                self.last_seen_position = None
                self.target_3d_position = None
                
            return self.last_seen_position, self.target_3d_position
            
        return None, None
    
    def get_target_distance(self):
        """Get distance to target in meters"""
        if self.target_3d_position is not None:
            return self.target_3d_position[2]  # Z coordinate is distance
        return None
    
    def process_frame(self, selected_person=None):
        """Process a frame and return the tracking result"""
        # Capture new frame
        frame = self.capture_frame()
        if frame is None:
            return None, None, None
        
        # Detect people
        person_detections = self.detect_people()
        
        # If a person is provided, select it as the target
        if selected_person is not None:
            self.select_target(frame, selected_person)
            return frame, selected_person["box"], selected_person["position_3d"]
        
        # Otherwise, try to find and track the target
        target_box, target_3d = self.find_target(frame, person_detections)
        
        # Visualize the result
        result_frame = frame.copy()
        
        # Draw all detected people
        for person in person_detections:
            x, y, w, h = person["box"]
            cv2.rectangle(result_frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        
        # Highlight the target
        if target_box is not None:
            x, y, w, h = target_box
            cv2.rectangle(result_frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
            
            # Display distance
            if target_3d is not None:
                distance = target_3d[2]  # Z is distance
                cv2.putText(result_frame, f"Target: {distance:.2f}m", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        return result_frame, target_box, target_3d
    
    def close(self):
        """Close camera and release resources"""
        self.zed.disable_object_detection()
        self.zed.disable_positional_tracking()
        self.zed.close()

def main():
    """Main function to run the ZED2 Re-ID tracker"""
    # Initialize tracker
    tracker = ZED2ReIDTracker(feature_dim=256, similarity_threshold=0.65)
    
    target_selected = False
    
    while True:
        # Process the frame
        result_frame, target_box, target_3d = tracker.process_frame()
        if result_frame is None:
            print("Failed to capture frame")
            time.sleep(0.1)
            continue
        
        # Display the result
        cv2.imshow("ZED2 Person Re-ID Tracker", result_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('s') and not target_selected:
            # Select a person as target (the first detected person)
            person_detections = tracker.detect_people()
            if person_detections:
                result_frame, _, _ = tracker.process_frame(person_detections[0])
                target_selected = True
                cv2.imshow("ZED2 Person Re-ID Tracker", result_frame)
        
    tracker.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
