import qdrant_client
from qdrant_client.models import VectorParams, Distance, PointStruct
import numpy as np
import uuid
import time
import threading
from collections import deque
import json
import os

class VectorGallery:
    """Vector database gallery for Re-ID using Qdrant embedded mode"""
    
    def __init__(self, feature_dim=256, collection_name="reid_targets", 
                 persistence_path="./vector_db", similarity_threshold=0.65):
        # Create embedded Qdrant client
        self.client = qdrant_client.QdrantClient(
            path=persistence_path,
            prefer_grpc=False
        )
        
        self.collection_name = collection_name
        self.feature_dim = feature_dim
        self.similarity_threshold = similarity_threshold
        
        # Create collection if it doesn't exist
        try:
            collection_info = self.client.get_collection(collection_name)
            print(f"Using existing collection: {collection_name}")
        except:
            print(f"Creating new collection: {collection_name}")
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=feature_dim, distance=Distance.COSINE)
            )
        
        self.lock = threading.Lock()
        
        # Cache for active targets to reduce DB lookups
        self.active_targets_cache = {}
        
        # In-memory buffer for recent vectors (for quick access during tracking)
        self.memory_buffer = {}
        
    def add_target(self, features, target_id=None, metadata=None):
        """Add a target to the gallery"""
        with self.lock:
            # Generate a target ID if not provided
            if target_id is None:
                target_id = str(uuid.uuid4())
            
            # Default metadata
            if metadata is None:
                metadata = {
                    "created_at": time.time(),
                    "updated_at": time.time(),
                    "frame_count": 1,
                    "active": True
                }
            
            # Add to vector database
            point_id = f"{target_id}_{metadata.get('frame_count', 1)}"
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    PointStruct(
                        id=point_id,
                        vector=features.tolist(),
                        payload={
                            "target_id": target_id,
                            **metadata
                        }
                    )
                ]
            )
            
            # Update in-memory buffer
            if target_id not in self.memory_buffer:
                self.memory_buffer[target_id] = deque(maxlen=15)
            self.memory_buffer[target_id].append(features)
            
            # Add to active targets cache
            self.active_targets_cache[target_id] = {
                "last_seen": time.time(),
                "metadata": metadata
            }
            
            return target_id
            
    def update_target(self, target_id, features, metadata_update=None):
        """Update an existing target with new features"""
        with self.lock:
            # Get current metadata
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=features.tolist(),
                limit=1,
                query_filter=qdrant_client.models.Filter(
                    must=[
                        qdrant_client.models.FieldCondition(
                            key="target_id",
                            match=qdrant_client.models.MatchValue(value=target_id)
                        )
                    ]
                )
            )
            
            if not search_result:
                # Target not found, add as new
                return self.add_target(features, target_id)
            
            # Get existing metadata
            metadata = search_result[0].payload
            frame_count = metadata.get("frame_count", 1) + 1
            
            # Update metadata
            metadata.update({
                "updated_at": time.time(),
                "frame_count": frame_count,
                "active": True
            })
            
            # Apply custom metadata updates
            if metadata_update:
                metadata.update(metadata_update)
            
            # Add new features
            point_id = f"{target_id}_{frame_count}"
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    PointStruct(
                        id=point_id,
                        vector=features.tolist(),
                        payload=metadata
                    )
                ]
            )
            
            # Update in-memory buffer
            if target_id not in self.memory_buffer:
                self.memory_buffer[target_id] = deque(maxlen=15)
            self.memory_buffer[target_id].append(features)
            
            # Update active targets cache
            self.active_targets_cache[target_id] = {
                "last_seen": time.time(),
                "metadata": metadata
            }
            
            return target_id
    
    def find_match(self, features, limit=5, memory_only=False, active_only=True):
        """Find the best matching target for the given features"""
        best_match = None
        highest_similarity = 0
        best_match_metadata = None
        
        # First check in-memory buffer for active targets (fast path)
        for target_id, feature_buffer in self.memory_buffer.items():
            # Skip inactive targets if requested
            if active_only and target_id in self.active_targets_cache:
                if not self.active_targets_cache[target_id]["metadata"].get("active", True):
                    continue
            
            for gallery_features in feature_buffer:
                similarity = np.dot(features, gallery_features)
                if similarity > highest_similarity:
                    highest_similarity = similarity
                    best_match = target_id
        
        # If good match found in memory or memory_only flag set, return result
        if highest_similarity > self.similarity_threshold or memory_only:
            if best_match and best_match in self.active_targets_cache:
                best_match_metadata = self.active_targets_cache[best_match]["metadata"]
            return best_match, highest_similarity, best_match_metadata
        
        # Otherwise, search in vector DB
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=features.tolist(),
            limit=limit,
            query_filter=qdrant_client.models.Filter(
                must=[
                    qdrant_client.models.FieldCondition(
                        key="active",
                        match=qdrant_client.models.MatchValue(value=True)
                    )
                ]
            ) if active_only else None
        )
        
        if search_result:
            best_db_match = search_result[0]
            similarity = 1.0 - best_db_match.score  # Convert distance to similarity
            
            if similarity > highest_similarity:
                best_match = best_db_match.payload.get("target_id")
                highest_similarity = similarity
                best_match_metadata = best_db_match.payload
        
        return best_match, highest_similarity, best_match_metadata
    
    def get_target_info(self, target_id):
        """Get information about a specific target"""
        # Check cache first
        if target_id in self.active_targets_cache:
            return self.active_targets_cache[target_id]["metadata"]
        
        # Search in DB
        search_result = self.client.scroll(
            collection_name=self.collection_name,
            limit=1,
            query_filter=qdrant_client.models.Filter(
                must=[
                    qdrant_client.models.FieldCondition(
                        key="target_id",
                        match=qdrant_client.models.MatchValue(value=target_id)
                    )
                ]
            )
        )
        
        if search_result and search_result[0]:
            return search_result[0][0].payload
        
        return None
    
    def deactivate_target(self, target_id):
        """Mark a target as inactive"""
        with self.lock:
            # Get all points for this target
            search_result = self.client.scroll(
                collection_name=self.collection_name,
                limit=100,
                query_filter=qdrant_client.models.Filter(
                    must=[
                        qdrant_client.models.FieldCondition(
                            key="target_id",
                            match=qdrant_client.models.MatchValue(value=target_id)
                        )
                    ]
                )
            )
            
            if not search_result or not search_result[0]:
                return False
            
            # Update all points for this target
            for point in search_result[0]:
                point_id = point.id
                payload = point.payload
                payload["active"] = False
                payload["deactivated_at"] = time.time()
                
                self.client.update_vectors(
                    collection_name=self.collection_name,
                    points=[
                        (point_id, payload)
                    ]
                )
            
            # Update cache
            if target_id in self.active_targets_cache:
                self.active_targets_cache[target_id]["metadata"]["active"] = False
                self.active_targets_cache[target_id]["metadata"]["deactivated_at"] = time.time()
            
            # Clear from memory buffer
            if target_id in self.memory_buffer:
                del self.memory_buffer[target_id]
            
            return True
    
    def cleanup_inactive(self, max_age_days=30):
        """Remove inactive targets older than specified days"""
        with self.lock:
            current_time = time.time()
            max_age_seconds = max_age_days * 24 * 60 * 60
            
            # Find inactive targets
            search_result = self.client.scroll(
                collection_name=self.collection_name,
                limit=1000,
                query_filter=qdrant_client.models.Filter(
                    must=[
                        qdrant_client.models.FieldCondition(
                            key="active",
                            match=qdrant_client.models.MatchValue(value=False)
                        )
                    ]
                )
            )
            
            if not search_result or not search_result[0]:
                return 0
            
            # Group by target_id
            targets_to_delete = set()
            for point in search_result[0]:
                deactivated_at = point.payload.get("deactivated_at", 0)
                if current_time - deactivated_at > max_age_seconds:
                    targets_to_delete.add(point.payload.get("target_id"))
            
            # Delete targets
            deleted_count = 0
            for target_id in targets_to_delete:
                self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=qdrant_client.models.Filter(
                        must=[
                            qdrant_client.models.FieldCondition(
                                key="target_id",
                                match=qdrant_client.models.MatchValue(value=target_id)
                            )
                        ]
                    )
                )
                deleted_count += 1
                
                # Remove from cache
                if target_id in self.active_targets_cache:
                    del self.active_targets_cache[target_id]
            
            return deleted_count
    
    def save_target_metadata(self, target_id, filename):
        """Save all metadata for a target to JSON file"""
        info = self.get_target_info(target_id)
        if info:
            with open(filename, 'w') as f:
                json.dump(info, f, indent=2)
            return True
        return False
        
    def close(self):
        """Clean up resources"""
        self.client.close()


# Integration with ZED2ReIDTracker
class EnhancedZED2ReIDTracker:
    """ZED2 Re-ID tracker with vector database integration"""
    
    def __init__(self, feature_dim=256, similarity_threshold=0.65, 
                 db_path="./vector_db"):
        # Initialize ZED camera and other components as before
        # (ZED initialization code here...)
        
        # Initialize vector gallery
        self.vector_gallery = VectorGallery(
            feature_dim=feature_dim,
            similarity_threshold=similarity_threshold,
            persistence_path=db_path
        )
        
        # Tracking state
        self.current_target_id = None
        self.last_seen_position = None
        self.tracking_lost_counter = 0
        self.max_tracking_lost = 45
        
        # Feature extraction and other components as before
        # (Feature extractor initialization code here...)
    
    def select_target(self, frame, person_data):
        """Select a target person to track"""
        box = person_data["box"]
        features = self.extract_features(frame, box)
        
        if features is not None:
            # Add to vector gallery
            self.current_target_id = self.vector_gallery.add_target(
                features,
                metadata={
                    "first_seen_position": person_data["position_3d"],
                    "height_estimate": person_data["box"][3],
                    "last_position": person_data["position_3d"],
                    "active": True
                }
            )
            
            self.last_seen_position = box
            self.target_3d_position = person_data["position_3d"]
            
            # Initialize Kalman filter state
            # (Kalman filter code here...)
            
            print(f"Target selected with ID: {self.current_target_id}")
            return True
        else:
            print("Failed to extract features for the target")
            return False
    
    def find_target(self, frame, person_detections):
        """Find the target person in the current frame"""
        self.frame_count += 1
        
        # If no target selected or no people detected, return
        if not self.current_target_id or not person_detections:
            return None, None
            
        # Skip feature extraction on some frames for efficiency
        if self.frame_count % self.process_every_n_frames != 0:
            return self.last_seen_position, self.target_3d_position
        
        # Kalman filter prediction
        # (Kalman filter code here...)
        
        # For each detected person, compare with target features
        best_match = None
        highest_similarity = 0
        best_3d_position = None
        
        for person in person_detections:
            box = person["box"]
            person_3d = person["position_3d"]
            
            # Calculate 3D distance to predicted position
            # (Distance calculation code here...)
            
            features = self.extract_features(frame, box)
            if features is not None:
                # Find match in vector gallery (optimized: memory-only first)
                match_id, similarity, metadata = self.vector_gallery.find_match(
                    features, memory_only=True
                )
                
                if match_id == self.current_target_id:
                    # Combined score (70% appearance, 30% proximity)
                    combined_score = 0.7 * similarity + 0.3 * proximity_score
                    
                    if combined_score > highest_similarity:
                        highest_similarity = combined_score
                        best_match = box
                        best_3d_position = person_3d
        
        # If memory search fails, try database search
        if highest_similarity < self.similarity_threshold:
            for person in person_detections:
                box = person["box"]
                person_3d = person["position_3d"]
                
                features = self.extract_features(frame, box)
                if features is not None:
                    # Full vector DB search
                    match_id, similarity, metadata = self.vector_gallery.find_match(
                        features, memory_only=False
                    )
                    
                    if match_id == self.current_target_id:
                        # Combined score (70% appearance, 30% proximity)
                        combined_score = 0.7 * similarity + 0.3 * proximity_score
                        
                        if combined_score > highest_similarity:
                            highest_similarity = combined_score
                            best_match = box
                            best_3d_position = person_3d
        
        # If a good match is found
        if highest_similarity > self.similarity_threshold:
            # Update the gallery with the new features
            if highest_similarity > 0.75:  # Only update if very confident
                features = self.extract_features(frame, best_match)
                if features is not None:
                    self.vector_gallery.update_target(
                        self.current_target_id, 
                        features,
                        metadata_update={
                            "last_position": best_3d_position,
                            "last_seen": time.time()
                        }
                    )
            
            # Update Kalman filter
            # (Kalman filter update code here...)
            
            self.tracking_lost_counter = 0
            self.last_seen_position = best_match
            self.target_3d_position = best_3d_position
            return best_match, best_3d_position
            
        elif self.last_seen_position is not None:
            # Increment lost counter if no match found
            self.tracking_lost_counter += 1
            
            # If lost for too many frames, consider the person gone
            if self.tracking_lost_counter > self.max_tracking_lost:
                # Deactivate but don't delete - allows reidentification
                # even after person leaves scene
                self.vector_gallery.deactivate_target(self.current_target_id)
                
                self.current_target_id = None
                self.last_seen_position = None
                self.target_3d_position = None
                
            return self.last_seen_position, self.target_3d_position
            
        return None, None
    
    def identify_person(self, frame, box, position_3d=None):
        """Identify an arbitrary person in the frame"""
        features = self.extract_features(frame, box)
        if features is not None:
            # Try to match with any known person
            match_id, similarity, metadata = self.vector_gallery.find_match(
                features, limit=3, active_only=False
            )
            
            if similarity > self.similarity_threshold:
                # Reactivate if inactive
                if metadata and not metadata.get("active", True):
                    self.vector_gallery.update_target(
                        match_id, 
                        features,
                        metadata_update={
                            "active": True,
                            "reactivated_at": time.time(),
                            "last_position": position_3d
                        }
                    )
                
                return match_id, similarity, metadata
        
        return None, 0.0, None
    
    def close(self):
        """Close camera and release resources"""
        # Close ZED camera
        # (ZED close code here...)
        
        # Close vector gallery
        self.vector_gallery.close()
