# Person Re-ID Robot Following System Specification

## 1. System Overview

### 1.1 Purpose
A person-following robot system that uses re-identification (Re-ID) technology to track and follow a specific person, even after temporary loss of visual contact. The system operates on a 4-wheeled skid-drive robot platform with ROS2 Humble.

### 1.2 Hardware Components
- **Robot Base**: 4-wheeled platform with skid-drive kinematics
- **Microcontroller**: Teensy running micro-ROS for low-level control
- **Main Computer**: Jetson Orin NX for perception and navigation
- **Sensors**:
  - ZED2 Stereolabs camera (RGB + depth)
  - RP-Lidar A2 for obstacle detection
  - IMU for orientation and movement tracking

### 1.3 Software Architecture
- **ROS2 Humble** as the core robotics framework
- **micro-ROS** for Teensy-Jetson communication
- **Nav2** for layered navigation and path planning
- **TensorRT** for neural network acceleration
- **Qdrant** vector database for persistent identity management

## 2. Functional Requirements

### 2.1 Person Re-Identification
- Select a person in a frame to track without pre-training
- Extract and store visual features in a vector database
- Match the person across frames even with changing appearance
- Re-identify the person after temporary visual loss
- Track the person in 3D space using stereo camera data

### 2.2 Person Following
- Maintain a configurable following distance from the target
- Follow the person through complex environments
- Avoid obstacles while maintaining visual contact
- Recover tracking after person leaves and re-enters field of view
- Provide smooth movement without erratic changes in direction

### 2.3 Navigation & Obstacle Avoidance
- Create and maintain a local costmap from ZED2 and Lidar data
- Use Nav2 for path planning and obstacle avoidance
- Implement "social navigation" parameters (maintain personal space)
- Handle dynamic obstacles in the environment
- Safely navigate outdoor terrain

### 2.4 Multi-Purpose Camera Utilization
- Balance Re-ID processing with environment mapping
- Create point cloud data for obstacle detection
- Support costmap generation for navigation
- Optimize resource allocation based on current tasks

### 2.5 Voice Command Interface
- Detect and recognize voice commands from ReSpeaker microphone array
- Use German as the default language for all voice interactions
- Control following behavior through natural language commands
- Provide audio feedback for command recognition
- Determine direction of voice commands for better person association
- Support gesture-based confirmation for critical commands
- Restrict following behavior to named persons only
- Implement multi-step confirmation for security and accuracy

## 3. System Architecture

### 3.1 ROS2 Node Structure
- **`zed_perception_node`**: Camera data processing and point cloud generation
- **`reid_tracking_node`**: Person detection and re-identification
- **`vector_db_node`**: Identity database management service
- **`person_following_node`**: Convert tracking to navigation commands
- **`local_costmap_node`**: Generate costmap for navigation
- **`nav2_controller`**: Handle path planning and execution
- **`microros_bridge_node`**: Communication with Teensy controller
- **`respeaker_node`**: Process audio from ReSpeaker microphone array
- **`voice_command_node`**: Interpret voice commands for system control

### 3.2 Message Flow
1. ZED2 camera provides RGB and depth data
2. Re-ID system identifies and tracks the target person
3. Vector database maintains identity persistence
4. ReSpeaker captures voice commands in German
5. Voice commands undergo verification process:
   - Direction of arrival determines where to look for speaker
   - Person must be identified in the database with a name
   - Gesture confirmation (hand raising) verifies command intent
   - Authorization check ensures person can issue the command
6. Person following node generates dynamic goals based on target position
7. Nav2 generates path plans that avoid obstacles
8. Command velocities are sent to the Teensy via micro-ROS
9. Status and sensor data flow back from Teensy to Jetson
10. Voice feedback confirms command recognition and execution

### 3.3 Data Storage
- **Vector Database**: Persistent storage of identity features
- **Local Costmap**: Temporary mapping of obstacles
- **Parameter Storage**: Configurable parameters for system behavior

### 3.4 ROS2 Workspace Structure
The system is organized in a modular workspace structure:
- **follower_bringup**: Launch files and configuration
- **follower_msgs**: Custom message definitions
- **follower_perception**: ZED2 integration and Re-ID tracking
- **follower_navigation**: Person following and obstacle avoidance
- **follower_vector_db**: Vector database for identity management
- **follower_voice**: ReSpeaker integration and voice command processing
- **follower_microros_bridge**: Communication with Teensy
- **follower_simulation**: Gazebo simulation support

## 4. Technical Implementation

### 4.1 Person Re-ID Module
- **Feature Extractor**: MobileNetV3-based architecture with TensorRT optimization
- **Vector Dimension**: 256-dimensional feature vectors (configurable)
- **Gallery Management**: Combination of in-memory cache and persistent storage
- **Matching Algorithm**: Cosine similarity with threshold-based decision
- **3D Tracking**: Kalman filter for position prediction

### 4.2 Vector Database Integration
- **Database**: Qdrant embedded mode for optimal edge deployment
- **Storage Structure**: Target IDs mapped to feature collections
- **Query Capability**: Similarity search with metadata filtering
- **Persistence**: On-disk storage with recovery capability
- **Memory Management**: Two-tier architecture (RAM + disk)

### 4.3 Navigation System
- **Costmap Generation**: Fusion of ZED2 depth and Lidar data
- **Controller**: Dynamic goal generation based on person position
- **Behavior Trees**: Custom Nav2 behaviors for person following
- **Recovery Actions**: Specific actions when person is temporarily lost
- **Parameter Tuning**: Outdoor-specific navigation parameters

### 4.4 Communication Interface
- **micro-ROS Messages**:
  - `geometry_msgs/Twist`: Command velocity
  - `sensor_msgs/Imu`: IMU data
  - `std_msgs/Int64MultiArray`: RoboClaw status
  - `std_msgs/UInt32`: Robot commands
- **High-Level Messages**:
  - `PersonTrackingStatus` (custom): Target tracking information with 3D position, confidence, and status
  - `FollowCommand` (custom): Configuration parameters for person following behavior
  - `PersonIdentity` (custom): Identity information for vector database management
  - `VoiceCommand` (custom): Recognized voice commands with metadata
  - `nav_msgs/OccupancyGrid`: Local costmap
  - `sensor_msgs/PointCloud2`: ZED2 point cloud
  - `audio_common_msgs/AudioData`: Raw audio data
- **Services**:
  - `VectorDBQuery` (custom): Query operations for the vector database
  - `VectorDBUpdate` (custom): Update operations for the vector database

## 5. Performance Considerations

### 5.1 Computational Optimization
- TensorRT model conversion for neural networks
- Parallel processing using CUDA streams
- Frame skipping for non-critical processing
- Resource allocation based on task priority

### 5.2 Memory Management
- Fixed-size vector gallery for bounded memory usage
- Temporal pruning of unnecessary features
- Efficient costmap representation for navigation

### 5.3 Power Efficiency
- Dynamic model precision (FP16/INT8) based on battery level
- Sensor duty cycling for power conservation
- Adaptive processing frequency

## 6. Implementation Phases

### 6.1 Phase 1: Basic Integration
- Package Re-ID code as ROS2 nodes
- Create ZED2 wrapper node
- Implement basic person following
- Set up vector database persistence

### 6.2 Phase 2: Navigation Integration
- Add Lidar data processing
- Create local costmap
- Implement Nav2-based obstacle avoidance
- Integrate with motion control

### 6.3 Phase 3: Advanced Features
- Multi-person tracking capability
- Outdoor robustness improvements
- Advanced behavior modes
- Performance optimization for battery life

## 7. Testing and Validation

### 7.1 Unit Testing
- Test individual components (Re-ID, vector DB, navigation)
- Verify accuracy of person identification
- Validate persistence across system restarts

### 7.2 Integration Testing
- Verify correct message flow between nodes
- Test combined camera and Lidar data fusion
- Validate micro-ROS communication reliability

### 7.3 Field Testing
- Indoor controlled environment testing
- Outdoor tests with varying lighting conditions
- Long-duration following tests for stability validation
- Multi-person discrimination tests

## 8. Future Expansion
- SLAM for global mapping capability
- Voice command interaction
- Multi-robot coordination for target tracking
- Gesture-based interaction with the target person
