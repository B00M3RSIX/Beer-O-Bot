
# ROS2 Workspace Structure for Person-Following Robot

## Directory Layout

follower_ws/
├── src/
│   ├── follower_bringup/             # Launch files and configuration
│   │   ├── launch/                   # Launch files for different configurations
│   │   │   ├── follower_complete.launch.py      # All nodes
│   │   │   ├── perception_only.launch.py        # Camera and detection only
│   │   │   ├── navigation_only.launch.py        # Navigation stack only
│   │   │   └── simulation.launch.py             # Gazebo simulation
│   │   ├── config/                   # Configuration files
│   │   │   ├── nav2_params.yaml              # Navigation parameters
│   │   │   ├── zed2_params.yaml              # ZED2 camera parameters
│   │   │   ├── lidar_params.yaml             # Lidar parameters
│   │   │   ├── voice_params.yaml             # Voice recognition parameters
│   │   │   └── gesture_detection_params.yaml # Gesture detection parameters
│   │   ├── rviz/                     # RViz configuration
│   │   └── CMakeLists.txt & package.xml
│   │
│   ├── follower_msgs/                # Custom message definitions
│   │   ├── msg/
│   │   │   ├── PersonTrackingStatus.msg       # Tracking status message
│   │   │   ├── FollowCommand.msg              # Follow command message
│   │   │   ├── PersonIdentity.msg             # Person identity message
│   │   │   ├── VoiceCommand.msg               # Voice command message
│   │   │   └── GestureDetection.msg           # Gesture detection message
│   │   ├── srv/
│   │   │   ├── VectorDBQuery.srv              # Vector DB query service
│   │   │   ├── VectorDBUpdate.srv             # Vector DB update service
│   │   │   └── AuthorizationCheck.srv         # Authorization check service
│   │   └── CMakeLists.txt & package.xml
│   │
│   ├── follower_perception/          # Perception stack
│   │   ├── include/follower_perception/
│   │   │   ├── zed_wrapper.hpp               # ZED2 camera wrapper
│   │   │   ├── person_detector.hpp           # Person detector
│   │   │   ├── reid_tracker.hpp              # Re-ID tracker
│   │   │   └── gesture_detector.hpp          # Gesture detector
│   │   ├── src/
│   │   │   ├── zed_node.cpp                  # ZED2 camera node
│   │   │   ├── person_detector_node.cpp      # Person detector node
│   │   │   ├── reid_tracker_node.cpp         # Re-ID tracker node
│   │   │   └── gesture_detector_node.cpp     # Hand-raising detector
│   │   ├── models/                   # Neural network models
│   │   │   ├── mobilenetv3_reid.engine       # TensorRT engine file
│   │   │   ├── gesture_detection.engine      # Gesture detection model
│   │   │   └── model_converter.py            # Script to convert models
│   │   └── CMakeLists.txt & package.xml
│   │
│   ├── follower_navigation/          # Navigation stack
│   │   ├── include/follower_navigation/
│   │   │   ├── person_following.hpp          # Person following behavior
│   │   │   ├── costmap_layers.hpp            # Custom costmap layers
│   │   │   └── recovery_behaviors.hpp        # Custom recovery behaviors
│   │   ├── src/
│   │   │   ├── person_following_node.cpp     # Person following node
│   │   │   ├── costmap_fusion_node.cpp       # Costmap fusion node
│   │   │   └── behavior_plugins/             # Nav2 behavior plugins
│   │   ├── behavior_trees/           # Custom behavior trees
│   │   │   ├── follow_person.xml             # Person following BT
│   │   │   ├── search_person.xml             # Person searching BT
│   │   │   └── rotate_to_direction.xml       # Rotation behavior BT
│   │   └── CMakeLists.txt & package.xml
│   │
│   ├── follower_vector_db/           # Vector database
│   │   ├── include/follower_vector_db/
│   │   │   ├── vector_database.hpp           # Vector DB interface
│   │   │   ├── qdrant_wrapper.hpp            # Qdrant wrapper
│   │   │   └── authorization.hpp             # Authorization manager
│   │   ├── src/
│   │   │   ├── vector_db_node.cpp            # Vector DB node
│   │   │   ├── db_service.cpp                # DB service implementation
│   │   │   └── authorization_service.cpp     # Authorization service
│   │   └── CMakeLists.txt & package.xml
│   │
│   ├── follower_voice/               # Voice command integration
│   │   ├── include/follower_voice/
│   │   │   ├── respeaker_wrapper.hpp         # ReSpeaker wrapper
│   │   │   ├── voice_command_processor.hpp   # Command processor
│   │   │   ├── command_verification.hpp      # Command verification
│   │   │   └── german_tts.hpp                # German TTS module
│   │   ├── src/
│   │   │   ├── respeaker_node.cpp            # ReSpeaker node
│   │   │   ├── voice_command_node.cpp        # Voice command processing
│   │   │   ├── command_verification_node.cpp # Command verification
│   │   │   └── command_bt_nodes.cpp          # BT nodes for commands
│   │   ├── config/
│   │   │   ├── command_patterns.yaml         # German command patterns
│   │   │   ├── wake_word_german.ppn          # German wake word model
│   │   │   ├── speech_recognition_german.rhn # German speech recognition
│   │   │   └── tts_german_voice.xml          # TTS voice configuration
│   │   ├── behavior_trees/           # Voice command BTs
│   │   │   ├── voice_command_processing.xml  # Main command processing
│   │   │   └── command_verification.xml      # Verification steps
│   │   └── CMakeLists.txt & package.xml
│   │
│   ├── follower_microros_bridge/     # micro-ROS bridge
│   │   ├── include/follower_microros_bridge/
│   │   │   └── microros_bridge.hpp
│   │   ├── src/
│   │   │   └── microros_bridge_node.cpp
│   │   └── CMakeLists.txt & package.xml
│   │
│   └── follower_simulation/          # Simulation support
│       ├── launch/
│       │   └── gazebo_world.launch.py
│       ├── worlds/
│       │   ├── indoor_environment.world
│       │   ├── outdoor_environment.world
│       │   └── voice_testing_environment.world
│       ├── models/
│       │   ├── follower_robot/
│       │   └── respeaker_mic_array/
│       └── CMakeLists.txt & package.xml

## Package Dependencies

### follower_bringup
- follower_perception
- follower_navigation
- follower_vector_db
- follower_voice
- follower_microros_bridge
- nav2_bringup
- zed_ros2_wrapper
- rplidar_ros

### follower_msgs
- std_msgs
- geometry_msgs
- builtin_interfaces
- action_msgs

### follower_perception
- follower_msgs
- zed_ros2_interfaces
- image_transport
- cv_bridge
- tensorrt_ros
- vision_msgs

### follower_navigation
- follower_msgs
- nav2_msgs
- nav2_core
- nav2_behavior_tree
- nav2_costmap_2d

### follower_vector_db
- follower_msgs
- rclcpp_components
- rclcpp_action

### follower_voice
- audio_common_msgs
- sound_play
- respeaker_ros
- nav2_behavior_tree
- std_srvs
- follower_msgs

### follower_microros_bridge
- micro_ros_msgs
- geometry_msgs
- std_msgs
- sensor_msgs

### follower_simulation
- gazebo_ros
- follower_msgs
- nav2_gazebo_spawner

