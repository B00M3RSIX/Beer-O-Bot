# Voice Command Integration with ReSpeaker Mic Array v2.0

## 1. Hardware Integration

### 1.1 ReSpeaker Mic Array v2.0
- 4-microphone circular array with LED ring
- Far-field voice capture capability (up to 5 meters)
- Direction of Arrival (DoA) estimation
- Beamforming for noise reduction
- USB connectivity with the Jetson Orin NX

### 1.2 Physical Setup
- Mount the ReSpeaker array on top of the robot
- Position for optimal audio capture (unobstructed by other components)
- Connect via USB to the Jetson Orin NX
- Power through USB bus or optional separate 5V supply

## 2. Software Components

### 2.1 ReSpeaker ROS2 Driver
- Fork and port of the [respeaker_ros](https://github.com/furushchev/respeaker_ros) package to ROS2
- Provides audio streaming, DoA estimation, and LED control
- Publishes raw audio data and processed audio features
- Integration with ALSA for audio capture

### 2.2 Voice Command Processor
- Lightweight keyword spotting using [Porcupine](https://github.com/Picovoice/porcupine) for wake word detection
- Command recognition using pattern matching or [Rhino](https://github.com/Picovoice/rhino) for intent recognition
- Voice activity detection to filter out silence
- **German language as default** with optional multilingual support
- Custom German language model for command recognition
- Adapting wake word detection for German pronunciation

### 2.3 Speech Feedback
- Text-to-speech feedback using `sound_play` package with German voice
- Audio cues for command recognition
- Configurable voice and volume settings
- LED feedback on the ReSpeaker array to indicate listening states

## 3. ROS2 Interface

### 3.1 Published Topics
- `/respeaker/audio` (audio_common_msgs/AudioData): Raw audio stream
- `/respeaker/direction` (std_msgs/Int32): Direction of arrival in degrees
- `/respeaker/sound_localization` (geometry_msgs/PoseStamped): Sound source 3D position
- `/respeaker/speech_detection` (std_msgs/Bool): Voice activity detection
- `/follower_voice/recognized_command` (follower_msgs/VoiceCommand): Recognized voice commands

### 3.2 Subscribed Topics
- `/follower_voice/feedback` (std_msgs/String): Text messages to be spoken
- `/respeaker/set_led_pattern` (std_msgs/String): Control LED ring patterns

### 3.3 Services
- `/follower_voice/set_active_commands` (std_srvs/SetParameters): Configure active command patterns
- `/follower_voice/calibrate_noise` (std_srvs/Trigger): Calibrate background noise profile

## 4. Command Set (German Default)

### 4.1 Basic Control Commands
- "Folge mir" - Begin following the person who issued the command (requires identity confirmation)
- "Stopp" / "Halt" - Emergency stop
- "Nicht mehr folgen" - Stop the following behavior
- "Weiter" / "Fortsetzen" - Resume previous behavior
- "Komm näher" / "Weiter weg" - Adjust following distance

### 4.2 Configuration Commands
- "Folge dichter" / "Mehr Abstand" - Change default following distance
- "Folge schneller" / "Folge langsamer" - Adjust maximum speed
- "Folge links" / "Folge rechts" - Change position preference

### 4.3 Navigation Commands
- "Geh nach Hause" / "Zur Basis" - Return to home position
- "Warte hier" - Stay in current position
- "Folge [Name]" - Follow a specific known person (requires named identity)

### 4.4 System Commands
- "Manuelle Steuerung" - Switch to joystick or remote control
- "Batteriestatus" - Report battery level
- "Systemstatus" - Report overall system status
- "Speicher mich als [Name]" - Associate current person with a name

## 5. Implementation Details

### 5.1 Command Recognition Pipeline
1. Continuous audio capture from ReSpeaker
2. Wake word detection ("Hey Roboter" or customizable German wake word)
3. Voice activity detection to identify command boundaries
4. Feature extraction for command audio
5. Pattern matching against predefined German command templates
6. Confidence scoring and threshold-based acceptance
7. Command execution and feedback

### 5.2 Integration with Person Following
- DoA information helps identify direction to search for command issuer
- Voice source localization can supplement visual tracking
- Command recognition triggers behavior tree transitions
- Voice feedback confirms robot's understanding and actions

### 5.3 Command-Person Association & Confirmation
- **Person Identification Requirement**: Robot only follows persons with associated names in database
- **Visual Confirmation Protocol**:
  1. When "Folge mir" command is received, the system checks if person is in camera frame
  2. If visible: Robot requests hand-raising confirmation ("Bitte hebe deine Hand")
  3. If not visible: Robot rotates toward sound source using Nav2 goal based on DoA
  4. After rotation, requests visual identification ("Wer hat den Befehl gegeben?")
  5. Person must be recognized and have a name in database to activate following
  6. If multiple people raise hands, robot asks for verbal confirmation of name
- **Confirmation Timeout**: If no confirmation within 10 seconds, command is canceled
- **LED Feedback**: Visual indicator of confirmation state (searching, confirmed, denied)
- **Voice Feedback**: Clear instructions and confirmation of successful association

### 5.4 Action Planning for Command Processing
- **State Machine Implementation**: Command processing follows a structured workflow
- **Behavior Tree Integration**: Custom behavior tree nodes for voice command handling
- **Sequential Action Planning**:
  ```
  1. Command Detection (Wake word + Command)
  2. Command Classification & Parameter Extraction
  3. Identity Verification Requirement Check
    a. If no identity verification needed → Execute Command
    b. If identity verification needed → Continue
  4. Source Localization
    a. Check if source is in camera frame
    b. If not in frame → Rotate to sound source direction
  5. Identity Confirmation
    a. Request gesture confirmation (hand raising)
    b. Run gesture detection algorithm
    c. Match detected person with vector database
    d. Verify person has associated name
  6. Command Authorization
    a. If person verified and authorized → Execute Command
    b. If verification failed → Reject with feedback
  7. Command Execution
    a. Execute command via Nav2 behavior trees
    b. Provide feedback on execution status
  ```
- **Parallel Processing**: Audio processing continues during visual confirmation
- **Timeout Handling**: Defined timeouts for each phase with graceful degradation

### 5.3 Performance Optimization
- Offload voice processing to a separate CPU core
- Local processing of wake word detection to reduce latency
- On-device command recognition without cloud dependencies
- Efficient voice activity detection to minimize processing

### 5.4 Configuration Options
- Customizable wake word
- Adjustable command recognition thresholds
- Noise profile adaptation for different environments
- Voice feedback volume control

## 6. Authorization and Security

### 6.1 Named Person Requirements
- Only persons with associated names in the database can issue following commands
- New persons must be explicitly registered with "Speicher mich als [Name]" command
- Registration requires authorization from an existing authorized person
- Database maintains authorization levels for different persons

### 6.2 Security Measures
- Voice print verification as optional additional security layer
- Command blackout period after failed authorization attempts
- Restricted commands based on authorization level
- Logging of all command attempts for security auditing

### 6.3 Privacy Considerations
- Local processing of all audio data (no cloud dependencies)
- Audio data not stored except during active command processing
- Clear visual indication when listening (LED feedback)
- Option to temporarily disable voice recognition

## 7. Future Enhancements
- Speaker identification to personalize responses
- Context-aware command interpretation
- Multi-language support with language switching
- More sophisticated NLU for complex commands
- Gesture command recognition to complement voice commands
