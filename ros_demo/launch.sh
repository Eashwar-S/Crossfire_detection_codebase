#!/bin/bash

# Define the World and Model variables
WORLD="baylands"
MODEL="x500_depth"

echo "Launching Simulation Stack in Separate Windows..."

# Window 1: QGroundControl
gnome-terminal --title="1. QGroundControl" -- bash -c "\
  echo 'Starting QGroundControl...'; \
  flatpak run --device=all org.mavlink.qgroundcontrol; \
  exec bash"

# Window 2: MicroXRCEAgent (Starts after 5s delay)
gnome-terminal --title="2. MicroXRCEAgent" -- bash -c "\
  echo 'Waiting 5s for QGC to initialize...'; \
  sleep 5; \
  echo 'Starting MicroXRCEAgent...'; \
  source /opt/ros/jazzy/setup.bash; \
  MicroXRCEAgent udp4 -p 8888; \
  exec bash"

# Window 3: PX4 Simulation (Starts after 8s delay)
gnome-terminal --title="3. PX4 Simulation" -- bash -c "\
  echo 'Waiting 8s for Agent to initialize...'; \
  sleep 8; \
  echo 'Starting PX4 SITL ($WORLD)...'; \
  cd ~/PX4-Autopilot; \
  PX4_GZ_WORLD=$WORLD PX4_GZ_MODEL=$MODEL make px4_sitl gz_${MODEL}; \
  exec bash"
  
gnome-terminal --title="4. ROS2 Parameter Bridge" -- bash -c "\
  echo 'Waiting 8s for Agent to initialize...'; \
  sleep 2; \
  cd; \
  ros2 run ros_gz_bridge parameter_bridge \
  --ros-args \
  -p config_file:=/home/crossfire/Crossfire_detection_codebase/ros_demo/gz_camera.yaml \
  -r /world/baylands/model/x500_depth_0/link/camera_link/sensor/IMX214/image:=/camera/image_raw \
  -r /world/baylands/model/x500_depth_0/link/camera_link/sensor/IMX214/camera_info:=/camera/camera_info; \
  exec bash"

gnome-terminal --title="5. Drone control script with detection" -- bash -c "\
  echo 'Waiting 8s for Agent to initialize...'; \
  sleep 20; \
  cd /home/crossfire/Crossfire_detection_codebase/ros_demo; \
  python3 fly_demo_yolo.py 33 0 -22; \
  exec bash"
  

#   gz service -s /world/baylands/set_pose --reqtype gz.msgs.Pose --reptype gz.msgs.Boolean --timeout 300 --req "name: 'person_walking', position: {x: 25, y: 25, z: 0.6}"
# data: true

