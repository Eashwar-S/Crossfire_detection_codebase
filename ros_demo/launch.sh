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