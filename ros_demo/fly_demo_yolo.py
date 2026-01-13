#!/usr/bin/env python3

import sys
import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

# ROS Msg Imports
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleLocalPosition, VehicleStatus
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# AI Imports
import cv2
from ultralytics import YOLO

class OffboardControl(Node):
    def __init__(self, north, east, down) -> None:
        super().__init__('offboard_control_yolo')
        
        # --- Configuration ---
        self.takeoff_height = -5.0
        self.acceptance_radius = 0.3
        self.transit_velocity = 2.0 
        
        # Visual Servoing Config
        self.target_height_ratio = 0.5  # Approximate 2m distance (Person fills 75% height)
        self.kp_yaw = 0.005                # Turn speed gain
        self.kp_dist = 0.01               # Forward speed gain
        self.max_yaw_rate = 0.01          # Rad/s limit
        self.max_fwd_vel = 0.05           # m/s limit

        # Final Target Waypoint (User Input)
        self.target_n = float(north)
        self.target_e = float(east)
        self.target_d = float(down)

        # Current State
        self.setpoint_n = 0.0
        self.setpoint_e = 0.0
        self.setpoint_d = 0.0
        self.mission_state = "TAKEOFF"  # States: TAKEOFF, TRANSIT, HOVER, TRACKING
        self.last_detection_time = 0

        # --- AI Setup ---
        self.bridge = CvBridge()
        # Load YOLOv8 Small model
        self.get_logger().info("Loading YOLOv8 Model (this may take a moment)...")
        self.model = YOLO("yolov8s.pt") 
        self.get_logger().info("YOLOv8 Model Loaded")

        # --- QoS Setup ---
        qos_command = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        qos_sensor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE, 
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # --- Publishers ---
        self.offboard_control_mode_publisher = self.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode', qos_command)
        self.trajectory_setpoint_publisher = self.create_publisher(
            TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos_command)
        self.vehicle_command_publisher = self.create_publisher(
            VehicleCommand, '/fmu/in/vehicle_command', qos_command)
        
        # Debug Image Publisher (View this in rqt_image_view)
        self.debug_image_publisher = self.create_publisher(
            Image, '/debug/image_labeled', 10)
        
        # --- Subscribers ---
        self.vehicle_local_position_subscriber = self.create_subscription(
            VehicleLocalPosition, '/fmu/out/vehicle_local_position_v1', self.vehicle_local_position_callback, qos_sensor)
        self.vehicle_status_subscriber = self.create_subscription(
            VehicleStatus, '/fmu/out/vehicle_status', self.vehicle_status_callback, qos_sensor)
        self.image_subscriber = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, qos_sensor)

        # --- Variables ---
        self.vehicle_local_position = VehicleLocalPosition()
        self.vehicle_status = VehicleStatus()
        self.offboard_setpoint_counter = 0
        self.tracking_velocity = [0.0, 0.0, 0.0] # vx, vy, vz
        self.tracking_yaw_rate = 0.0

        # Timer runs at 10Hz
        self.dt = 0.1 
        self.timer = self.create_timer(self.dt, self.timer_callback)
        
        self.get_logger().info(f"Mission: Fly to ({self.target_n}, {self.target_e}) and Scan for People")

    # ----------------------------------------------------------------
    # CALLBACKS
    # ----------------------------------------------------------------
    def vehicle_local_position_callback(self, msg):
        self.vehicle_local_position = msg

    def vehicle_status_callback(self, msg):
        self.vehicle_status = msg

    def image_callback(self, msg):
        # Skip processing during takeoff to save resources
        if self.mission_state == "TAKEOFF":
            return

        try:
            # 1. Convert ROS Image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            height, width, _ = cv_image.shape

            # 2. Run YOLO Inference
            results = self.model(cv_image, verbose=False)
            
            person_detected = False
            target_box = None

            # 3. Process Detections
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    cls = int(box.cls[0])
                    
                    # Check if it is a Person (Class 0)
                    if cls == 0:
                        person_detected = True
                        
                        # Get Box Coordinates (x1, y1, x2, y2)
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        # Calculate Center and Size
                        b_w = x2 - x1
                        b_h = y2 - y1
                        target_box = [x1 + b_w/2, y1 + b_h/2, b_w, b_h] # cx, cy, w, h

                        # --- VISUALIZATION ---
                        # Draw Red Rectangle
                        cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 0, 255), 3)
                        
                        # Add Label
                        conf = float(box.conf[0])
                        label = f"Person: {conf:.2f}"
                        cv2.putText(cv_image, label, (x1, y1 - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        
                        # We only track the first person found
                        break 
                if person_detected:
                    break

            # 4. Control Logic
            if person_detected:
                self.last_detection_time = self.get_clock().now().nanoseconds
                
                cx, cy, w, h = target_box
                
                # Calculate Visual Errors
                # X Error: -1 (Left) to +1 (Right)
                error_x = (width/2 - cx) / (width/2) 
                
                # Distance Error: Positive (Far) to Negative (Too Close)
                current_ratio = h / height
                error_dist = self.target_height_ratio - current_ratio

                # Draw Status on Screen
                status_text = f"TRACKING | ErrX: {error_x:.2f} | DistErr: {error_dist:.2f}"
                cv2.putText(cv_image, status_text, (20, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Transition State
                if self.mission_state != "TRACKING":
                    self.get_logger().info("PERSON DETECTED: Drone control taking over!")
                    self.mission_state = "TRACKING"

                # Compute Velocities (P-Controller)
                # Yaw Rate (Turn to center X)
                self.tracking_yaw_rate = float(self.kp_yaw * error_x)
                self.tracking_yaw_rate = np.clip(self.tracking_yaw_rate, -self.max_yaw_rate, self.max_yaw_rate)

                # Forward Velocity (Approach target)
                fwd_vel = float(self.kp_dist * error_dist)
                fwd_vel = np.clip(fwd_vel, -self.max_fwd_vel, self.max_fwd_vel)
                
                # Deadband: Stop if very close to target distance
                if abs(error_dist) < 0.05: 
                    fwd_vel = 0.0
                
                # Convert Body Frame Velocity to Global NED Frame
                # (Simple 2D rotation)
                current_yaw = self.vehicle_local_position.heading
                vel_n = fwd_vel * math.cos(current_yaw)
                vel_e = fwd_vel * math.sin(current_yaw)
                self.tracking_velocity = [vel_n, vel_e, 0.0]

            else:
                # Logic when target is lost
                now = self.get_clock().now().nanoseconds
                # If lost for > 2 seconds, stop tracking
                if self.mission_state == "TRACKING" and (now - self.last_detection_time) > 2e9:
                    self.get_logger().info("Target Lost. Hovering.")
                    self.mission_state = "HOVER"
                    # Capture current pos to hover there
                    self.setpoint_n = self.vehicle_local_position.x
                    self.setpoint_e = self.vehicle_local_position.y
                    self.setpoint_d = self.vehicle_local_position.z
                    
                    cv2.putText(cv_image, "TARGET LOST - HOVERING", (20, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # 5. Publish Debug Image
            out_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding="bgr8")
            self.debug_image_publisher.publish(out_msg)

        except Exception as e:
            self.get_logger().error(f"CV Error: {e}")

    # ----------------------------------------------------------------
    # COMMAND PUBLISHERS
    # ----------------------------------------------------------------
    def arm(self):
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)
        self.get_logger().info('Arming...')

    def engage_offboard_mode(self):
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)
        self.get_logger().info("Switching to Offboard Mode...")

    def publish_offboard_control_heartbeat_signal(self):
        msg = OffboardControlMode()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        
        if self.mission_state == "TRACKING":
            # Velocity Control Mode
            msg.position = False
            msg.velocity = True
            msg.acceleration = False
        else:
            # Position Control Mode
            msg.position = True
            msg.velocity = False
            msg.acceleration = False
            
        self.offboard_control_mode_publisher.publish(msg)

    def publish_position_setpoint(self, x, y, z):
        msg = TrajectorySetpoint()
        msg.position = [x, y, z]
        msg.yaw = 1.57079 # Face East default
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.trajectory_setpoint_publisher.publish(msg)

    def publish_velocity_setpoint(self, vx, vy, vz, yaw_rate):
        msg = TrajectorySetpoint()
        msg.position = [float('nan'), float('nan'), float('nan')] # NaN tells PX4 to ignore position
        msg.velocity = [vx, vy, vz]
        msg.yawspeed = yaw_rate
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.trajectory_setpoint_publisher.publish(msg)

    def publish_vehicle_command(self, command, **params) -> None:
        msg = VehicleCommand()
        msg.command = command
        msg.param1 = params.get("param1", 0.0)
        msg.param2 = params.get("param2", 0.0)
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.vehicle_command_publisher.publish(msg)

    # ----------------------------------------------------------------
    # MAIN LOOP
    # ----------------------------------------------------------------
    def timer_callback(self):
        self.publish_offboard_control_heartbeat_signal()

        # Startup Logic
        if self.offboard_setpoint_counter == 10:
            self.engage_offboard_mode()
            self.arm()
        if self.offboard_setpoint_counter < 11:
            self.offboard_setpoint_counter += 1
            return 

        current_z = self.vehicle_local_position.z

        # --- STATE MACHINE ---
        
        if self.mission_state == "TAKEOFF":
            if self.setpoint_d > self.takeoff_height:
                self.setpoint_d -= self.transit_velocity * self.dt
            self.publish_position_setpoint(0.0, 0.0, self.setpoint_d)
            
            if current_z <= (self.takeoff_height + self.acceptance_radius):
                self.get_logger().info("Takeoff Complete. Starting Transit...")
                self.setpoint_d = self.takeoff_height
                self.mission_state = "TRANSIT"

        elif self.mission_state == "TRANSIT":
            # Move towards target waypoint
            dn = self.target_n - self.setpoint_n
            de = self.target_e - self.setpoint_e
            dd = self.target_d - self.setpoint_d
            distance = math.sqrt(dn**2 + de**2 + dd**2)
            step_size = self.transit_velocity * self.dt

            if distance < step_size:
                self.setpoint_n = self.target_n
                self.setpoint_e = self.target_e
                self.setpoint_d = self.target_d
                self.mission_state = "HOVER"
                self.get_logger().info("Waypoint Reached. Hovering and Scanning...")
            else:
                self.setpoint_n += (dn / distance) * step_size
                self.setpoint_e += (de / distance) * step_size
                self.setpoint_d += (dd / distance) * step_size
            
            self.publish_position_setpoint(self.setpoint_n, self.setpoint_e, self.setpoint_d)

        elif self.mission_state == "HOVER":
            # Hold Position, Wait for Camera Callback to detect person
            self.publish_position_setpoint(self.target_n, self.target_e, self.target_d)

        elif self.mission_state == "TRACKING":
            # Visual Servoing Mode (Velocity Control)
            # Maintain altitude using simple P-control
            alt_error = self.takeoff_height - self.vehicle_local_position.z
            vz = 0.5 * alt_error 
            
            self.publish_velocity_setpoint(
                self.tracking_velocity[0], # North
                self.tracking_velocity[1], # East
                vz,                        # Down (Altitude Hold)
                self.tracking_yaw_rate     # Yaw Speed
            )

        self.offboard_setpoint_counter += 1

def main(args=None):
    if len(sys.argv) != 4:
        print("Usage: python3 fly_drone_yolo.py <N> <E> <D>")
        return
    rclpy.init(args=args)
    offboard_control = OffboardControl(sys.argv[1], sys.argv[2], sys.argv[3])
    try:
        rclpy.spin(offboard_control)
    except KeyboardInterrupt:
        pass
    finally:
        offboard_control.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()