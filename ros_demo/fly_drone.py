#!/usr/bin/env python3

import sys
import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleLocalPosition, VehicleStatus

class OffboardControl(Node):
    def __init__(self, north, east, down) -> None:
        super().__init__('offboard_control_smooth')
        
        # --- Configuration ---
        self.takeoff_height = -5.0
        self.acceptance_radius = 0.3
        self.velocity = 2.0  # Speed in m/s (Adjust this to go faster/slower)
        
        # Final Target Waypoint
        self.target_n = float(north)
        self.target_e = float(east)
        self.target_d = float(down)

        # Current "Virtual" Setpoint (Starts at 0,0,0)
        # We will move this point slowly towards the target
        self.setpoint_n = 0.0
        self.setpoint_e = 0.0
        self.setpoint_d = 0.0

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

        # --- Communication ---
        self.offboard_control_mode_publisher = self.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode', qos_command)
        self.trajectory_setpoint_publisher = self.create_publisher(
            TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos_command)
        self.vehicle_command_publisher = self.create_publisher(
            VehicleCommand, '/fmu/in/vehicle_command', qos_command)
        
        self.vehicle_local_position_subscriber = self.create_subscription(
            VehicleLocalPosition, '/fmu/out/vehicle_local_position_v1', self.vehicle_local_position_callback, qos_sensor)
        self.vehicle_status_subscriber = self.create_subscription(
            VehicleStatus, '/fmu/out/vehicle_status', self.vehicle_status_callback, qos_sensor)

        # --- Variables ---
        self.vehicle_local_position = VehicleLocalPosition()
        self.vehicle_status = VehicleStatus()
        self.offboard_setpoint_counter = 0
        self.mission_state = "TAKEOFF" 

        # Timer runs at 10Hz (0.1 seconds)
        self.dt = 0.1 
        self.timer = self.create_timer(self.dt, self.timer_callback)
        
        self.get_logger().info(f"Mission: Smooth Fly to ({self.target_n}, {self.target_e}, {self.target_d}) at {self.velocity} m/s")

    def vehicle_local_position_callback(self, msg):
        self.vehicle_local_position = msg

    def vehicle_status_callback(self, msg):
        self.vehicle_status = msg

    def arm(self):
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)
        self.get_logger().info('Arming...')

    def engage_offboard_mode(self):
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)
        self.get_logger().info("Switching to Offboard Mode...")

    def publish_offboard_control_heartbeat_signal(self):
        msg = OffboardControlMode()
        msg.position = True
        msg.velocity = False
        msg.acceleration = False
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_control_mode_publisher.publish(msg)

    def publish_position_setpoint(self, x, y, z):
        msg = TrajectorySetpoint()
        msg.position = [x, y, z]
        msg.yaw = 1.57079 # Face East
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

    def timer_callback(self):
        self.publish_offboard_control_heartbeat_signal()

        # Startup Logic
        if self.offboard_setpoint_counter == 10:
            self.engage_offboard_mode()
            self.arm()
        if self.offboard_setpoint_counter < 11:
            self.offboard_setpoint_counter += 1
            return 

        # --- Mission State Machine ---
        current_z = self.vehicle_local_position.z

        if self.mission_state == "TAKEOFF":
            # Slowly move setpoint UP to -5.0
            if self.setpoint_d > self.takeoff_height:
                self.setpoint_d -= self.velocity * self.dt # Move up (negative Z)
            
            self.publish_position_setpoint(0.0, 0.0, self.setpoint_d)
            
            # Check if drone reached the height
            if current_z <= (self.takeoff_height + self.acceptance_radius):
                self.get_logger().info("Takeoff Complete. Starting Transit...")
                self.setpoint_n = 0.0
                self.setpoint_e = 0.0
                self.setpoint_d = self.takeoff_height # Snap to exact height
                self.mission_state = "TRANSIT"

        elif self.mission_state == "TRANSIT":
            # 1. Calculate Vector to Target
            dn = self.target_n - self.setpoint_n
            de = self.target_e - self.setpoint_e
            dd = self.target_d - self.setpoint_d
            distance = math.sqrt(dn**2 + de**2 + dd**2)

            # 2. Move Setpoint towards Target
            step_size = self.velocity * self.dt

            if distance < step_size:
                # We are close enough to snap to target
                self.setpoint_n = self.target_n
                self.setpoint_e = self.target_e
                self.setpoint_d = self.target_d
                self.mission_state = "HOVER"
                self.get_logger().info("Target Reached. Hovering...")
            else:
                # Normalize and scale by velocity
                self.setpoint_n += (dn / distance) * step_size
                self.setpoint_e += (de / distance) * step_size
                self.setpoint_d += (dd / distance) * step_size
                
                if self.offboard_setpoint_counter % 20 == 0:
                    self.get_logger().info(f"Moving... Dist to Go: {distance:.2f}m")

            self.publish_position_setpoint(self.setpoint_n, self.setpoint_e, self.setpoint_d)

        elif self.mission_state == "HOVER":
            self.publish_position_setpoint(self.target_n, self.target_e, self.target_d)
            
        self.offboard_setpoint_counter += 1

def main(args=None):
    if len(sys.argv) != 4:
        print("Usage: python3 fly_drone.py <N> <E> <D>")
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