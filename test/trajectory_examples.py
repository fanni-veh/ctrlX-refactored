#!/usr/bin/env python3
"""
ctrlX CORE Motor Trajectory Examples
Demonstrates various motion patterns and trajectories
"""

import time
from final_datalayer_api import CtrlXDataLayer
import math

class MotorTrajectories:
    """
    Motor controller with various trajectory examples
    """
    
    def __init__(self, ip, username, password, axis_name):
        """Initialize motor controller"""
        self.dl = CtrlXDataLayer(ip, username, password)
        self.axis_name = axis_name
        self.axis_base = f"motion/axs/{axis_name}"
        
        print(f"Initializing trajectory controller for {axis_name}...")
        
        if not self.dl.get_token():
            raise Exception("Failed to authenticate")
        
        print(f"✓ Ready for {axis_name}")
    
    def get_position(self):
        """Get current position"""
        data = self.dl.read_node(f"{self.axis_base}/state/values/actual/pos")
        return data.get('value')
    
    def get_velocity(self):
        """Get current velocity"""
        data = self.dl.read_node(f"{self.axis_base}/state/values/actual/vel")
        return data.get('value')
    
    def is_moving(self):
        """Check if axis is moving"""
        vel = abs(self.get_velocity())
        return vel > 0.01  # Threshold for considering "stopped"
    
    def enable_power(self):
        """Enable axis power"""
        print(f"Enabling power...")
        path = f"motion/axs/{self.axis_name}/cmd/power"
        self.dl.write_node(path, True, "bool8")
        time.sleep(0.5)
        print(f"✓ Power enabled")
    
    def move_absolute(self, position, velocity=1.0, acceleration=1.0, deceleration=1.0):
        """
        Move to absolute position
        
        Args:
            position: Target position (degrees or mm depending on axis config)
            velocity: Movement velocity (default 1.0)
            acceleration: Acceleration (default 1.0, max is ~2.0)
            deceleration: Deceleration (default 1.0, max is ~2.0)
        """
        cmd_path = f"motion/axs/{self.axis_name}/cmd/pos-abs"
        
        cmd = {
            "axsPos": str(position),
            "buffered": False,
            "lim": {
                "vel": str(velocity),
                "acc": str(acceleration),
                "dec": str(deceleration),
                "jrkAcc": "0",
                "jrkDec": "0"
            }
        }
        
        print(f"Moving to position {position} (vel={velocity}, acc={acceleration})")
        self.dl.write_node(cmd_path, cmd)
    
    def wait_for_stop(self, timeout=30, poll_interval=0.1):
        """
        Wait for motion to complete
        
        Args:
            timeout: Maximum time to wait (seconds)
            poll_interval: How often to check (seconds)
            
        Returns:
            True if stopped, False if timeout
        """
        start_time = time.time()
        
        while (time.time() - start_time) < timeout:
            if not self.is_moving():
                return True
            time.sleep(poll_interval)
        
        print("⚠ Motion timeout")
        return False
    
    def print_status(self):
        """Print current status"""
        pos = self.get_position()
        vel = self.get_velocity()
        print(f"  Position: {pos:.2f}, Velocity: {vel:.3f}")
    
    # ========== TRAJECTORY EXAMPLES ==========
    
    def trajectory_forward_backward(self, distance=20.0, velocity=2.0, cycles=3):
        """
        Simple forward and backward motion
        
        Args:
            distance: How far to move forward (default 20)
            velocity: Movement speed (default 2.0)
            cycles: Number of back-and-forth cycles (default 3)
        """
        print("\n" + "=" * 60)
        print(f"Trajectory: Forward-Backward ({cycles} cycles)")
        print("=" * 60)
        
        start_pos = self.get_position()
        print(f"Starting position: {start_pos:.2f}")
        
        for i in range(cycles):
            print(f"\nCycle {i+1}/{cycles}")
            
            # Move forward
            print("  → Moving forward...")
            self.move_absolute(start_pos + distance, velocity=velocity)
            self.wait_for_stop()
            self.print_status()
            time.sleep(0.5)
            
            # Move backward
            print("  ← Moving backward...")
            self.move_absolute(start_pos, velocity=velocity)
            self.wait_for_stop()
            self.print_status()
            time.sleep(0.5)
        
        print(f"\n✓ Completed {cycles} cycles")
    
    def trajectory_stepwise(self, step_size=5.0, num_steps=4, velocity=1.5, pause=1.0):
        """
        Stepwise motion with pauses
        
        Args:
            step_size: Size of each step (default 5)
            num_steps: Number of steps (default 4)
            velocity: Movement speed (default 1.5)
            pause: Pause between steps in seconds (default 1.0)
        """
        print("\n" + "=" * 60)
        print(f"Trajectory: Stepwise Motion ({num_steps} steps of {step_size})")
        print("=" * 60)
        
        start_pos = self.get_position()
        print(f"Starting position: {start_pos:.2f}")
        
        # Move forward in steps
        print("\nMoving forward in steps...")
        for i in range(num_steps):
            target = start_pos + (step_size * (i + 1))
            print(f"  Step {i+1}/{num_steps} → {target:.2f}")
            self.move_absolute(target, velocity=velocity)
            self.wait_for_stop()
            self.print_status()
            time.sleep(pause)
        
        # Return to start
        print(f"\nReturning to start position {start_pos:.2f}...")
        self.move_absolute(start_pos, velocity=velocity)
        self.wait_for_stop()
        self.print_status()
        
        print("\n✓ Stepwise motion completed")
    
    def trajectory_triangular_wave(self, amplitude=15.0, velocity=2.0, cycles=2):
        """
        Triangular wave pattern
        
        Args:
            amplitude: Peak amplitude (default 15)
            velocity: Movement speed (default 2.0)
            cycles: Number of complete waves (default 2)
        """
        print("\n" + "=" * 60)
        print(f"Trajectory: Triangular Wave ({cycles} cycles)")
        print("=" * 60)
        
        start_pos = self.get_position()
        print(f"Starting position: {start_pos:.2f}")
        
        positions = [
            start_pos,
            start_pos + amplitude,
            start_pos,
            start_pos - amplitude,
            start_pos
        ]
        
        for cycle in range(cycles):
            print(f"\nCycle {cycle+1}/{cycles}")
            for i, target in enumerate(positions[1:], 1):
                direction = "↗" if target > self.get_position() else "↘"
                print(f"  {direction} Moving to {target:.2f}")
                self.move_absolute(target, velocity=velocity)
                self.wait_for_stop()
                self.print_status()
                time.sleep(0.3)
        
        print("\n✓ Triangular wave completed")
    
    def trajectory_oscillation(self, amplitude=10.0, velocity=1.0, oscillations=5):
        """
        Oscillating motion (like a pendulum)
        
        Args:
            amplitude: Oscillation amplitude (default 10)
            velocity: Movement speed (default 1.0)
            oscillations: Number of oscillations (default 5)
        """
        print("\n" + "=" * 60)
        print(f"Trajectory: Oscillation ({oscillations} cycles)")
        print("=" * 60)
        
        center_pos = self.get_position()
        print(f"Center position: {center_pos:.2f}")
        
        for i in range(oscillations):
            print(f"\nOscillation {i+1}/{oscillations}")
            
            # Swing to positive side
            target = center_pos + amplitude
            print(f"  → Swinging to +{amplitude}: {target:.2f}")
            self.move_absolute(target, velocity=velocity)
            self.wait_for_stop()
            self.print_status()
            
            # Swing to negative side
            target = center_pos - amplitude
            print(f"  ← Swinging to -{amplitude}: {target:.2f}")
            self.move_absolute(target, velocity=velocity)
            self.wait_for_stop()
            self.print_status()
        
        # Return to center
        print(f"\nReturning to center: {center_pos:.2f}")
        self.move_absolute(center_pos, velocity=velocity)
        self.wait_for_stop()
        self.print_status()
        
        print("\n✓ Oscillation completed")
    
    def trajectory_speed_profile(self, distance=30.0, speeds=[0.5, 1.0, 1.5, 2.0]):
        """
        Test different speeds over the same distance
        
        Args:
            distance: Distance to move (default 30)
            speeds: List of velocities to test (default [0.5, 1.0, 1.5, 2.0])
        """
        print("\n" + "=" * 60)
        print(f"Trajectory: Speed Profile Test")
        print("=" * 60)
        
        start_pos = self.get_position()
        print(f"Starting position: {start_pos:.2f}")
        
        for i, speed in enumerate(speeds, 1):
            print(f"\nTest {i}/{len(speeds)}: Velocity = {speed}")
            
            # Move forward at this speed
            target = start_pos + distance
            print(f"  → Moving to {target:.2f} at v={speed}")
            start_time = time.time()
            self.move_absolute(target, velocity=speed, acceleration=1.5)
            self.wait_for_stop()
            elapsed = time.time() - start_time
            print(f"  Time taken: {elapsed:.2f}s")
            self.print_status()
            time.sleep(0.5)
            
            # Return to start
            print(f"  ← Returning to {start_pos:.2f}")
            self.move_absolute(start_pos, velocity=speed, acceleration=1.5)
            self.wait_for_stop()
            time.sleep(0.5)
        
        print("\n✓ Speed profile test completed")
    
    def trajectory_continuous_rotation(self, rotations=2, velocity=2.0):
        """
        Continuous rotation (for rotational axes)
        
        Args:
            rotations: Number of full rotations (default 2)
            velocity: Rotation speed (default 2.0)
        """
        print("\n" + "=" * 60)
        print(f"Trajectory: Continuous Rotation ({rotations} rotations)")
        print("=" * 60)
        
        start_pos = self.get_position()
        print(f"Starting position: {start_pos:.2f}")
        
        # Assuming 360 degrees per rotation (adjust if using radians)
        degrees_per_rotation = 360
        
        for i in range(rotations):
            target = start_pos + (degrees_per_rotation * (i + 1))
            print(f"\nRotation {i+1}/{rotations} → {target:.2f}°")
            self.move_absolute(target, velocity=velocity)
            self.wait_for_stop()
            self.print_status()
            time.sleep(0.3)
        
        # Return to start
        print(f"\nReturning to start: {start_pos:.2f}°")
        self.move_absolute(start_pos, velocity=velocity)
        self.wait_for_stop()
        self.print_status()
        
        print("\n✓ Rotation completed")


def main():
    """Main function with trajectory menu"""
    print("\n" + "=" * 70)
    print("ctrlX CORE Motor Trajectory Examples")
    print("=" * 70)
    
    # Configuration
    CTRLX_IP = "192.168.1.1"
    USERNAME = "boschrexroth"
    PASSWORD = "Muenchen81825"
    AXIS_NAME = "Axis_1"
    
    try:
        # Initialize
        motor = MotorTrajectories(CTRLX_IP, USERNAME, PASSWORD, AXIS_NAME)
        
        # Enable power
        motor.enable_power()
        time.sleep(1)
        
        # Show menu
        while True:
            print("\n" + "=" * 70)
            print("Available Trajectories:")
            print("=" * 70)
            print("1. Forward-Backward Motion")
            print("2. Stepwise Motion")
            print("3. Triangular Wave")
            print("4. Oscillation (Pendulum)")
            print("5. Speed Profile Test")
            print("6. Continuous Rotation")
            print("7. Custom Motion")
            print("0. Exit")
            print("=" * 70)
            
            choice = input("\nSelect trajectory (0-7): ")
            
            if choice == '0':
                print("\nExiting...")
                break
            
            elif choice == '1':
                distance = float(input("Distance (default 20): ") or "20")
                velocity = float(input("Velocity (default 2.0): ") or "2.0")
                cycles = int(input("Cycles (default 3): ") or "3")
                motor.trajectory_forward_backward(distance, velocity, cycles)
            
            elif choice == '2':
                step_size = float(input("Step size (default 5): ") or "5")
                num_steps = int(input("Number of steps (default 4): ") or "4")
                velocity = float(input("Velocity (default 1.5): ") or "1.5")
                motor.trajectory_stepwise(step_size, num_steps, velocity)
            
            elif choice == '3':
                amplitude = float(input("Amplitude (default 15): ") or "15")
                velocity = float(input("Velocity (default 2.0): ") or "2.0")
                cycles = int(input("Cycles (default 2): ") or "2")
                motor.trajectory_triangular_wave(amplitude, velocity, cycles)
            
            elif choice == '4':
                amplitude = float(input("Amplitude (default 10): ") or "10")
                velocity = float(input("Velocity (default 1.0): ") or "1.0")
                oscillations = int(input("Oscillations (default 5): ") or "5")
                motor.trajectory_oscillation(amplitude, velocity, oscillations)
            
            elif choice == '5':
                distance = float(input("Distance (default 30): ") or "30")
                motor.trajectory_speed_profile(distance)
            
            elif choice == '6':
                rotations = int(input("Rotations (default 2): ") or "2")
                velocity = float(input("Velocity (default 2.0): ") or "2.0")
                motor.trajectory_continuous_rotation(rotations, velocity)
            
            elif choice == '7':
                print("\nCustom Motion:")
                position = float(input("Target position: "))
                velocity = float(input("Velocity (default 1.5): ") or "1.5")
                acceleration = float(input("Acceleration (max 2.0, default 1.0): ") or "1.0")
                motor.move_absolute(position, velocity, acceleration, acceleration)
                motor.wait_for_stop()
                motor.print_status()
            
            else:
                print("Invalid choice!")
            
            input("\nPress Enter to continue...")
        
        print("\n✓ Session completed!")
        print("Note: Power is still ON. Use web UI to disable if needed.")
        
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
