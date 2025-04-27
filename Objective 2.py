from matplotlib import pyplot as plt
import rclpy  # Import ROS client library for Python
from rclpy.node import Node  # Import Node class from ROS Python library
from std_msgs.msg import Float32MultiArray  # Import message type for ROS
from dynamixel_sdk import *  # Import Dynamixel SDK for servo control
import numpy as np  # Import numpy for numerical operations
from rclpy.logging import LoggingSeverity  # Import LoggingSeverity for setting log levels
 
# Defining the SimplePublisher class which inherits from Node
class SimplePublisher(Node):
 
    def __init__(self):
        super().__init__('publisher_mcontrol')  # Initialize the node with the name 'publisher_mcontrol'
        # Create a publisher object with String message type on the topic 'advanced_topic'
        # The second argument '10' is the queue size
        self.publisher_ = self.create_publisher(Float32MultiArray, 'joint_pos', 10)
        self.publisher_F = self.create_publisher(Float32MultiArray, 'joint_F', 10)
        self.subscription = self.create_subscription(Float32MultiArray,'joint_state',self.send_command, 10)

        #user defined
        self.l=[100,110]
        self.initial_pos = [-50, 160]
        self.final_pos = [50, 160]
        self.no_sweeps = 25
        self.y_limit = 180
        self.force_threshold = 2
        self.pos_threshold = 20

        self.finishing_pos = [100, 120]

        #to be updated
        self.sweep_number = 0
        self.contact_positions = []
        self.force_sensing = False
        self.desired_angles = self.better_IK(self.l, self.initial_pos)

        #constants
        self.increment = (self.final_pos[0] - self.initial_pos[0]) / self.no_sweeps
        


    def send_command(self,msg):

        pos = Float32MultiArray()
        force_F = Float32MultiArray()

        angles = [msg.data[0],msg.data[1]]
        velocity = [msg.data[2],msg.data[3]]
        current = [msg.data[4],msg.data[5]]
        
        current_pos = self.forward_kinematics(self.l, angles)
        J = self.jacobian_IK(self.l,np.deg2rad(angles))
        current_force = self.Compute_Force(J, current)

        print("Current position: ", current_pos)
        print("Desired Position: ", self.forward_kinematics(self.l, self.desired_angles))
        print("Position Error: ", np.linalg.norm(self.forward_kinematics(self.l, self.desired_angles) - current_pos))
        print("Current force: ", current_force)

        if (np.linalg.norm(current_force) >= self.force_threshold) and (self.force_sensing):
                print("Contact detected at position: ", current_pos)
                self.contact_positions.append(current_pos + current_force)
                self.force_sensing = False
                self.sweep_number += 1
                self.desired_angles = self.better_IK(self.l, [(self.initial_pos[0] + self.increment * self.sweep_number), self.initial_pos[1]])

        desired_position = self.forward_kinematics(self.l, self.desired_angles)

        if np.linalg.norm(desired_position - current_pos) < self.pos_threshold:

            self.force_sensing = True

            if self.sweep_number == self.no_sweeps:
                #print("Finished sweeping")
                complete_contact_positions = np.array(self.contact_positions)
                print("Contact positions: ", complete_contact_positions)
                self.desired_angles = self.better_IK(self.l, self.finishing_pos)
                self.force_sensing = False

            elif current_pos[1] >= self.y_limit:
                #print("No contact detected")
                self.force_sensing = False
                self.sweep_number += 1
                
                self.desired_angles = self.better_IK(self.l, [(self.initial_pos[0] + self.increment * self.sweep_number), self.initial_pos[1]])
                

            else:
                self.desired_angles = self.better_IK(self.l, self.forward_kinematics(self.l, self.desired_angles) + [0, 1] )


        pos.data = [float(self.desired_angles[0]),float(self.desired_angles[1])]  # Setting the message
        self.publisher_.publish(pos)  # Publishing the message



    def forward_kinematics(self,l,theta):
        theta1  = np.deg2rad(theta[0] + 90)
        theta2 = np.deg2rad(theta[1])
        x = l[0] * np.cos(theta1) + l[1] * np.cos(theta1 + theta2)
        y = l[0] * np.sin(theta1) + l[1] * np.sin(theta1 + theta2)
        return np.array([x, y])
 
    def jacobian_IK(self,l,theta):
        J=np.array([[-l[0] * np.sin(theta[0]) - l[1] * np.sin(theta[0] + theta[1]), -l[1] * np.sin(theta[0] + theta[1])],
                     [l[0] * np.cos(theta[0])+l[1] * np.cos(theta[0]+theta[1]), l[1] * np.cos(theta[0]+theta[1])]])
        return J

    def Compute_Force(self, J, current):
        F  = np.linalg.inv(J.T)@(current)
        return F   

    def plot_force_map(self):
        if not self.mapping_data:
            print("No contact data recorded.")
        return
 
    def better_IK(self,l,P):
        x = P[0]
        y = P[1]
        L1 = l[0]
        L2 = l[1]

        distance_sq = x**2 + y**2
        max_reach = L1 + L2
        min_reach = abs(L1 - L2)
        if distance_sq > max_reach**2 or distance_sq < min_reach**2:
            return None, None
        
        cos_theta2 = (x**2 + y**2 - L1**2 - L2**2) / (2 * L1 * L2)
        cos_theta2 = np.clip(cos_theta2, -1.0, 1.0)
        theta2 = np.arccos(cos_theta2)
        theta1 = np.arctan2(y, x) - np.arctan2(L2 * np.sin(theta2), L1 + L2 * np.cos(theta2))
        
        theta1_deg = np.degrees(theta1)
        theta2_deg = np.degrees(theta2)
        
        if 0 <= theta1_deg <= 180 and -90 <= theta2_deg <= 90:
            return np.array([theta1_deg - 90, theta2_deg])
        return None, None
    
    def print_graph(self):
        # Extract x and y coordinates
        complete_contact_positions = np.array(self.contact_positions)
        x = complete_contact_positions[:, 0]
        y = complete_contact_positions[:, 1]

        # Create the plot
        plt.figure(figsize=(8, 6))
        plt.scatter(x, y, color='red', label='Contact Positions')

        # Set axis limits (adjust these values as needed)
        plt.xlim(-100, 100)    # x-axis range: min=-50, max=50
        plt.ylim(0, 200)    # y-axis range: min=170, max=180

        # Add labels, title, and grid
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.title('Contact Positions Plot')
        plt.legend()
        plt.grid(True)

        # Show the plot
        plt.savefig('contact_positions.png')
        print("Plot saved to 'contact_positions.png'")
        plt.close()


    
def main(args=None):
    rclpy.init(args=args)
    simple_publisher = SimplePublisher()

    try:
        rclpy.spin(simple_publisher)
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received, shutting down...")
        #pass
    finally:
        simple_publisher.print_graph()

        # Destroy the node
        if rclpy.ok():
            simple_publisher.destroy_node()
        
        # Only try to shutdown if ROS is still running
        if rclpy.ok():
            rclpy.shutdown()
        
        # Explicitly close all matplotlib figures
        plt.close('all')
 
 
