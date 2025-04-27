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
        self.theta=np.zeros(2)
        self.l=[100,110]
        self.des_pos=[]
        self.q=[0.0,0.0]
        self.step=0.3
        self.damping_factor=0.01 # for jacobian DLS method
        self.gain= 0.1 #for jacobian transpose method
        self.current= [0.0, 0.0]
        self.desired = [0.0, 0.0]
        # self.desired_increments = np.array([[[-50, 140], [-50, 150], [-50, 155], [-50, 160], [-50, 165], [-50, 170], [-50, 175], [-50, 180], [-50,185]],
        #                             [[0, 155], [0, 157.5], [0, 160], [0, 165], [0, 170], [0, 175], [0, 180], [0, 185], [0,190]],
        #                             [[50,140], [50, 150], [50, 155], [50, 160], [50, 165], [50, 170], [50, 175], [50, 180], [50,185]]])
 
        self.desired_increments = np.array([         
                                    [[-50, 155], [-50, 156], [-50, 157], [-50, 158], [-50, 159], [-50, 160], [-50, 161], [-50, 162], [-50, 163], [-50, 164],
                                    [-50, 165], [-50, 166], [-50, 167], [-50, 168], [-50, 169], [-50, 170], [-50, 171], [-50, 172], [-50, 173], [-50, 174],
                                    [-50, 175], [-50, 176], [-50, 177], [-50, 178], [-50, 179], [-50, 180], [-50, 181], [-50, 182], [-50, 183], [-50, 184], [-50, 185]],
                                    
                                    [[0, 155], [0, 156], [0, 157], [0, 158], [0, 159], [0, 160], [0, 161], [0, 162], [0, 163], [0, 164],
                                    [0, 165], [0, 166], [0, 167], [0, 168], [0, 169], [0, 170], [0, 171], [0, 172], [0, 173], [0, 174],
                                    [0, 175], [0, 176], [0, 177], [0, 178], [0, 179], [0, 180], [0, 181], [0, 182], [0, 183], [0, 184], [0, 185]],

                                    [[50, 155], [50, 156], [50, 157], [50, 158], [50, 159], [50, 160], [50, 161], [50, 162], [50, 163], [50, 164],
                                    [50, 165], [50, 166], [50, 167], [50, 168], [50, 169], [50, 170], [50, 171], [50, 172], [50, 173], [50, 174],
                                    [50, 175], [50, 176], [50, 177], [50, 178], [50, 179], [50, 180], [50, 181], [50, 182], [50, 183], [50, 184], [50, 185]]])

        self.increment = 0
        self.line_number = 0
        self.threshold = 100.0  # Distance threshold to move to next waypoint
        self.des_pos = self.desired_increments[self.line_number, self.increment]
        self.force_threshold = 2.0
        self.saved_positions = []
        self.required_steps = 5
        self.consecutive_count = 0
        self.prev_pos = [0,0]
        self.saved_positions = [] #new code
        self.saved_forces = [] #new code
        self.saved_errors = [] #new code for error tracking
        self.mapping_pub = self.create_publisher(Float32MultiArray, 'mapping_xy_F', 10) #new code
        self.mapping_data = [] #new code
        self.Y_error = []
 
        
 
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
 
    def anal_IK(self,l,P):
        D=(P[0]**2+P[1]**2-l[0]**2-l[1]**2)/(2*l[0]*l[1])
        self.q[1]=np.arctan(np.sqrt(1-D**2)/(D+0.00001))
        self.q[0]=np.arctan(P[1]/P[0])-np.arctan(l[1]*np.sin(self.q[1])/(l[0]+l[1]*np.cos(self.q[1])))
        return np.rad2deg(self.q)
    
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
 
    def send_command(self,msg):
        pos=Float32MultiArray()
        force_F=Float32MultiArray()
        self.theta=[msg.data[0],msg.data[1]]
        self.current=[msg.data[4],msg.data[5]]
        J = self.jacobian_IK(self.l,np.deg2rad(self.theta))
        current_pos = self.forward_kinematics(self.l, self.theta)

        if self.des_pos[1] == 185:
            print("desired Y:", self.des_pos[1])
            print("Reached Y:", current_pos[1])
            print("error:",  self.des_pos[1] - current_pos[1])
            self.Y_error.append(self.des_pos[1] - current_pos[1])
            print("error appended: ", self.Y_error[0])
        
 
        # error = self.des_pos - current_pos
 
        J = self.jacobian_IK(self.l,np.deg2rad(self.theta))
 
        self.desired = self.better_IK(self.l,self.des_pos)
        F = self.Compute_Force(J,self.current)
        pos.data = [float(self.desired[0]),float(self.desired[1])]  # Setting the message
 
        if np.linalg.norm(self.des_pos - current_pos) < self.threshold:
            print("Reached Position: ", current_pos)
            if self.increment < self.desired_increments.shape[1] - 1:
                self.increment += 1
            elif self.increment >= self.desired_increments.shape[1] - 1:
                # Always store the final desired position and error for the line we just finished
                self.saved_positions.append(self.des_pos.copy())
                self.saved_forces.append(F.copy())
                self.saved_errors.append(current_pos[0] - self.des_pos[0])

                if self.line_number < self.desired_increments.shape[0] - 1:
                    self.line_number += 1
                    self.increment = 0
                else:
                    print("finished")
                    print(self.Y_error)
                    self.plot_force_map()
                    rclpy.shutdown()
            time.sleep(0.5)
 
        
        # print(self.theta1, self.theta)
        print("Force Magnitude: "),
        print(np.linalg.norm(F))
        #print(F[0])
        #print(F[1])
        print("Desired Position: "),
        print(self.des_pos)

    
        if np.linalg.norm(F) >= self.force_threshold:
            print("Force threshold exceeded, registering position!")
            self.saved_positions.append(self.des_pos.copy())
            self.saved_forces.append(F.copy())
            self.saved_errors.append(current_pos[0] - self.des_pos[0])

        self.des_pos = self.desired_increments[self.line_number, self.increment]
            
        data_point = [current_pos[0], current_pos[1], F[0], F[1]]
 
        # Save the data
        self.mapping_data.append(data_point)
 
        # Create and publish the message
        mapping_msg = Float32MultiArray()
        mapping_msg.data = data_point
        self.mapping_pub.publish(mapping_msg)
        # Target clamping: Limit the step size to avoid large jumps
        
 
        
 
        force_F.data = [float(F[0]),float(F[1])]
 
        self.publisher_.publish(pos)  # Publishing the message
 
 
        self.publisher_F.publish(force_F)
 
        data = np.array(self.mapping_data)
        x, y = data[:, 0], data[:, 1]
        fx, fy = data[:, 2], data[:, 3]
 
        # plt.figure(figsize=(8, 6))
        # plt.quiver(x, y, fx, fy, color='red', angles='xy', scale_units='xy', scale=1, label='Force Vectors')
        # plt.scatter(x, y, color='blue', label='Contact Points')
        # plt.title("2D Force Mapping")
        # plt.xlabel("X Position (mm)")
        # plt.ylabel("Y Position (mm)")
        # plt.axis('equal')
        # plt.grid(True)
        # plt.legend()
        # plt.show()
 
    def Compute_Force(self, J, current):
        F  = np.linalg.inv(J.T)@(current)
        return F   

    def plot_force_map(self):
        if not self.mapping_data:
            print("No contact data recorded.")
        else:
            for pos, y_error, force in zip(self.saved_positions, self.Y_error, self.saved_forces):
                pos_list = pos.tolist() if hasattr(pos, 'tolist') else list(pos)
                x = pos_list[0]
                y = pos_list[1]
                if 0 <= y_error < 4:
                    tissue = "undefined"
                elif 4 <= y_error < 10:
                    tissue = "adipose tissue"
                elif 10 <= y_error < 20:
                    tissue = "skeletal muscle"
                elif y_error >= 20:
                    tissue = "bone"
                else:
                    tissue = "undefined"
                print("{:<10} {:<10} {:<10.2f} {:<20}".format(x, y, y_error, tissue))
 
def main(args=None):
    rclpy.init(args=args)
    simple_publisher = SimplePublisher()
 
    try:
        rclpy.spin(simple_publisher)
    except KeyboardInterrupt:
        pass
    finally:
        simple_publisher.destroy_node()
        rclpy.shutdown()
 
 
        # self.prev_pos = current_pos
 
 
        # else:
        #     if np.linalg.norm(F) >= self.force_threshold:
        #         # Save the current position in a list
        #         self.saved_positions.append(current_pos.copy())
        #         self.i += 2
        #         self.des_pos = np.array(self.desired_increments[self.i])
 
            # # Jump to i + 2 if within bounds
            # if self.i + 2 < len(self.desired_increments):
            #     self.i += 2
            # else:
            #     self.i = len(self.desired_increments) - 1  # Move to last element if not enough steps
 
            # self.des_pos = np.array(self.desired_increments[self.i])
 
              
 
 
# The main function which serves as the entry point for the program
# def main(args=None):
#     rclpy.init(args=args)  # Initialize the ROS2 Python client library
#     simple_publisher = SimplePublisher()  # Create an instance of the SimplePublisher
 
#     try:
#         rclpy.spin(simple_publisher)  # Keep the node alive and listening for messages
#     except KeyboardInterrupt:  # Allow the program to exit on a keyboard interrupt (Ctrl+C)
#         pass
 
#     simple_publisher.destroy_node()  # Properly destroy the node
#     rclpy.shutdown()  # Shutdown the ROS2 Python client library
 
# # This condition checks if the script is executed directly (not imported)
# if __name__ == '__main__':
#     main()  # Execute the main function