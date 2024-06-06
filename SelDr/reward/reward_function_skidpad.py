import random

import numpy as np
from scipy.interpolate import CubicSpline

from action_space import ActionSpace

from utils.SkidPadDirection import SkidPadDirection
from utils.vector_math import VectorMath


class SkidPadRaceRewardFunction:

    def __init__(self):

        #Abs start pos: ((41.2, 321.3), 180.0),

        self.waypoint_and_rotation = [
            ((-0.3, 0.0), 180.0),
            ((-0.3, 5.0), 180.0),
            ((-0.3, 16.9), 180.0),  # Center
            ((-2.0, 22.3), 217.7),  # AboveCenterRight
            ((-9.2, 25.8), 270.0),  # RightTop
            ((-18.2, 16.7), 0.0),  # RightRight
            ((-9.1, 8.0), 90.0),  # RightBottom
            ((-0.3, 16.9), 180.0),  # Center
            ((-9.2, 25.8), 270.0),  # RightTop
            ((-18.2, 16.7), 0.0),  # RightRight
            ((-9.1, 8.0), 90.0),  # RightBottom
            ((-1.7, 12), 149.3), # BelowCenterRight
            ((-0.3, 16.9), 180.0),  # Center
            ((1.5, 21.3), 147.3), # AboveCenterLeft
            ((9.5, 25.8), 90.0),  # LeftTop
            ((18.2, 16.8), 0.0),  # LeftLeft
            ((9.4, 8.0), 270.0),  # LeftBottom
            ((-0.3, 16.9), 180.0),  # Center
            ((9.5, 25.8), 90.0),  # LeftTop
            ((18.2, 16.8), 0.0),  # LeftLeft
            ((9.4, 8.0), 270.0),  # LeftBottom
            ((1.3, 12.4), 205.2), # BelowCenterLeft
            ((-0.3, 16.9), 180.0),  # Center
            ((0.0, 23.1), 178.7), # AboveCenter
            ((0.0, 27.5), 180.0),  # AfterFinish (StopArea)
        ]

        self.waypoints = [x[0] for x in self.waypoint_and_rotation]
        self.rotations = [x[1] for x in self.waypoint_and_rotation]

        waypoints_np = np.array(self.waypoints)

        # Create time parameter t based on the number of waypoints
        t = np.linspace(0, 1, len(waypoints_np))

        # Creating the splines
        self.cs_x = CubicSpline(t, waypoints_np[:, 0], bc_type='clamped')
        self.cs_y = CubicSpline(t, waypoints_np[:, 1], bc_type='clamped')

        self.current_point = 0

    def compute_reward(self, start_location, previous_location, current_location, has_collided, current_rotation,
                       chosen_action: ActionSpace = None, velocity = None):

        # Initialize reward
        reward = 0

        if self.current_point != len(self.waypoints):

            # calculate the relative locations
            current_relative_location_vec = current_location - start_location
            current_relative_location = (current_relative_location_vec.x, current_relative_location_vec.y)

            # Check if the car is closer than 3 meters to a point.
            next_waypoint = self.waypoints[self.current_point]
            distance_to_next_waypoint = VectorMath.euclidean_distance(current_relative_location, next_waypoint)

            if distance_to_next_waypoint < 3:
                # If closer than 3 meters to the next waypoint, move to the next waypoint
                self.current_point = (self.current_point + 1)

            abs_velocity = velocity.length()

            # If the agent is not moving, give him a negative reward.
            if abs_velocity < 1:
                reward -= 10 * (1 - abs_velocity)

            clipped_abs_velocity = min(abs_velocity, 10)

            distance_to_centerline = self.find_closest_point(current_relative_location)

            reward += (0.5 - distance_to_centerline) * clipped_abs_velocity * 3


        else:

            # Reward for breaking (breaking = -1)
            reward -= chosen_action.drive * 10.

            # Reward for staying in the center
            center_line_x_position = start_location.x

            # TODO: Change this reward calculation for staying in the center, else he might want to enter the end
            #  position from the side to gain more reward.....!!!! bigbrain thought

            # Calculate the absolute lateral distance from the center line for both previous and current positions
            previous_lateral_distance = abs(previous_location.x - center_line_x_position)
            current_lateral_distance = abs(current_location.x - center_line_x_position)

            # Calculate the difference in lateral distances to determine if closer or further from center
            # If the car moves closer to the center, this value will be positive, which should increase the reward
            # If the car moves away from the center, this value will be negative, which should decrease the reward
            lateral_distance_change = previous_lateral_distance - current_lateral_distance

            # Add this difference to the reward
            reward += lateral_distance_change * 10  # Scale the reward for moving towards the center

            distance = VectorMath.euclidean_distance(previous_location, current_location)

            if distance < 0.01:
                reward = 500
                return reward, True

        if has_collided:
            reward = -200

        return reward, False

    def reset(self):
        self.current_point = 0

    def get_set_random_location(self):
        # More possibility to get placed at the start instead of the end, so the agent learns the entire course:
        #point_num = int((len(self.waypoints) - 2) * (random.random() ** 2))

        point_num = random.randint(0, len(self.waypoints)-3)

        # Return none to let caller know the point is the first one and doesn't need to modify car's position:
        if point_num == 0:
            return None, None

        # Set the current point to the next point.
        self.current_point = point_num+1
        return self.waypoints[point_num], self.rotations[point_num]

    def get_relative_vector_to_next_waypoint(self):
        cur_waypoint = self.waypoints[self.current_point % len(self.waypoints)]
        return cur_waypoint

    def get_current_point(self):
        return self.current_point

    def get_distance_to_next_waypoint(self):
        return

    def get_drive_direction_to_next_waypoint(self):
        if self.current_point < 10:
            return SkidPadDirection.RIGHT.value
        if self.current_point < 20:
            return SkidPadDirection.LEFT.value
        else:
            return SkidPadDirection.FORWARD.value

    def find_closest_point(self, agent_position, num_samples=500):

        x_segment, y_segment = self.get_segmented_splines(num_samples)
        # Find the point with the minimum distance to the agent's position
        distances = np.sqrt((x_segment - agent_position[0]) ** 2 + (y_segment - agent_position[1]) ** 2)
        min_index = np.argmin(distances)
        min_distance = distances[min_index]

        return min_distance

    def get_splines(self):
        return self.cs_x, self.cs_y

    def get_segmented_splines(self, num_samples=1000):
        # Determine segment indices based on current point
        if self.current_point == 0:
            start_index = 0
        else:
            start_index = self.current_point - 1

        # Set end_index to one point beyond current_point, if possible
        end_index = self.current_point + 1
        if end_index >= len(self.waypoints):  # Check if we've reached or exceeded the list boundary
            end_index = len(self.waypoints) - 1  # Use the last waypoint if no next point is available

        # Calculate range in t based on these indices
        num_waypoints = len(self.waypoints)
        t_start = start_index / num_waypoints
        t_end = (end_index + 1) / num_waypoints  # +1 because we want the segment to reach the waypoint at end_index

        # Sample points only from the current segment of the spline
        t_segment = np.linspace(t_start, t_end, num_samples)
        return self.cs_x(t_segment), self.cs_y(t_segment)

    def get_track_completion_percentage(self):
        # The total number of waypoints minus the first and the last waypoint for actual driving
        total_waypoints = len(self.waypoints) - 2  # Excluding start and stop waypoints
        # Current point index, ensure it does not count the finish point if it should not be counted
        current_point_index = min(self.current_point, total_waypoints)

        # Calculate completion percentage
        completion_percentage = (current_point_index / total_waypoints) * 100
        return completion_percentage
