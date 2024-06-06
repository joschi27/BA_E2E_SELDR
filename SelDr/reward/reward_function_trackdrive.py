import random

import numpy as np
from scipy.interpolate import CubicSpline

from action_space import ActionSpace
import math

from reward.trackdrive_waypoints import TrackDriveWaypoints
from reward.nascar_trackdrive_waypoints import NascarTrackDriveWaypoints
from utils.vector_math import VectorMath


class TrackDriveRaceRewardFunction:
    def __init__(self, track_index, eval_track=False):

        track_lib = TrackDriveWaypoints()

        if eval_track:
            self.waypoints, self.rotations, self.stop_position = track_lib.get_track(track_index, eval_track)
        else:
            self.waypoints, self.rotations, self.stop_position = track_lib.get_track(track_index)

        self.current_point = 0

        # Convert waypoints to numpy array and ensure it's closed
        waypoints_np = np.array(
            self.waypoints + [self.waypoints[0]])  # Add the first point at the end to close the track

        self.waypoints.append(self.waypoints[0])

        # Create time parameter t based on the number of waypoints
        t = np.linspace(0, 1, len(waypoints_np))

        # Creating the splines for a closed track
        self.cs_x = CubicSpline(t, waypoints_np[:, 0], bc_type='periodic')
        self.cs_y = CubicSpline(t, waypoints_np[:, 1], bc_type='periodic')

    def compute_reward(self, start_location, previous_location, current_location, has_collided, current_rotation,
                       chosen_action: ActionSpace = None, velocity = None):

        # Initialize reward
        reward = 0

        if self.current_point == len(self.waypoints):
            self.current_point = 0

        # calculate the relative locations
        current_relative_location_vec = current_location - start_location
        current_relative_location = (current_relative_location_vec.x, current_relative_location_vec.y)

        last_relative_location_vec = previous_location - start_location
        last_relative_location = (last_relative_location_vec.x, last_relative_location_vec.y)

        # Check if the car is closer than 3 meters to a point.
        next_waypoint = self.waypoints[self.current_point]
        distance_to_next_waypoint = VectorMath.euclidean_distance(current_relative_location, next_waypoint)

        if distance_to_next_waypoint < 3:
            # If closer than 3 meters to the next waypoint, move to the next waypoint
            self.current_point = (self.current_point + 1)

        movement_distance = abs(VectorMath.euclidean_distance(current_relative_location, last_relative_location))

        abs_velocity = velocity.length()

        # If the agent is not moving, give him a negative reward.
        if abs_velocity < 1:
            reward -= 10 * (1 - abs_velocity)

        clipped_abs_velocity = min(abs_velocity, 10)

        # Get car rotation
        yaw = math.radians(current_rotation.yaw)
        front_vector = np.array([math.cos(yaw), math.sin(yaw)])
        front_distance = 1.5
        front_offset = front_vector * front_distance

        # Calculating front position in Pygame coordinates
        front_position = np.array(
            [current_relative_location[0] + front_offset[0], current_relative_location[1] + front_offset[1]])

        distance_to_centerline = self.find_closest_point(front_position)

        reward += (0.5 - distance_to_centerline) * clipped_abs_velocity * 3

        if has_collided:
            reward = -250

        return reward, False

    def reset(self):
        self.current_point = 0

    def get_set_random_location(self):
        # More possibility to get placed at the start instead of the end, so the agent learns the entire course:
        #point_num = int((len(self.waypoints) - 2) * (random.random() ** 2))

        point_num = random.randint(0, len(self.waypoints) - 1)

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

    def find_closest_point(self, agent_position, num_samples=5000):
        # Sample points along the spline
        t_new = np.linspace(0, 1, num_samples)
        x_new, y_new = self.cs_x(t_new), self.cs_y(t_new)

        # Find the point with the minimum distance to the agent's position
        distances = np.sqrt((x_new - agent_position[0]) ** 2 + (y_new - agent_position[1]) ** 2)
        min_index = np.argmin(distances)

        min_distance = distances[min_index]

        return min_distance

    def get_splines(self):
        return self.cs_x, self.cs_y

    def get_track_completion_percentage(self):
        completion_percentage = (min(self.current_point, len(self.waypoints)) / len(self.waypoints)) * 100
        return completion_percentage