from action_space import ActionSpace
import math

class AccelerationRaceRewardFunction:
    drive_distance = 82

    def __init__(self):
        self.last_action = None

    def compute_reward(self, start_location, previous_location, current_location, has_collided, current_rotation, chosen_action: ActionSpace = None, velocity = None):

        # Initialize reward
        reward = 0

        # Perfect center line - we want the agent to drive on this line.
        center_line_x_position = start_location.x

        # Calculate the absolute lateral distance from the center line
        current_lateral_distance = abs(current_location.x - center_line_x_position)

        abs_velocity = velocity.length()

        if self.has_reached_finish(start_location, current_location):
            # Reward for breaking (breaking = -1
            reward -= chosen_action.drive * 20.

            # Calculate the Euclidean distance
            distance = math.sqrt((previous_location.x - current_location.x) ** 2 +
                                 (previous_location.y - current_location.y) ** 2 +
                                 (previous_location.z - current_location.z) ** 2)

            # This means the agent is pretty much stopped.
            if distance < 0.01:
                reward = 500
                return reward, True

        else:
            # Reward for moving forward along the Y-axis
            forward_movement_reward = current_location.y - previous_location.y

            # Check the direction of movement
            if forward_movement_reward > 0:
                # Moving forward, assign the calculated reward
                reward += forward_movement_reward * 10

        reward += (0.5 - current_lateral_distance) * abs_velocity * 3

        if has_collided:
            reward = -100

        # Make sure the agent does not just steer as much as he wants, we want the outputs to be stable
        if self.last_action is not None:
            last_steer = self.last_action.steering
            last_drive = self.last_action.drive

            current_steer = chosen_action.steering
            current_drive = chosen_action.drive

            steer_diff = abs(current_steer - last_steer)
            drive_diff = abs(current_drive - last_drive)

            reward -= steer_diff + drive_diff

        self.last_action = chosen_action

        return reward, False

    def has_reached_finish(self, start_location, current_location):
        if current_location.y - start_location.y > self.drive_distance:
            return True
        return False

    def get_relative_vector_to_next_waypoint(self, start_location, current_location):
        return [current_location.x - start_location.x, current_location.y - start_location.y + self.drive_distance]

    def get_track_completion_percentage(self, current_location, start_location):
        # Distance traveled from the start along the Y-axis
        distance_traveled = start_location.y - current_location.y

        # Track length is predefined as self.drive_distance
        completion_percentage = (distance_traveled / self.drive_distance) * 100

        # Ensure the completion does not exceed 100%
        return min(completion_percentage, 100.0)