import math

import numpy as np
import pygame
from PIL import Image

import hyperparameters
from action_space import ActionSpace


class CarDebugInfo:
    _instance = None

    def __new__(cls):
        # This method creates a new instance if one doesn't exist
        if cls._instance is None:
            cls._instance = super(CarDebugInfo, cls).__new__(cls)

            cls._instance.graph_data = []
            cls._instance.debug_data = {}

            cls._instance.camera_image = None
            cls._instance.camera_width = None
            cls._instance.camera_height = None

            cls._instance.game_display = None
            cls._instance.display_width = 1024
            cls._instance.display_height = 512
            if hyperparameters.SHOW_PYGAME_SCREEN:
                cls._instance.start_pygame_visuals()

            cls._instance.camera_image = None

        return cls._instance

    def set_image(self, image):
        self.camera_image = image

    def set_image_dimensions(self, width, height):
        self.camera_width = width
        self.camera_height = height

    def set_debug_info(self, key, value):
        self.debug_data[key] = value

    def get_debug_info(self, key):
        return self.debug_data.get(key)

    def render_debug_info(self, max_reward=10, min_reward=-10, bar_width=400, bar_height=20,
                          position=(50, 50)):

        screen = self.game_display

        self.render_camera(screen)

        self.render_reward(bar_height, bar_width, max_reward, min_reward, position, screen)

        self.render_waypoint_text(position, screen)

        screen_center = (screen.get_width() // 2, screen.get_height() // 2)

        y_scale = screen.get_height() / 120
        x_scale = screen.get_width() / 120

        if self.get_debug_info("splines"):
            self.render_splines(screen, self.get_debug_info("splines")[0], self.get_debug_info("splines")[1], screen_center, x_scale, y_scale)

        if self.get_debug_info("segment_splines"):
            self.render_segmented_splines(screen, self.get_debug_info("segment_splines")[0], self.get_debug_info("segment_splines")[1],
                                screen_center, x_scale, y_scale)

        if self.get_debug_info("waypoint_vector") and self.get_debug_info("relative_location_vector"):
            self.render_location_waypoint(screen, self.get_debug_info("waypoint_vector"), self.get_debug_info("relative_location_vector"), screen_center, x_scale, y_scale)

        self.render_steer_throttle_square(screen)

        # Add the accelerometer vector to the graph data:
        accelerometer_vector = self.get_debug_info("accelerometer_vector")
        self.graph_data.append([accelerometer_vector[0][0], accelerometer_vector[0][1]])
        self.draw_accelerometer_graph(screen, self.graph_data, [10, screen.get_height()-110, 200, 100])

        if self.get_debug_info("extracted_features") is not None:
            self.render_yolo_detections(screen, self.get_debug_info("extracted_features"), ["large_orange_cone", "orange_cone", "blue_cone", "yellow_cone"], self.display_width, self.display_height)

        pygame.display.flip()
        pygame.event.pump()

    def render_location_waypoint(self, screen, waypoint_vector, relative_location_vector, screen_center, x_scale, y_scale):
        rel_loc_vec_pygame = (
            screen_center[0] - relative_location_vector.x * x_scale,
            screen_center[1] - relative_location_vector.y * y_scale,
        )

        # Draw the relative location vector
        #pygame.draw.line(screen, (255, 0, 0), screen_center, rel_loc_vec_pygame, 2)
        pygame.draw.circle(screen, (255, 255, 0), rel_loc_vec_pygame, 5)
        waypoint_vec_pygame = (
            screen_center[0] - waypoint_vector[0] * x_scale,
            screen_center[1] - waypoint_vector[1] * y_scale,
        )

        # Draw the waypoint vector
        pygame.draw.line(screen, (0, 255, 0), waypoint_vec_pygame, rel_loc_vec_pygame, 2)
        pygame.draw.circle(screen, (255, 0, 255), waypoint_vec_pygame, 5)
        pygame.draw.circle(screen, (255, 0, 255), waypoint_vec_pygame, 20, 3)

        if self.get_debug_info("current_rotation") is not None:
            yaw = math.radians(self.get_debug_info("current_rotation").yaw)
            front_vector = np.array([math.cos(yaw), math.sin(yaw)])
            front_distance = 2
            front_offset = front_vector * front_distance

            # Calculating front position in Pygame coordinates
            front_position = np.array(
                [relative_location_vector.x + front_offset[0], relative_location_vector.y + front_offset[1]])
            front_pos_pygame = (
                screen_center[0] - front_position[0] * x_scale,
                screen_center[1] - front_position[1] * y_scale,
            )

            # Draw the front offset vector
            pygame.draw.line(screen, (0, 0, 255), rel_loc_vec_pygame, front_pos_pygame, 4)
            pygame.draw.circle(screen, (0, 255, 255), front_pos_pygame, 5)

    def render_splines(self, screen, cs_x, cs_y, screen_center, x_scale, y_scale, num_points=200):
        # Generate points along the spline
        t_values = np.linspace(0, 1, num_points)
        spline_points_x = cs_x(t_values)
        spline_points_y = cs_y(t_values)

        # Convert spline points to screen coordinates
        spline_points = [
            (screen_center[0] - x * x_scale, screen_center[1] - y * y_scale)
            for x, y in zip(spline_points_x, spline_points_y)
        ]

        # Draw the spline on the screen
        # Set the color of the spline line here (e.g., blue color (0, 0, 255))
        spline_color = (0, 0, 255)
        # Pygame draw lines between each pair of points
        for i in range(len(spline_points) - 1):
            pygame.draw.line(screen, spline_color, spline_points[i], spline_points[i + 1], 2)

    def render_segmented_splines(self, screen, spline_points_x, spline_points_y, screen_center, x_scale, y_scale):

        # Convert spline points to screen coordinates
        spline_points = [
            (screen_center[0] - x * x_scale, screen_center[1] - y * y_scale)
            for x, y in zip(spline_points_x, spline_points_y)
        ]

        # Draw the spline on the screen
        # Set the color of the spline line here (e.g., blue color (0, 0, 255))
        spline_color = (255, 0, 0)
        # Pygame draw lines between each pair of points
        for i in range(len(spline_points) - 1):
            pygame.draw.line(screen, spline_color, spline_points[i], spline_points[i + 1], 2)

    def render_waypoint_text(self, position, screen):
        # Initialize Pygame font
        if not pygame.font.get_init():
            pygame.font.init()
        font = pygame.font.Font(None, 24)  # Use Pygame's default font and set size
        current_waypoint = self.get_debug_info("current_waypoint")
        reward_text = f"CheckPoint: {current_waypoint}, "
        # Format the reward to 5 decimal places and render it as text
        reward_text += "Reward: {:.5f}".format(self.get_debug_info("last_reward"))

        if self.get_debug_info("driving_direction") is not None:
            reward_text += " Driving Direction: " + str(self.get_debug_info("driving_direction"))

        text_surface = font.render(reward_text, True, (0, 0, 0))
        # Calculate text position (to the top of the progress bar)
        text_position = (position[0], position[1] - 20)
        # Blit the text surface onto the game display
        screen.blit(text_surface, text_position)

    def render_reward(self, bar_height, bar_width, max_reward, min_reward, position, screen):
        # Map the reward to a 0-1 range for the progress bar
        normalized_reward = (self.get_debug_info("last_reward") - min_reward) / (max_reward - min_reward)
        filled_bar_length = normalized_reward * bar_width

        # Define colors
        background_color = (255, 255, 255)  # White background
        filled_color = (0, 255, 0) if self.get_debug_info("last_reward") >= 0 else (
            255, 0, 0)  # Green for positive, red for negative rewards

        # Draw the background of the progress bar
        background_rect = pygame.Rect(position[0], position[1], bar_width, bar_height)
        pygame.draw.rect(screen, background_color, background_rect)

        # Draw the filled part of the progress bar
        filled_rect = pygame.Rect(position[0], position[1], filled_bar_length, bar_height)
        pygame.draw.rect(screen, filled_color, filled_rect)

    def render_camera(self, screen):
        img = np.frombuffer(self.camera_image.raw_data, dtype=np.uint8).reshape(
            (self.camera_height, self.camera_width, 4))
        img = img[:, :, :3]  # Extract the BGR values
        img = img[:, :, [2, 1, 0]]  # Convert BGR to RGB

        # Cutting off the top 50% and bottom 20%
        top_cut = int(self.camera_height * 0.50)
        bottom_cut = int(self.camera_height * 0.80)
        img = img[top_cut:bottom_cut, :]

        img = np.array(Image.fromarray(img).resize((self.display_width, self.display_height)))
        img = img.swapaxes(0, 1)
        surface = pygame.surfarray.make_surface(img)
        screen.blit(surface, (0, 0))

    def render_steer_throttle_square(self, screen):
        square_size = 120  # Increase the size of the square for better visibility
        margin = 20  # Margin from the bottom right corner

        # Bottom right corner position
        square_position = (screen.get_width() - square_size - margin, screen.get_height() - square_size - margin)

        # Draw the square more distinctly with a thicker border and different color
        pygame.draw.rect(screen, (0, 128, 255),
                         pygame.Rect(square_position[0], square_position[1], square_size, square_size),
                         4)  # Blue color and thicker border

        # Initialize Pygame font
        if not pygame.font.get_init():
            pygame.font.init()
        font = pygame.font.Font(None, 24)  # Use Pygame's default font and set size

        # Text for steering and throttle annotations
        steer_left_text = font.render("-1", True, (0, 0, 0))
        steer_right_text = font.render("1", True, (0, 0, 0))
        throttle_top_text = font.render("1", True, (0, 0, 0))
        throttle_bottom_text = font.render("-1", True, (0, 0, 0))
        steer_text = font.render("Steer", True, (0, 0, 0))
        throttle_text = font.render("Throttle", True, (255, 255, 255))

        # Blit the text surfaces to the screen
        screen.blit(steer_text, (square_position[0] + square_size / 2 - steer_text.get_width() / 2,
                                 square_position[1] - steer_text.get_height() - 25))
        screen.blit(throttle_text, (square_position[0] - throttle_text.get_width() - 25,
                                    square_position[1] + square_size / 2 - throttle_text.get_height() / 2))
        screen.blit(steer_left_text, (square_position[0] - steer_left_text.get_width() - 5,
                                      square_position[1] + square_size / 2 - steer_left_text.get_height() / 2))
        screen.blit(steer_right_text, (
            square_position[0] + square_size + 5,
            square_position[1] + square_size / 2 - steer_right_text.get_height() / 2))
        screen.blit(throttle_top_text, (square_position[0] + square_size / 2 - throttle_top_text.get_width() / 2,
                                        square_position[1] - throttle_top_text.get_height() - 5))
        screen.blit(throttle_bottom_text, (square_position[0] + square_size / 2 - throttle_bottom_text.get_width() / 2,
                                           square_position[1] + square_size + 5))

        # Current_throttle is just 0 as we can't know that from carla
        current_throttle = 0

        # Calculate the positions for current and predicted steer/throttle
        current_pos = (
            # Take just left wheel steering angle for ease of use
            square_position[0] + (square_size / 2) * (1 + self.get_debug_info("wheel_steering_angles")[0]),
            square_position[1] + square_size - (square_size / 2) * (1 + current_throttle)
        )
        predicted_pos = (
            square_position[0] + (square_size / 2) * (1 + self.get_debug_info("predicted_action").steering),
            square_position[1] + square_size - (square_size / 2) * (1 + self.get_debug_info("predicted_action").drive)
        )

        # Draw circles for current and predicted steer/throttle
        pygame.draw.circle(screen, (0, 255, 0), (int(current_pos[0]), int(current_pos[1])), 5)  # Green for current
        pygame.draw.circle(screen, (255, 0, 0), (int(predicted_pos[0]), int(predicted_pos[1])), 5)  # Red for predicted

        # Optionally, you can draw a cross at the center (0,0) position
        center_pos = (square_position[0] + square_size / 2, square_position[1] + square_size / 2)
        pygame.draw.line(screen, (100, 100, 100), (center_pos[0] - 10, center_pos[1]),
                         (center_pos[0] + 10, center_pos[1]), 1)
        pygame.draw.line(screen, (100, 100, 100), (center_pos[0], center_pos[1] - 10),
                         (center_pos[0], center_pos[1] + 10), 1)

    def draw_accelerometer_graph(self, screen, accel_data, graph_area):
        # Define graph properties
        max_data_length = 100  # Number of data points displayed on the graph at any time
        max_accel_value = 30.  # Assuming accelerometer values range from -1 to 1

        if len(accel_data) > max_data_length:
            accel_data.pop(0)

        # Clear the graph area
        pygame.draw.rect(screen, (0, 128, 255),
                         pygame.Rect(graph_area),
                         4)  # Blue color and thicker border

        for x in range(2):
            for i in range(1, len(accel_data)):
                # Map data to graph coordinates
                x1 = graph_area[0] + (i - 1) * (graph_area[2] / max_data_length)
                y1 = graph_area[1] + (graph_area[3] / 2) * (1 - (accel_data[i - 1][x] / max_accel_value))
                x2 = graph_area[0] + i * (graph_area[2] / max_data_length)
                y2 = graph_area[1] + (graph_area[3] / 2) * (1 - (accel_data[i][x] / max_accel_value))

                # Draw line between previous and current data point
                pygame.draw.line(screen, (1 if x == 0 else 255, 0 if x == 1 else 255, 0), (x1, y1), (x2, y2), 1)

    def render_yolo_detections(self, screen, features, class_names, image_width, image_height):
        # Assume features is a flattened tensor [center_x, center_y, class, distance, center_x, center_y, ...]
        for i in range(0, len(features), 4):
            if features[i + 3] > 0:  # Check if the normalized distance is not zero
                # Scale x and y to match the dimensions of the Pygame display
                x = features[i] * image_width
                y = features[i + 1] * image_height

                # Define the circle's position and size
                center = (int(x), int(y))
                radius = 5

                # Draw the circle on the Pygame surface
                pygame.draw.circle(screen, (255, 0, 0), center, radius)

                # Determine the class ID and calculate the actual distance
                class_id = int(features[i + 2])
                distance = features[i + 3]

                # Setup font and render the text
                font = pygame.font.Font(None, 24)  # Use default font and adjust size as needed
                text = f"{class_names[class_id]}, {distance:.2f}"  # Display the class and distance
                text_surface = font.render(text, True, (255, 0, 0))
                screen.blit(text_surface, (int(x), int(y)))

    def start_pygame_visuals(self):
        pygame.init()
        pygame.display.set_caption(f"DEBUG Camera View")
        self.game_display = pygame.display.set_mode((self.display_width, self.display_height), pygame.HWSURFACE | pygame.DOUBLEBUF)
        pygame.display.flip()