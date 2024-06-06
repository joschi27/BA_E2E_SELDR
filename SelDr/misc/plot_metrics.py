import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d
import seaborn as sns


def calculate_statistics(json_file):
    # Load data from the JSON file
    with open(json_file, 'r') as file:
        data = json.load(file)

    metrics = data['metrics']

    # Calculating Average Completion Rate and Average Speed
    completion_rates = [run['Completion Rate'] for run in metrics]
    speeds = [run['Average Speed'] for run in metrics]
    crash_statuses = [run['Crashed'] for run in metrics]

    average_completion_rate = np.mean(completion_rates)
    average_speed = np.mean(speeds)

    # Calculating Crash Rate
    crash_rate = 100 * sum(crash_statuses) / len(crash_statuses)

    print(f"Average Completion Rate: {average_completion_rate:.2f}%")
    print(f"Average Speed: {average_speed:.2f} m/s")
    print(f"Crash Rate: {crash_rate:.2f}%")


def plot_crash_and_cone_locations_heatmap(crash_json_file, cone_json_file):
    # Load crash data from the JSON file
    with open(crash_json_file, 'r') as file:
        crash_data = json.load(file)
    crash_metrics = crash_data['metrics']

    # Extracting crash locations, check for 'Crashed' and 'Crash Locations' existence
    crash_locations = [run['Crash Locations'] for run in crash_metrics if run.get('Crashed') and 'Crash Locations' in run]
    if not crash_locations:
        print("No crash locations recorded.")
        return

    # Separating x and y coordinates for crashes
    crash_x = [-loc[0] for loc in crash_locations]
    crash_y = [loc[1] for loc in crash_locations]

    # Load cone data from the JSON file
    with open(cone_json_file, 'r') as file:
        cones = json.load(file)

    # Extracting cone locations and class ids
    cone_x = [-(cone[0] - start_location_x) for cone in cones]
    cone_y = [cone[1] - start_location_y for cone in cones]
    cone_types = [cone[2] for cone in cones]

    # Map cone class ids to colors for visualization
    cone_colors = {0: 'gold', 1: 'orange', 2: 'blue', 3: 'yellow'}
    cone_color_names = {0: 'Big Orange Cone', 1: 'Orange Cone', 2: 'Blue Cone', 3: 'Yellow Cone'}

    # Create a scatter plot for crash locations
    plt.figure(figsize=(6, 6))
    plt.scatter(crash_x, crash_y, c='red', s=50, alpha=0.5, edgecolors='none', marker='o', label='Crash Locations')

    # Overlay cone locations
    plt.scatter(cone_x, cone_y, c=[cone_colors[cid] for cid in cone_types], s=10, edgecolors='black', marker='o', alpha=0.5)

    # Correcting the aspect ratio for the plot
    plt.gca().set_aspect('equal', adjustable='datalim')
    plt.axis('off')  # Turn off the axis
    plt.grid(False)  # Disable the grid

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Create a legend for crash locations and cone types
    legend_handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, markeredgecolor='none', label='Crash Locations')
    ] + [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cone_colors[i], markersize=10, markeredgecolor='black', label=cone_color_names[i])
        for i in cone_colors
    ]
    plt.legend(handles=legend_handles, title="Legend", loc='upper left', bbox_to_anchor=(0.1, 0.9))

    # Add a distance marker (10m) to the plot at the bottom right
    x_min, x_max = plt.gca().get_xlim()
    y_min, y_max = plt.gca().get_ylim()
    marker_x = x_max - 15
    marker_y = y_min + 5

    # Draw the striped horizontal line
    for i in range(10):
        color = 'k' if i % 2 == 0 else 'w'
        plt.plot([marker_x + i, marker_x + i + 1], [marker_y, marker_y], color=color, lw=1)

    # Draw the start and end vertical bars
    plt.plot([marker_x, marker_x], [marker_y - 1, marker_y + 1], 'k-', lw=1)
    plt.plot([marker_x + 10, marker_x + 10], [marker_y - 1, marker_y + 1], 'k-', lw=1)

    # Add the label
    plt.text(marker_x + 5, marker_y - 2, '10m', ha='center', color='black')

    # Save and show the plot
    plt.savefig('crash_chart.svg', format='svg')
    plt.show()


def plot_speed_spline_and_cones(json_file, cone_json_file):
    # Load path and speed data
    with open(json_file, 'r') as file:
        data = json.load(file)
    paths = data['metrics']

    # Load cone data
    with open(cone_json_file, 'r') as file:
        cones = json.load(file)

    # Preparing cone data
    cone_x = [-(cone[0] - start_location_x) for cone in cones]
    cone_y = [cone[1] - start_location_y for cone in cones]
    cone_types = [cone[2] for cone in cones]

    # Accel
    # plt.figure(figsize=(3, 6))

    # Skidpad
    plt.figure(figsize=(6, 6))

    # Trackdrive
    # plt.figure(figsize=(6, 6))

    ax = plt.gca()

    # Find global minimum and maximum speeds
    global_min_speed = min(min(p[2] for p in path['Path with Speeds']) for path in paths)
    global_max_speed = max(max(p[2] for p in path['Path with Speeds']) for path in paths)

    # Create a normalization object based on these global extremes
    norm = plt.Normalize(global_min_speed, global_max_speed)

    # Define the color map based on viridis_r with red instead of yellow
    colors = [(0.267004, 0.004874, 0.329415),  # viridis_r purple
              (0.229739, 0.322361, 0.545706),  # viridis_r blue
              (0.127568, 0.566949, 0.550556),  # viridis_r green
              (0.993248, 0.906157, 0.143936),  # original viridis_r yellow
              (0.900000, 0.100000, 0.000000)]  # replace with red
    custom_cmap = LinearSegmentedColormap.from_list("custom_viridis_r", colors)

    for path in paths:
        x = [-p[0] for p in path['Path with Speeds']]
        y = [p[1] for p in path['Path with Speeds']]
        speeds = [p[2] for p in path['Path with Speeds']]

        # Number of points for fine interpolation
        num_fine_points = 300

        # Interpolate path and speeds
        t = np.linspace(0, 1, len(x))
        cs_x = CubicSpline(t, x)
        cs_y = CubicSpline(t, y)
        cs_speeds = CubicSpline(t, speeds)
        t_fine = np.linspace(0, 1, num_fine_points)
        x_fine = cs_x(t_fine)
        y_fine = cs_y(t_fine)
        speeds_fine = cs_speeds(t_fine)

        # Prepare for plotting
        points = np.array([x_fine, y_fine]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap=custom_cmap, norm=norm, alpha=1)
        lc.set_array(speeds_fine)
        lc.set_linewidth(2)
        ax.add_collection(lc)

    # Plot cones
    cone_colors = {0: 'gold', 1: 'orange', 2: 'blue', 3: 'yellow'}
    plt.scatter(cone_x, cone_y, c=[cone_colors[cid] for cid in cone_types], s=10, edgecolors='black', marker='o', alpha=0.5)

    plt.colorbar(lc, ax=ax, label='Speed (m/s)', shrink=0.5)
    plt.axis('equal')
    ax.set_xlim(ax.get_xlim()[0] * 1, ax.get_xlim()[1] * 1)
    plt.axis('off')
    plt.grid(False)

    # Add a distance marker (10m) to the plot at the bottom right
    x_min, x_max = plt.gca().get_xlim()
    y_min, y_max = plt.gca().get_ylim()
    marker_x = x_max - 15
    marker_y = y_min + 5

    # Draw the striped horizontal line
    for i in range(10):
        color = 'k' if i % 2 == 0 else 'w'
        plt.plot([marker_x + i, marker_x + i + 1], [marker_y, marker_y], color=color, lw=1)

    # Draw the start and end vertical bars
    plt.plot([marker_x, marker_x], [marker_y - 1, marker_y + 1], 'k-', lw=1)
    plt.plot([marker_x + 10, marker_x + 10], [marker_y - 1, marker_y + 1], 'k-', lw=1)

    # Add the label
    plt.text(marker_x + 5, marker_y - 5, '10m', ha='center', color='black')

    #plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    plt.savefig('speed_chart.svg', format='svg')
    plt.show()

# Acceleration
# file = '..\metrics\ppo_agent_episode_900_MultiAgentEnvAcceleration.h5_metrics.json'
# cone_file = '..\metrics\\accel_eval_cones_location.json'
# start_location_x = 312.6
# start_location_y = 357.5

# Skidpad
file = '..\metrics\ppo_agent_episode_9900_MultiAgentEnvSkidPad.h5_metrics.json'
cone_file = '..\metrics\\skidpad_eval_cones_location.json'
start_location_x = 94.0
start_location_y = 355.5

# Trackdrive
# file = '..\metrics\ppo_agent_episode_6000_MultiAgentEnvTrackDrive.h5_metrics.json'
# cone_file = '..\metrics\\trackdrive_eval_cones_location.json'
# start_location_x = 70.8
# start_location_y = 373.2

calculate_statistics(file)
plot_crash_and_cone_locations_heatmap(file, cone_file)
plot_speed_spline_and_cones(file, cone_file)

