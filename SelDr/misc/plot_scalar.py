import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from tensorflow.python.summary.summary_iterator import summary_iterator


def extract_scalars(log_dir, tags, max_episodes=10000):
    """Extracts scalar values from a TensorBoard log directory for specific tags, up to a maximum number of episodes."""
    scalar_data = {tag: [] for tag in tags}
    steps = []

    for file_name in os.listdir(log_dir):
        file_path = os.path.join(log_dir, file_name)
        if os.path.isfile(file_path):
            for e in summary_iterator(file_path):
                step = e.step
                if step >= max_episodes:
                    break  # Stop if we've reached the maximum number of episodes
                while len(steps) <= step:
                    steps.append(len(steps))
                    for tag in tags:
                        scalar_data[tag].append(float('nan'))  # Fill with NaN to maintain alignment

                for v in e.summary.value:
                    if v.tag in tags:
                        scalar_data[v.tag][step] = v.simple_value

    # Truncate the data to the maximum number of episodes
    steps = steps[:max_episodes]
    for tag in tags:
        scalar_data[tag] = scalar_data[tag][:max_episodes]

    data = {tag: pd.Series(values) for tag, values in scalar_data.items()}
    data['step'] = pd.Series(steps)
    return pd.DataFrame(data)



def plot_scalars(df, output_file, isAccel=False, isSkidpad=False, isTrackdrive=False):
    """Creates and saves custom line plots with shaded error bounds."""
    plt.figure(figsize=(12, 7))

    for i, (column, title, color) in enumerate(zip(['reward_avg', 'speed_avg_ms', 'survived_steps_avg', 'action_std'],
                                                   ['Average Reward', 'Average Speed (m/s)', 'Average Survived Steps', 'Action Standard Deviation'],
                                                   ['green', 'orange', 'blue', 'gray']), 1):
        x = df['step']
        y = df[column]

        if column == 'action_std':
            ax = plt.subplot(2, 2, i)
            ax.plot(x, y, color=color)
            ax.set_title(title, fontsize=18)
            ax.set_xlabel('Episodes')
            ax.set_ylabel('Value')
            continue

        # Smooth the original data
        y_smoothed = gaussian_filter1d(y, sigma=5)

        # Calculate bin size
        bin_size = len(df['step']) // 40

        # Compute the binned max and min values for fill_between
        bins = np.array_split(y, bin_size)
        upper = np.concatenate([np.full(len(bin), bin.max()) for bin in bins])
        lower = np.concatenate([np.full(len(bin), bin.min()) for bin in bins])

        # Smooth the upper and lower bounds to make the edges continuous and smooth
        upper_smoothed = gaussian_filter1d(upper, sigma=10)
        lower_smoothed = gaussian_filter1d(lower, sigma=10)

        ax = plt.subplot(2, 2, i)
        ax.plot(x, y_smoothed, color=color)
        ax.fill_between(x, upper_smoothed, lower_smoothed, alpha=0.2, color=color)
        ax.set_title(title, fontsize=18)
        ax.set_xlabel('Episodes')
        ax.set_ylabel('Value')

        # For trackdrive as there are some outliers
        if column == "reward_avg" and isTrackdrive or column == "reward_avg" and isSkidpad:
            ax.set_ylim(-1000, 1500)

    plt.tight_layout()
    plt.savefig(output_file, format='svg')
    #plt.show()
    print(f"Chart saved as {output_file}")


def main(log_dir, output_file):
    tags = ['reward_avg', 'speed_avg_ms', 'survived_steps_avg', 'action_std']
    print(f"Extracting scalars from {log_dir} for tags {tags}")
    df = extract_scalars(log_dir, tags)
    if not df.empty:
        print(f"Found data, creating plot...")
        plot_scalars(df, output_file, isSkidpad=True)
    else:
        print(f"No scalar data found for tags {tags} in directory '{log_dir}'")


if __name__ == "__main__":
    #log_directory = '..\\logs\\run_2024-05-26_16-42-09_MultiAgentEnvAcceleration'
    #log_directory = '..\\logs\\run_2024-05-20_16-15-59_MultiAgentEnvTrackDrive'
    log_directory = '..\\logs\\run_2024-05-26_17-12-31_MultiAgentEnvSkidPad'
    output_svg_file = 'scalar_chart.svg'  # Output SVG file name

    main(log_directory, output_svg_file)
    print(f"Chart saved as {output_svg_file}")
