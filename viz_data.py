#!/usr/bin/env python
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def cell_to_image(cell_str, grid_size=(32, 32)):
    """
    Convert a string representing a flattened binary grid into a 2D numpy array.
    This example assumes the string consists of '0' and '1' characters.
    """
    digits = re.findall(r'[01]', cell_str)
    arr = np.array(list(map(int, digits)))
    if len(arr) != grid_size[0] * grid_size[1]:
        raise ValueError(f"Expected {grid_size[0]*grid_size[1]} digits, but got {len(arr)}")
    arr = arr.reshape(grid_size)
    return arr  # Return a 2D array; we can display it as a grayscale image


def create_animation(images, output_path, interval=500):
    """
    Create and save an animation (GIF) from a list of 2D image arrays.
    
    Args:
      images (list of numpy.ndarray): List of image frames.
      output_path (str): Path to save the resulting GIF.
      interval (int): Delay between frames in milliseconds.
    """
    fig, ax = plt.subplots()
    # Display the first frame
    im = ax.imshow(images[0], cmap='gray', vmin=0, vmax=1)
    ax.axis("off")  # Hide axes for animation

    def update(frame):
        im.set_data(images[frame])
        return [im]

    anim = FuncAnimation(fig, update, frames=len(images), interval=interval, blit=True)
    anim.save(output_path, writer='imagemagick')
    plt.close(fig)


def main(csv_file, grid_size=(32, 32), output_folder='animations'):
    """
    Process each simulation (row) in the CSV file, convert its states to images,
    generate an animated GIF, and save it in the output folder.
    
    Args:
      csv_file (str): Path to the CSV file containing simulation states.
      grid_size (tuple): Expected grid size (rows, cols) for each state.
      output_folder (str): Directory in which to save the animations.
    """
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    df = pd.read_csv(csv_file)
    # Assume state columns are named like "State 1", "State 2", etc.
    state_columns = [col for col in df.columns if col.lower().startswith("state")]
    
    for idx, row in df.iterrows():
        images = []
        for col in state_columns:
            state_str = str(row[col])
            try:
                img = cell_to_image(state_str, grid_size=grid_size)
                images.append(img)
            except Exception as e:
                print(f"Error processing row {idx}, column {col}: {e}")
        if images:
            output_path = os.path.join(output_folder, f"simulation_{idx}.gif")
            try:
                create_animation(images, output_path)
                print(f"Saved animation for simulation {idx} to {output_path}")
            except Exception as e:
                print(f"Error saving animation for simulation {idx}: {e}")
        else:
            print(f"No valid images for simulation {idx}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate animations from a Conway simulation CSV.")
    parser.add_argument("--csv", type=str, required=True, help="Path to the CSV file containing simulation states.")
    parser.add_argument("--grid_rows", type=int, default=32, help="Number of rows in the state grid.")
    parser.add_argument("--grid_cols", type=int, default=32, help="Number of columns in the state grid.")
    parser.add_argument("--output_folder", type=str, default="animations", help="Folder to save animations.")
    parser.add_argument("--interval", type=int, default=500, help="Interval between frames in milliseconds.")
    args = parser.parse_args()

    main(args.csv, grid_size=(args.grid_rows, args.grid_cols), output_folder=args.output_folder)
