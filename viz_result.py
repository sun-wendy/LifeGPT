#!/usr/bin/env python
import os
import re
import glob
import cv2
import time
import imageio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from matplotlib.animation import FuncAnimation
from einops import repeat

# Set environment variable to avoid duplicate lib issues (if needed)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# Autoregressive generation
class Tokenizer:
    def __init__(self, n_pad: int, device: torch.device, pad_byte: int = 0):
        self.n_pad = n_pad
        self.device = device
        self.pad_byte = pad_byte

    def tokenize_str(self, sentence: str, encoding="utf8", do_padding=True):
        base = list(bytes(sentence, encoding))
        if do_padding:
            if len(base) < self.n_pad:
                base.extend([self.pad_byte] * (self.n_pad - len(base)))
            assert len(base) == self.n_pad, f"n_pad is too small, use {len(base)} or greater."
        tensor = torch.Tensor(base)
        return tensor.long().to(self.device)

    def texts_to_sequences(self, texts: list, encoding="utf8", do_padding=True):
        sentences = [self.tokenize_str(sentence, do_padding=do_padding).unsqueeze(0) for sentence in texts]
        return torch.cat(sentences, dim=0).to(self.device)

    def sequences_to_texts(self, texts: torch.Tensor, encoding="utf8"):
        out = []
        for seq in texts:
            chars = []
            i = 0
            while i < len(seq) and seq[i] != 0:
                chars.append(int(seq[i]))
                i += 1
            try:
                out.append(bytes(chars).decode(encoding))
            except:
                pass
        return out


def empty_cuda_cache():
    torch.cuda.empty_cache()


def extract_task(string_input, end_task_token='>'):
    j = string_input.find(end_task_token)
    return string_input[:j+1]

def extract_sample(string_input, start_token='[', end_token=']'):
    i = string_input.find(start_token)
    j = string_input.find(end_token)
    return string_input[i+1:j]


# Load model
# (Assume your model is built using x_transformers and wrapped in AutoregressiveWrapper)
def load_model(model_path, max_length, num_words):
    empty_cuda_cache()
    from x_transformers import TransformerWrapper, Decoder
    from x_transformers.autoregressive_wrapper import AutoregressiveWrapper
    model = TransformerWrapper(
        num_tokens=num_words,
        max_seq_len=max_length,
        attn_layers=Decoder(
            dim=256,
            depth=12,
            heads=8,
            attn_dim_head=64,
            rotary_pos_emb=True,
            attn_flash=True
        )
    )
    model = AutoregressiveWrapper(model)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()  # call eval to set dropouts to evaluation mode
    print(f"Model loaded from {model_path}")
    return model


# Autoregressive generation
def autoregressive_generation(model, tokenizer, initial_input_str, num_steps, generate_length, temp):
    """
    Given an initial input string, autoregressively generates num_steps states.
    The prompt format is assumed to be: "@PredictNextState<STATE1> [...]$"
    """
    current_input = extract_task(initial_input_str, end_task_token='>')
    generated_states = []
    for step in range(num_steps):
        current_input_tensor = torch.Tensor(tokenizer.texts_to_sequences(current_input, do_padding=False)).to(device)
        current_input_tensor = current_input_tensor.transpose(0, 1).long()
        with torch.no_grad():
            sample = model.generate(
                prompts=current_input_tensor,
                seq_len=generate_length,
                temperature=temp,
                cache_kv=True
            )
        output_str = extract_sample(tokenizer.sequences_to_texts(sample[:1])[0])
        generated_states.append(output_str)
        current_input = f"@PredictNextState<{output_str}>"
        del sample, current_input_tensor
        torch.cuda.empty_cache()
    return generated_states


# Save Data to CSV
def parse_and_convert_ground_truth(df, row_index):
    """
    For a given row index, extract all ground truth states.
    Assumes states are stored in columns named "State 1", "State 2", ... in the DataFrame.
    """
    # Extract the state columns from the row (skipping any non-state column)
    row = df.iloc[row_index]
    # Assuming the first column might be an identifier, get only columns that start with "State"
    states = [row[col] for col in df.columns if col.lower().startswith("state")]
    return states

def save_data_to_csv(df, row_indices, autoreg_list, temp, base_dir="AR_result"):
    """
    Saves the ground truth and autoregressively generated data into a CSV file.
    Two rows per sample are saved: one for Ground Truth and one for Autoregressive prediction.
    """
    os.makedirs(base_dir, exist_ok=True)
    data = []
    for i, row_index in enumerate(row_indices):
        gt_states = parse_and_convert_ground_truth(df, row_index)
        autoreg_states = autoreg_list[i]
        num_steps = len(autoreg_states)
        initial_state = gt_states[0]
        gt_row = [initial_state] + gt_states[1:num_steps+1]
        autoreg_row = [initial_state] + autoreg_states
        data.append(["Ground Truth"] + gt_row)
        data.append(["Autoregressive"] + autoreg_row)
    columns = ["Type"] + [f"State {i+1}" for i in range(len(gt_row))]
    data_df = pd.DataFrame(data, columns=columns).astype(str)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    csv_path = os.path.join(base_dir, f"data_temp_{temp}_{timestamp}.csv")
    data_df.to_csv(csv_path, index=False)
    print(f"Saved data to CSV at {csv_path}")


# Visualization
def convert_to_square_image(state, cell_size=1):
    """
    Converts a binary string (of 0s and 1s) into a square 2D numpy array.
    The state length must be a perfect square.
    If cell_size > 1, each cell is repeated accordingly.
    """
    length = len(state)
    size = int(np.sqrt(length))
    assert size * size == length, "State length is not a perfect square"
    image = np.array([int(pixel) for pixel in state]).reshape((size, size))
    # Increase cell size by repeating rows and cols
    image = np.kron(image, np.ones((cell_size, cell_size)))
    return image

def is_perfect_square(length):
    size = int(np.sqrt(length))
    return size * size == length

def check_state_length(state):
    return len(state) == 1024  # Assuming 32x32 grid

def plot_image(ax, state, ylabel, cmap):
    ax.set_ylabel(ylabel, fontsize=10, rotation=90, labelpad=10, va='center')
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    if not is_perfect_square(len(state)) or not check_state_length(state):
        ax.text(0.5, 0.5, "DATA NOT SQUARE", fontsize=10, ha='center', va='center', rotation=45)
    else:
        ax.imshow(1 - convert_to_square_image(state, cell_size=10), cmap=cmap, vmin=0, vmax=1)

def plot_discrepancy(ax, discrepancy_matrix, ylabel):
    sns.heatmap(discrepancy_matrix, cmap="plasma", cbar=False, square=True, ax=ax)
    ax.set_ylabel(ylabel, fontsize=10, rotation=90, labelpad=10, va='center')
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

def calculate_discrepancy_matrix(gt_state, pred_state):
    if len(gt_state) == len(pred_state):
        discrepancy_matrix = np.array([
            1 if gt_state[k] != pred_state[k] else 0
            for k in range(len(gt_state))
        ]).reshape((32, 32))
    else:
        print('State lengths differ; using default discrepancy.')
        discrepancy_matrix = np.full((32, 32), 0.5)
    return discrepancy_matrix

def create_gifs_from_csv(csv_path_pattern, temp, output_dir="Figures", fps=5):
    """
    Creates looping GIFs and MP4 videos from CSV files containing ground truth and autoregressive states.
    Each sample yields a figure with three rows: GT, Predicted, and their Discrepancy.
    """
    csv_files = glob.glob(csv_path_pattern)
    if not csv_files:
        print(f"No CSV files found for temperature {temp}")
        return
    csv_path = csv_files[0]
    df = pd.read_csv(csv_path, dtype=str)
    num_columns = len([col for col in df.columns if "State" in col])
    num_samples = df.shape[0] // 2  # Since two rows per simulation
    os.makedirs(output_dir, exist_ok=True)
    for i in range(num_samples):
        frames = []
        for j in range(0, num_columns):  # For each state (starting from State 2)
            fig, axes = plt.subplots(3, 1, figsize=(3, 9), dpi=300)
            gt_row = df.iloc[2 * i, 1:]
            pred_row = df.iloc[2 * i + 1, 1:]
            # Plot ground truth
            plot_image(axes[0], gt_row.iloc[j], "GT", "gray")
            cell = pred_row.iloc[j]
            # Plot predicted state
            plot_image(axes[1], pred_row.iloc[j], "Predicted", "gray")
            # Plot discrepancy matrix
            discrepancy = calculate_discrepancy_matrix(gt_row.iloc[j], pred_row.iloc[j])
            plot_discrepancy(axes[2], discrepancy, "Discrepancy")
            # fig.suptitle(f"Sample {i+1} | Temp: {temp} | Epoch: {epoch}/50", fontsize=10)
            plt.tight_layout(pad=2.0)
            temp_frame = os.path.join(output_dir, f"temp_frame_{j}.png")
            plt.savefig(temp_frame, dpi=300)
            plt.close(fig)
            frames.append(Image.open(temp_frame))
        # Create GIF and MP4
        gif_path = os.path.join(output_dir, f"evolution_temp_{temp}_sample_{i+1}.gif")
        frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=1000//fps, loop=0)
        mp4_path = os.path.join(output_dir, f"evolution_temp_{temp}_sample_{i+1}.mp4")
        height, width = frames[0].size[1], frames[0].size[0]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_out = cv2.VideoWriter(mp4_path, fourcc, fps, (width, height))
        for frame in frames:
            video_out.write(np.array(frame.convert("RGB")))
        video_out.release()
        print(f"Saved GIF at {gif_path}")
        print(f"Saved MP4 at {mp4_path}")
        for frame in frames:
            os.remove(frame.filename)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # test_file = "Conway_GPT/conway_states_0_1_100by32by32by256_toroidal_20250412_194900.csv"
    test_file = "LifeGPT/conway_test_states_32by32_20240827_172807.csv"
    df_test = pd.read_csv(test_file)
    
    # Generate autoregressive input strings from test data
    start_char = '@'
    end_char = '$'
    def generate_data(df, future_steps):
        X_data = []
        for i in range(len(df['State 1'])):
            future_state_col = f'State {future_steps}'
            if future_state_col in df.columns:
                s = f"{start_char}PredictNextState<{df['State 1'][i]}> [{df[future_state_col][i]}]{end_char}"
                X_data.append(s)
        return X_data

    future_steps_list = [32]  # example; adjust as needed
    X_data_test = {steps: generate_data(df_test, steps) for steps in future_steps_list}
    X_data_test_new = X_data_test[32]

    # Calculate maximum length & generate_length dynamically
    max_length = len(X_data_test_new[0])
    GENERATE_LENGTH = max_length - len(extract_task(X_data_test[32][0],end_task_token='>'))
    print("generate_length = ", GENERATE_LENGTH)

    tokenizer_X = Tokenizer(max_length, device)
    model_path = f"LifeGPT/model_parameters/07_22_2024_Conway_2_State_Jump_Rot_Pos_On_Masking_On_Broad_Entropy_Homog_2025-04-12 08-39-12/LifeGPT_v7_epoch_30.pt"
    model = load_model(model_path, max_length, num_words=256)

    # For each temperature, perform autoregressive generation and save data to CSV.
    temperatures = [0]
    num_steps = 31  # Number of steps to generate (excluding the initial state)
    ground_truth_df = df_test  # Using df_test as ground truth
    
    # List to hold row indices (assume each test input corresponds to a row in df_test)
    for temp in temperatures:
        print(f"Generating for temperature {temp}...")
        autoregressive_states = []
        row_indices = []
        selected_indices = list(range(2, 9))
        for item_idx in selected_indices:
            initial_input_str = str(X_data_test_new[item_idx])
            gen_states = autoregressive_generation(model, tokenizer_X, initial_input_str, num_steps, GENERATE_LENGTH, temp)
            autoregressive_states.append(gen_states)
            row_indices.append(item_idx)
            del gen_states
            torch.cuda.empty_cache()
        save_data_to_csv(ground_truth_df, row_indices, autoregressive_states, temp)

    # Visualization
    from PIL import Image
    output_gif_dir = "Figures"
    os.makedirs(output_gif_dir, exist_ok=True)
    for temp in temperatures:
        csv_path_pattern = os.path.join("AR_result", f"data_temp_{temp}_*.csv")
        create_gifs_from_csv(csv_path_pattern, temp, output_dir=output_gif_dir, fps=5)
