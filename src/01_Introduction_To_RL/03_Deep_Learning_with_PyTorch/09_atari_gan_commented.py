# ========================================
# IMPORTS
# ========================================
# General purpose libraries
import cv2                    # For image resizing and color transformations
import time                   # For tracking training duration and reporting intervals
import random                 # For random environment selection
import argparse               # For parsing command line arguments
import typing as tt            # For type hints

# PyTorch core libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard.writer import SummaryWriter  # For logging to TensorBoard

# torchvision utilities for image grid visualization
import torchvision.utils as vutils

# Gymnasium (OpenAI Gym successor) for RL environments
import gymnasium as gym
from gymnasium import spaces

# Numerical computing
import numpy as np

# ========================================
# LOGGER CONFIGURATION
# ========================================
log = gym.logger
log.set_level(gym.logger.INFO)  # Configure Gym’s logger to show info-level messages


# ========================================
# GLOBAL CONSTANTS
# ========================================
LATENT_VECTOR_SIZE = 100        # Size of the generator input noise vector (z)
DISCR_FILTERS = 64              # Base number of filters for discriminator
GENER_FILTERS = 64              # Base number of filters for generator
BATCH_SIZE = 16                 # Number of images per training iteration

IMAGE_SIZE = 64                 # All environment frames will be resized to 64x64
LEARNING_RATE = 0.0001          # Adam optimizer learning rate
REPORT_EVERY_ITER = 100         # Interval to print training losses
SAVE_IMAGE_EVERY_ITER = 1000    # Interval to save generated images to TensorBoard


# ========================================
# INPUT WRAPPER CLASS
# ========================================
class InputWrapper(gym.ObservationWrapper):
    """
    This wrapper preprocesses raw environment frames before feeding them to the GAN.
    Steps:
      1. Resize image to (IMAGE_SIZE, IMAGE_SIZE)
      2. Reorder axes from (H, W, C) to (C, H, W) so PyTorch can read them
    """

    def __init__(self, *args):
        super(InputWrapper, self).__init__(*args)
        old_space = self.observation_space
        assert isinstance(old_space, spaces.Box)  # Ensure observation is continuous (pixel array)
        # Define new observation space after preprocessing
        self.observation_space = spaces.Box(
            self.observation(old_space.low),   # Apply preprocessing to lower bounds
            self.observation(old_space.high),  # Apply preprocessing to upper bounds
            dtype=np.float32
        )

    def observation(self, observation: gym.core.ObsType) -> gym.core.ObsType:
        # Resize frame to 64x64 pixels
        new_obs = cv2.resize(observation, (IMAGE_SIZE, IMAGE_SIZE))
        # Move color channels to the first dimension -> (C, H, W)
        new_obs = np.moveaxis(new_obs, 2, 0)
        return new_obs.astype(np.float32)


# ========================================
# DISCRIMINATOR NETWORK
# ========================================
class Discriminator(nn.Module):
    """
    The Discriminator tries to distinguish between real and fake images.
    Input: (C, 64, 64)
    Output: scalar probability that input is real.
    """
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()

        # A series of convolutional layers that downsample image progressively
        self.conv_pipe = nn.Sequential(
            # 1st Conv block
            nn.Conv2d(in_channels=input_shape[0], out_channels=DISCR_FILTERS,
                      kernel_size=4, stride=2, padding=1),
            nn.ReLU(),

            # 2nd Conv block
            nn.Conv2d(DISCR_FILTERS, DISCR_FILTERS * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(DISCR_FILTERS * 2),
            nn.ReLU(),

            # 3rd Conv block
            nn.Conv2d(DISCR_FILTERS * 2, DISCR_FILTERS * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(DISCR_FILTERS * 4),
            nn.ReLU(),

            # 4th Conv block
            nn.Conv2d(DISCR_FILTERS * 4, DISCR_FILTERS * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(DISCR_FILTERS * 8),
            nn.ReLU(),

            # Final layer – collapse to a single scalar output (real/fake probability)
            nn.Conv2d(DISCR_FILTERS * 8, 1, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid()  # Bound output between 0 and 1
        )
    
    def forward(self, x):
        conv_out = self.conv_pipe(x)
        # Flatten output to a vector of size (batch_size)
        return conv_out.view(-1, 1).squeeze(dim=1)


# ========================================
# GENERATOR NETWORK
# ========================================
class Generator(nn.Module):
    """
    The Generator takes a latent noise vector and "deconvolves" it into an image.
    Input: (LATENT_VECTOR_SIZE, 1, 1)
    Output: (3, 64, 64)
    """
    def __init__(self, output_shape):
        super(Generator, self).__init__()

        self.pipe = nn.Sequential(
            # 1st deconv block
            nn.ConvTranspose2d(LATENT_VECTOR_SIZE, GENER_FILTERS * 8,
                               kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(GENER_FILTERS * 8),
            nn.ReLU(),

            # 2nd deconv block
            nn.ConvTranspose2d(GENER_FILTERS * 8, GENER_FILTERS * 4,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(GENER_FILTERS * 4),
            nn.ReLU(),

            # 3rd deconv block
            nn.ConvTranspose2d(GENER_FILTERS * 4, GENER_FILTERS * 2,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(GENER_FILTERS * 2),
            nn.ReLU(),

            # 4th deconv block
            nn.ConvTranspose2d(GENER_FILTERS * 2, GENER_FILTERS,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(GENER_FILTERS),
            nn.ReLU(),

            # Final layer outputs 3-channel RGB image
            nn.ConvTranspose2d(GENER_FILTERS, output_shape[0],
                               kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # Scale pixel values to [-1, 1]
        )

    def forward(self, x):
        return self.pipe(x)


# ========================================
# BATCH GENERATOR FUNCTION
# ========================================
def iterate_batches(envs: tt.List[gym.Env], batch_size: int = BATCH_SIZE) -> tt.Generator[torch.Tensor, None, None]:
    """
    Continuously collect batches of observations (images) from multiple Gym environments.
    Each batch will contain `batch_size` images, normalized to [-1, 1].
    """
    batch = [e.reset()[0] for e in envs]  # Start each environment and collect initial frame
    env_gen = iter(lambda: random.choice(envs), None)  # Randomly sample an environment each iteration

    while True:
        e = next(env_gen)
        action = e.action_space.sample()  # Take a random action
        obs, reward, is_done, is_trunc, _ = e.step(action)

        # Filter out near-black frames (helps stabilize training)
        if np.mean(obs) > 0.01:
            batch.append(obs)

        # Once enough frames are collected, normalize and yield
        if len(batch) == batch_size:
            batch_np = np.array(batch, dtype=np.float32)
            # Normalize to [-1, 1] for Tanh compatibility
            yield torch.tensor(batch_np * 2.0 / 255.0 - 1.0)
            batch.clear()

        # If the episode ended, reset that environment
        if is_done or is_trunc:
            e.reset()


# ========================================
# MAIN TRAINING LOOP
# ========================================
if __name__ == "__main__":
    # Command line argument to set device (CPU or GPU)
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", default="cpu", help="Device name, default=cpu")
    args = parser.parse_args()

    device = torch.device(args.dev)

    # Create multiple Atari-like environments and wrap them for preprocessing
    envs = [
        InputWrapper(gym.make(name))
        for name in ('Breakout-v4', 'AirRaid-v4', 'Pong-v4')
    ]
    shape = envs[0].observation_space.shape  # Example shape: (3, 64, 64)

    # Initialize Generator and Discriminator
    net_discr = Discriminator(input_shape=shape).to(device)
    net_gener = Generator(output_shape=shape).to(device)

    # Binary cross-entropy loss for real/fake classification
    objective = nn.BCELoss()

    # Adam optimizers for both networks (DCGAN standard betas)
    gen_optimizer = optim.Adam(params=net_gener.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    dis_optimizer = optim.Adam(params=net_discr.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

    # TensorBoard writer for logging
    writer = SummaryWriter()

    # Containers for moving averages of losses
    gen_losses = []
    dis_losses = []
    iter_no = 0

    # Predefined label tensors
    true_labels_v = torch.ones(BATCH_SIZE, device=device)
    fake_labels_v = torch.zeros(BATCH_SIZE, device=device)
    ts_start = time.time()

    # Training loop fetching environment frames
    for batch_v in iterate_batches(envs):
        # ================================
        # GENERATE FAKE IMAGES
        # ================================
        gen_input_v = torch.FloatTensor(BATCH_SIZE, LATENT_VECTOR_SIZE, 1, 1)
        gen_input_v.normal_(0, 1)  # Sample random noise from N(0, 1)
        gen_input_v = gen_input_v.to(device)
        batch_v = batch_v.to(device)

        # Forward pass through generator
        gen_output_v = net_gener(gen_input_v)

        # ================================
        # TRAIN DISCRIMINATOR
        # ================================
        dis_optimizer.zero_grad()

        # 1. Predict on real images
        dis_output_true_v = net_discr(batch_v)
        # 2. Predict on fake images (detach so generator isn't updated yet)
        dis_output_fake_v = net_discr(gen_output_v.detach())

        # Compute discriminator loss on both real and fake batches
        dis_loss = objective(dis_output_true_v, true_labels_v) + objective(dis_output_fake_v, fake_labels_v)
        dis_loss.backward()
        dis_optimizer.step()
        dis_losses.append(dis_loss.item())

        # ================================
        # TRAIN GENERATOR
        # ================================
        gen_optimizer.zero_grad()
        # Re-run discriminator on fake images (this time with gradients)
        dis_output_v = net_discr(gen_output_v)
        # Generator’s goal: fool discriminator -> wants discriminator to output "real" (1)
        gen_loss_v = objective(dis_output_v, true_labels_v)
        gen_loss_v.backward()
        gen_optimizer.step()
        gen_losses.append(gen_loss_v.item())

        # ================================
        # LOGGING AND VISUALIZATION
        # ================================
        iter_no += 1

        # Every REPORT_EVERY_ITER, print and log to TensorBoard
        if iter_no % REPORT_EVERY_ITER == 0:
            dt = time.time() - ts_start
            log.info("Iter %d in %.2fs: gen_loss=%.3e, dis_loss=%.3e",
                     iter_no, dt, np.mean(gen_losses), np.mean(dis_losses))
            ts_start = time.time()
            writer.add_scalar("gen_loss", np.mean(gen_losses), iter_no)
            writer.add_scalar("dis_loss", np.mean(dis_losses), iter_no)
            gen_losses = []
            dis_losses = []

        # Every SAVE_IMAGE_EVERY_ITER, save image grids for both real and fake images
        if iter_no % SAVE_IMAGE_EVERY_ITER == 0:
            img = vutils.make_grid(gen_output_v.data[:64], normalize=True)
            writer.add_image("fake", img, iter_no)
            img = vutils.make_grid(batch_v.data[:64], normalize=True)
            writer.add_image("real", img, iter_no)
