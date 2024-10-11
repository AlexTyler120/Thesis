import pickle
import matplotlib.pyplot as plt
import numpy as np
# Function to load pickle file and return the data
def load_pickle_data(file_name):
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
    return data

# Load data from three channels
channel_1_data = load_pickle_data('loss_values_0_powell_03.pkl')
channel_2_data = load_pickle_data('loss_values_1_powell_03.pkl')
channel_3_data = load_pickle_data('loss_values_2_powell_03.pkl')

# Extract estimates and loss values
channel_1_estimates, channel_1_losses = zip(*channel_1_data)  # Unpack the estimates and losses
channel_2_estimates, channel_2_losses = zip(*channel_2_data)
channel_3_estimates, channel_3_losses = zip(*channel_3_data)
print(f"channel 1 loss: {np.abs(0.4 - channel_1_estimates[-1])}")
print(f"channel 2 loss: {np.abs(0.4 - (1 - channel_2_estimates[-1]))}")
print(f"channel 3 loss: {np.abs(0.4 - channel_3_estimates[-1])}")
print(f" Mean loss: {np.mean([np.abs(0.4 - channel_1_estimates[-1]), np.abs(0.4 - (1 - channel_2_estimates[-1])), np.abs(0.4 - channel_3_estimates[-1])])}")
print(f"Max number of iterations {np.max([len(channel_1_losses),len(channel_3_losses),len(channel_2_losses)])}")
# Plot the loss values for each channel
plt.figure(figsize=(8, 8))

# Plot for Channel 1
plt.plot(range(len(channel_1_losses)), channel_1_estimates, label='Red Channel', color='r')

# Plot for Channel 2
plt.plot(range(len(channel_2_losses)), channel_2_estimates, label='Green Channel', color='g')

# Plot for Channel 3
plt.plot(range(len(channel_3_losses)), channel_3_estimates, label='Blue Channel', color='b')

# plot horizontal line at y =0.6
plt.axhline(y=0.4, color='black', linestyle='--', label='True Weighting')

# Add labels and title
plt.xlabel('Step')
plt.ylabel('Weighting Estimate')
plt.title('Weighting Estimates')

# Add legend
plt.legend()

# Display the plot
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 8))

# Plot for Channel 1
plt.plot(range(len(channel_1_losses)), channel_1_losses, label='Red Channel', color='r')

# Plot for Channel 2
plt.plot(range(len(channel_2_losses)), channel_2_losses, label='Green Channel', color='g')

# Plot for Channel 3
plt.plot(range(len(channel_3_losses)), channel_3_losses, label='Blue Channel', color='b')

# Add labels and title
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Loss Values')

# Add legend
plt.legend()

# Display the plot
plt.grid(True)

# Add zoom box for last 20 iterations
ax = plt.gca()  # Get the current axis

# Create inset of the current plot for zooming
inset_ax = plt.axes([0.5, 0.3, 0.35, 0.35])  # [x, y, width, height] in normalized figure coordinates

# Plot the zoomed-in data for the last 20 iterations
zoom_start = max(0, min(len(channel_1_losses),len(channel_3_losses),len(channel_2_losses)) - 20)  # Ensure we don't go out of bounds

inset_ax.plot(range(zoom_start, len(channel_1_losses)), channel_1_losses[zoom_start:], color='r')
inset_ax.plot(range(zoom_start, len(channel_2_losses)), channel_2_losses[zoom_start:], color='g')
inset_ax.plot(range(zoom_start, len(channel_3_losses)), channel_3_losses[zoom_start:], color='b')

# Add horizontal line in the inset plot
# inset_ax.axhline(y=0.6, color='black', linestyle='--')

# Set inset axis limits
inset_ax.set_xlim(zoom_start, max(len(channel_1_losses),len(channel_3_losses),len(channel_2_losses)))  # x-axis limits for zoom
inset_ax.set_ylim(min(min(channel_1_losses[zoom_start:]), min(channel_2_losses[zoom_start:]), min(channel_3_losses[zoom_start:]), 800),
                  max(max(channel_1_losses[zoom_start:]) + 1000, max(channel_2_losses[zoom_start:]), max(channel_3_losses[zoom_start:])))  # y-axis limits for zoom

# Add grid and title for the zoomed-in plot
inset_ax.grid(True)
inset_ax.set_title('Last 20 Iterations')

plt.show()