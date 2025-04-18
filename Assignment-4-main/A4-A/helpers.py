import numpy as np
import matplotlib.pyplot as plt
from IPython import display


def plot_loss(
    loss_values: list[float],
    loss_values_2: list[float] = None,
    dynamic: bool = False,
    eval_interval: int = 1,
):
    """
    Plots training and validation loss over epochs, aligning validation loss based on eval_interval.

    Parameters:
    - loss_values (list[float]): List of training loss values to plot.
    - loss_values_2 (list[float], optional): List of validation loss values to plot.
    - dynamic (bool): If True, updates the plot dynamically.
    - eval_interval (int): Interval at which validation loss is evaluated.
    """
    # Generate epochs for training loss
    epochs1 = np.arange(len(loss_values))
    loss_values = np.array(loss_values)

    # Initialize plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs1, loss_values, color="b", label="Training Loss")
    plt.fill_between(epochs1, loss_values, color="b", alpha=0.1)

    # Generate epochs for validation loss, aligned with eval_interval
    if loss_values_2 is not None:
        epochs2 = np.arange(0, len(loss_values_2) * eval_interval, eval_interval)
        loss_values_2 = np.array(loss_values_2)
        plt.plot(epochs2, loss_values_2, color="r", label="Validation Loss")
        plt.fill_between(epochs2, loss_values_2, color="r", alpha=0.1)

    # Set y-axis limits based on both loss arrays
    all_losses = (
        np.concatenate([loss_values, loss_values_2])
        if loss_values_2 is not None
        else loss_values
    )
    plt.ylim(
        min(all_losses) - min(all_losses) * 0.05,
        max(all_losses) + max(all_losses) * 0.05,
    )

    # Formatting
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.show()
    plt.pause(0.5)  # for live updating

    # For dynamic display in Jupyter Notebook
    display.display(plt.gcf())
    if dynamic:
        display.clear_output(wait=True)
