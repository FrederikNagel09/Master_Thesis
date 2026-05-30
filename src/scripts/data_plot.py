import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import torchvision.datasets as datasets


def create_seamless_grid(output_filename="src/results/mnist_cifar10_perfect_grid.png"):
    print("Loading datasets...")
    mnist_dataset = datasets.MNIST(root="./data", train=True, download=True)
    cifar_dataset = datasets.CIFAR10(root="./data", train=True, download=True)

    # 1. Control the exact gap between the two main grids
    # 0.25 means the gap width is 25% of a single image's width
    gap_width = 0.25
    width_ratios = [1, 1, 1, 1, 1, gap_width, 1, 1, 1, 1, 1]

    # 2. Math to perfectly match the canvas to the grid aspect ratio
    # Total width units = 10 images + the gap. Total height units = 5 images.
    scaling_factor = 1.1
    fig_width = (10 + gap_width) * scaling_factor
    fig_height = (5 * scaling_factor) + 0.4  # Extra room for the bottom labels

    fig = plt.figure(figsize=(fig_width, fig_height), facecolor="white")

    # 3. Create a unified 5x11 grid with absolute ZERO internal padding
    gs = gridspec.GridSpec(5, 11, width_ratios=width_ratios, wspace=0, hspace=0)

    mnist_idx = 0
    cifar_idx = 0

    for row in range(5):
        for col in range(11):
            # Skip drawing anything in the middle divider column
            if col == 5:
                continue

            ax = fig.add_subplot(gs[row, col])

            # --- Left Side: MNIST ---
            if col < 5:
                img, _ = mnist_dataset[mnist_idx]
                ax.imshow(img, cmap="gray")
                mnist_idx += 1

                # Place label under the middle column (col 2) of the bottom row (row 4)
                if row == 4 and col == 2:
                    ax.set_xlabel("a) MNIST", fontsize=12, fontweight="bold", labelpad=8)

            # --- Right Side: CIFAR-10 ---
            else:
                img, _ = cifar_dataset[cifar_idx]
                ax.imshow(img)
                cifar_idx += 1

                # Place label under the middle column (col 8) of the bottom row (row 4)
                if row == 4 and col == 8:
                    ax.set_xlabel("b) CIFAR-10", fontsize=12, fontweight="bold", labelpad=8)

            # Clean up axes manually so we don't accidentally hide the bottom text labels
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

    # Save with tight bounding bounds
    plt.savefig(output_filename, bbox_inches="tight", dpi=300)
    print(f"Success! Seamless plot saved as '{output_filename}'")


if __name__ == "__main__":
    create_seamless_grid()
