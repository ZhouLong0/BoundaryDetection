import matplotlib.pyplot as plt
import numpy as np

def visualize_video_chunks(video_chunks_list, titles=None, figsize_scale=3):
    """
    Plots a grid of video frames where each row corresponds to one video chunk.
    
    Args:
        video_chunks_list (List[List[PIL.Image]]): A list where each element is a list of frames (a chunk).
        titles (List[str], optional): A list of titles for each row (e.g., probability scores).
        figsize_scale (int): Scalar to control the size of the output plot.
    
    Returns:
        fig: The matplotlib Figure object.
    """
    n_rows = len(video_chunks_list)
    # Find the maximum number of frames in any chunk to determine columns
    n_cols = max(len(chunk) for chunk in video_chunks_list) if n_rows > 0 else 0

    if n_rows == 0 or n_cols == 0:
        print("No data to plot.")
        return None

    # dynamic figure size based on rows/cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * figsize_scale, n_rows * figsize_scale))
    
    # If there's only one row or one column, axes might not be 2D. 
    # Ensure axes is always a 2D array for consistent indexing.
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = np.expand_dims(axes, 0)
    elif n_cols == 1:
        axes = np.expand_dims(axes, 1)

    for row_idx, chunk in enumerate(video_chunks_list):
        # Set row title if provided
        if titles and row_idx < len(titles):
            # Place title on the first axis of the row, or to the left of the row
            axes[row_idx, 0].set_ylabel(titles[row_idx], rotation=0, labelpad=40, ha='right', fontsize=12, fontweight='bold')

        for col_idx in range(n_cols):
            ax = axes[row_idx, col_idx]
            
            # Check if this chunk has a frame at this index (handle variable length chunks)
            if col_idx < len(chunk):
                ax.imshow(chunk[col_idx])
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                # Hide axis if no frame exists for this column
                ax.axis('off')

    plt.tight_layout()
    plt.show()
    return fig