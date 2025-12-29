from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors



def make_thumbnail_cell(pil_img, cell_px, pad, bg_color=255):
    """
    Create a thumbnail cell by resizing the image while preserving aspect ratio,
    then centering it in a cell_px x cell_px square with padding.
    
    Args:
        pil_img: PIL Image (RGB)
        cell_px: size of the square cell in pixels
        pad: padding around the thumbnail
        bg_color: background color (default 255 for white)
    
    Returns:
        PIL Image of size (cell_px, cell_px)
    """
    # Create a square background
    cell_img = Image.new('RGB', (cell_px, cell_px), color=(bg_color, bg_color, bg_color))
    
    # Calculate maximum size for thumbnail (preserving aspect ratio)
    max_size = cell_px - 2 * pad
    img_width, img_height = pil_img.size
    
    # Calculate scaling factor
    scale = min(max_size / img_width, max_size / img_height)
    new_width = int(img_width * scale)
    new_height = int(img_height * scale)
    
    # Resize image
    resized = pil_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Calculate position to center the resized image
    x_offset = (cell_px - new_width) // 2
    y_offset = (cell_px - new_height) // 2
    
    # Paste resized image onto cell
    cell_img.paste(resized, (x_offset, y_offset))
    
    return cell_img

def build_similarity_thumbnail_canvas(
    row_images,
    col_images,
    sim,
    cell_px,
    pad,
    cmap_name,
    vmin,
    vmax,
    gridline_px,
):
    """
    Build a composite RGB canvas for (N_rows+1)x(N_cols+1) grid with thumbnails on
    top row (col_images) and left column (row_images), and colored similarity cells.

    Args:
        row_images: list of PIL Images for rows (set A)
        col_images: list of PIL Images for cols (set B)
        sim: numpy array of shape (N_rows, N_cols) with similarity values
        cell_px: pixel size of each cell
        pad: padding for thumbnails
        cmap_name: matplotlib colormap name
        vmin, vmax: normalization range for similarity values
        gridline_px: thickness of gridlines (0 to disable)

    Returns:
        numpy array of shape (H, W, 3) with dtype uint8
    """
    n_rows = len(row_images)
    n_cols = len(col_images)
    assert sim.shape == (
        n_rows,
        n_cols,
    ), f"Expected similarity matrix shape ({n_rows}, {n_cols}), got {sim.shape}"

    cmap = plt.get_cmap(cmap_name)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    total_rows = n_rows + 1
    total_cols = n_cols + 1
    canvas_width = total_cols * cell_px + (total_cols - 1) * gridline_px
    canvas_height = total_rows * cell_px + (total_rows - 1) * gridline_px

    canvas = np.full((canvas_height, canvas_width, 3), 255, dtype=np.uint8)

    row_thumb_cells = [make_thumbnail_cell(img, cell_px, pad) for img in row_images]
    col_thumb_cells = [make_thumbnail_cell(img, cell_px, pad) for img in col_images]

    def get_cell_pos(row, col):
        y_start = row * (cell_px + gridline_px)
        x_start = col * (cell_px + gridline_px)
        return y_start, x_start

    # Top row thumbnails (row 0, columns 1..n_cols)
    for j in range(n_cols):
        y_start, x_start = get_cell_pos(0, j + 1)
        canvas[y_start : y_start + cell_px, x_start : x_start + cell_px] = np.array(
            col_thumb_cells[j]
        )

    # Left column thumbnails (rows 1..n_rows, column 0)
    for i in range(n_rows):
        y_start, x_start = get_cell_pos(i + 1, 0)
        canvas[y_start : y_start + cell_px, x_start : x_start + cell_px] = np.array(
            row_thumb_cells[i]
        )

    # Similarity cells (rows 1..n_rows, cols 1..n_cols)
    for i in range(n_rows):
        for j in range(n_cols):
            y_start, x_start = get_cell_pos(i + 1, j + 1)
            sim_value = sim[i, j]
            rgba = cmap(norm(sim_value))
            rgb = (np.array(rgba[:3]) * 255).astype(np.uint8)
            canvas[y_start : y_start + cell_px, x_start : x_start + cell_px] = rgb

    # Text annotations
    canvas_pil = Image.fromarray(canvas)
    draw = ImageDraw.Draw(canvas_pil)

    try:
        font_size = max(12, cell_px // 8)
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size
        )
    except Exception:
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

    for i in range(n_rows):
        for j in range(n_cols):
            y_start, x_start = get_cell_pos(i + 1, j + 1)
            sim_value = sim[i, j]
            text = f"{sim_value:.2f}"

            cell_color = canvas[y_start : y_start + cell_px, x_start : x_start + cell_px]
            avg_brightness = float(np.mean(cell_color))
            text_color = (255, 255, 255) if avg_brightness < 128 else (0, 0, 0)

            if font:
                bbox = draw.textbbox((0, 0), text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
            else:
                text_width = len(text) * 6
                text_height = 10

            text_x = x_start + (cell_px - text_width) // 2
            text_y = y_start + (cell_px - text_height) // 2
            draw.text((text_x, text_y), text, fill=text_color, font=font)

    canvas = np.array(canvas_pil)

    if gridline_px > 0:
        gridline_color = [200, 200, 200]
        for col in range(1, total_cols):
            x = col * (cell_px + gridline_px) - gridline_px
            canvas[:, x : x + gridline_px] = gridline_color
        for row in range(1, total_rows):
            y = row * (cell_px + gridline_px) - gridline_px
            canvas[y : y + gridline_px, :] = gridline_color

    return canvas


def plot_canvas_with_colorbar(canvas, cmap_name, vmin, vmax, savepath=None, title=None, block=False):
    """
    Display the composite canvas with matplotlib and add a colorbar.

    Args:
        canvas: numpy array of shape (H, W, 3) with dtype uint8
        cmap_name: name of matplotlib colormap
        vmin, vmax: normalization range for similarity values
        savepath: path where the heatmap will be saved (default: None, no saving)
        title: optional title for the plot
        block: whether to block execution when showing the plot (default: False)
    """
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(canvas)
    ax.axis('off')

    # Add title if provided
    if title:
        ax.set_title(title, fontsize=16, pad=20)

    # Create a ScalarMappable for the colorbar
    cmap = plt.get_cmap(cmap_name)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    
    # Add colorbar
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Cosine Similarity', rotation=270, labelpad=20)
    
    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath, dpi=150, bbox_inches='tight')
        print(f"Similarity heatmap saved to: {savepath}")
        plt.close()
    else:
        # Try to display the plot, close if display is not available
        try:
            plt.show(block=block)
        except Exception:
            plt.close()
            
def plot_matrix(
    images1: list,
    images2: list,
    sim: np.ndarray,
    *,
    cell_px: int = 128,
    thumb_pad: int = 4,
    gridline_px: int = 1,
    cmap_name: str = "YlGnBu",
    vmin: float = -1.0,
    vmax: float = 1.0,
    savepath: str | None = None,
    title: str | None = None,
    block: bool = False,
) -> np.ndarray:
    """
    Plot a similarity heatmap grid between two image sets with thumbnails.

    - Rows correspond to images1
    - Columns correspond to images2

    Args:
        images1: List of PIL Images (rows)
        images2: List of PIL Images (cols)
        sim: Similarity matrix (numpy array) of shape (len(images1), len(images2))
        cell_px, thumb_pad, gridline_px, cmap_name, vmin, vmax: styling params
        savepath: If provided, saves the figure; otherwise tries to show
        title: Optional title for the plot
        block: whether to block execution when showing the plot (default: False)

    Returns:
        canvas: numpy array (H, W, 3) uint8
    """
    canvas = build_similarity_thumbnail_canvas(
        images1, images2, sim, cell_px, thumb_pad, cmap_name, vmin, vmax, gridline_px
    )
    plot_canvas_with_colorbar(canvas, cmap_name, vmin, vmax, savepath, title, block)
    return canvas
