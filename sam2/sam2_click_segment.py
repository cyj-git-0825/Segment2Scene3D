%matplotlib widget

import numpy as np
import torch
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Helper functions
def show_mask(mask, ax, random_color=False, borders=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

# Load model
checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
sam2_model = build_sam2(model_cfg, checkpoint, device="cuda")
predictor = SAM2ImagePredictor(sam2_model)

# Load image
image = Image.open("./notebooks/images/common_room.jpg")
image = np.array(image.convert("RGB"))
predictor.set_image(image)
print(f"Image size: {image.shape[1]} x {image.shape[0]}")
print("Ready!")

------------------------------------------------------------------------------------------------------------------
# click anywhere you want to segment
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display

coords = []
all_masks = []

fig, ax = plt.subplots(figsize=(12, 8))
ax.imshow(image)
ax.set_title("Left-click to add objects")
ax.axis('off')

def on_click(event):
    if event.inaxes is None:
        return
    x, y = int(event.xdata), int(event.ydata)
    coords.append([x, y])
    
    input_point = np.array([[x, y]])
    input_label = np.array([1])
    masks, scores, logits = predictor.predict(
        point_coords=input_point, point_labels=input_label, multimask_output=True)
    
    best = np.argsort(scores)[::-1][0]
    all_masks.append(masks[best])
    
    ax.clear()
    ax.imshow(image)
    for mask in all_masks:
        show_mask(mask, ax, random_color=True)
    for cx, cy in coords:
        ax.plot(cx, cy, 'g*', markersize=20)
    ax.set_title(f"Objects: {len(coords)} | Score: {scores[best]:.3f}")
    ax.axis('off')
    fig.canvas.draw()

def clear_all(b):
    coords.clear()
    all_masks.clear()
    ax.clear()
    ax.imshow(image)
    ax.set_title("Cleared! Click to start again")
    ax.axis('off')
    fig.canvas.draw()

def undo_last(b):
    if coords:
        coords.pop()
        all_masks.pop()
        ax.clear()
        ax.imshow(image)
        for mask in all_masks:
            show_mask(mask, ax, random_color=True)
        for cx, cy in coords:
            ax.plot(cx, cy, 'g*', markersize=20)
        ax.set_title(f"Objects: {len(coords)} | Undid last point")
        ax.axis('off')
        fig.canvas.draw()

clear_btn = widgets.Button(description='Clear All', button_style='danger')
undo_btn = widgets.Button(description='Undo Last', button_style='warning')
clear_btn.on_click(clear_all)
undo_btn.on_click(undo_last)

fig.canvas.mpl_connect('button_press_event', on_click)
display(widgets.HBox([undo_btn, clear_btn]))
plt.show()
-------------------------------------------------------------------------------------------------------------
#generate black/white mask
def save_mask(b):
    if not all_masks:
        print("No mask to save! Click on an object first.")
        return
    
    # Combine all masks into one
    combined_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for mask in all_masks:
        combined_mask[mask > 0] = 255
    
    # Save as black and white image
    mask_img = Image.fromarray(combined_mask)
    save_path = "./notebooks/images/common_room_mask.png"
    mask_img.save(save_path)
    print(f"Mask saved to {save_path}")
    
    # Display the mask
    fig2, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[0].axis('off')
    axes[1].imshow(combined_mask, cmap='gray')
    axes[1].set_title("Black & White Mask")
    axes[1].axis('off')
    plt.tight_layout()
    plt.show()

save_btn = widgets.Button(description='Save Mask', button_style='success')
save_btn.on_click(save_mask)
display(save_btn)
