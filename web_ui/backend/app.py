from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import open3d as o3d
import numpy as np
import cv2
import os
import uuid

app = Flask(__name__)
CORS(app)

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), '..', 'uploads')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'output')
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def process_ply(ply_path, yaw_deg, output_id):
    """Rotate point cloud, render 2D image and mask."""
    pcd = o3d.io.read_point_cloud(ply_path)

    # Rotate around Z axis only
    R = pcd.get_rotation_matrix_from_xyz((0, 0, np.radians(yaw_deg)))
    pcd.rotate(R, center=pcd.get_center())

    # Ground plane alignment
    pts = np.asarray(pcd.points)
    pcd.translate((0, 0, -pts[:, 2].min()))

    # Center X/Y
    pts = np.asarray(pcd.points)
    cx = (pts[:, 0].min() + pts[:, 0].max()) / 2
    cy = (pts[:, 1].min() + pts[:, 1].max()) / 2
    pcd.translate((-cx, -cy, 0))

    # Project to 2D (X-Z front view)
    pts = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    x, z = pts[:, 0], pts[:, 2]

    img_w, img_h = 640, 480
    margin = 0.05
    x_min, x_max = x.min(), x.max()
    z_min, z_max = z.min(), z.max()
    x_range, z_range = x_max - x_min, z_max - z_min
    scale = min(img_w * (1 - 2 * margin) / x_range, img_h * (1 - 2 * margin) / z_range)

    px = ((x - x_min) * scale + (img_w - x_range * scale) / 2).astype(int)
    py = ((z_max - z) * scale + (img_h - z_range * scale) / 2).astype(int)

    valid = (px >= 0) & (px < img_w) & (py >= 0) & (py < img_h)
    px, py = px[valid], py[valid]
    colors_v = colors[valid]

    # Rendered image with inpainting
    rendered = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    rendered[py, px] = (colors_v * 255).astype(np.uint8)

    contours, _ = cv2.findContours(
        (rendered.max(axis=2) > 0).astype(np.uint8) * 255,
        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    fill = np.zeros((img_h, img_w), dtype=np.uint8)
    cv2.drawContours(fill, contours, -1, 255, thickness=cv2.FILLED)
    gaps = (fill == 255) & (rendered.max(axis=2) == 0)
    rendered = cv2.inpaint(rendered, gaps.astype(np.uint8) * 255, 3, cv2.INPAINT_TELEA)

    # Mask with contour fill
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    mask[py, px] = 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_filled = np.zeros_like(mask)
    cv2.drawContours(mask_filled, contours, -1, 255, thickness=cv2.FILLED)

    # Save
    out_dir = os.path.join(OUTPUT_DIR, output_id)
    os.makedirs(out_dir, exist_ok=True)

    rendered_path = os.path.join(out_dir, 'rendered.png')
    mask_path = os.path.join(out_dir, 'mask.png')
    cv2.imwrite(rendered_path, cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR))
    cv2.imwrite(mask_path, mask_filled)

    return rendered_path, mask_path


def composite(scene_path, rendered_path, mask_path, x_pos, y_pos, scale_pct, output_id):
    """Place rendered object onto scene at given position."""
    scene = cv2.imread(scene_path)
    rendered = cv2.imread(rendered_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # Scale object
    target_h = int(scene.shape[0] * scale_pct / 100)
    s = target_h / rendered.shape[0]
    target_w = int(rendered.shape[1] * s)
    rendered = cv2.resize(rendered, (target_w, target_h))
    mask = cv2.resize(mask, (target_w, target_h))

    # Position (x_pos, y_pos are percentages)
    x_off = int(scene.shape[1] * x_pos / 100)
    y_off = int(scene.shape[0] * y_pos / 100) - target_h  # bottom-aligned

    # Clamp
    x_off = max(0, min(x_off, scene.shape[1] - target_w))
    y_off = max(0, min(y_off, scene.shape[0] - target_h))

    x_end = min(x_off + target_w, scene.shape[1])
    y_end = min(y_off + target_h, scene.shape[0])
    w, h = x_end - x_off, y_end - y_off

    mask_region = mask[:h, :w]
    rendered_region = rendered[:h, :w]
    mask_3ch = cv2.merge([mask_region, mask_region, mask_region]) / 255.0

    result = scene.copy()
    roi = result[y_off:y_end, x_off:x_end]
    result[y_off:y_end, x_off:x_end] = (rendered_region * mask_3ch + roi * (1 - mask_3ch)).astype(np.uint8)

    out_dir = os.path.join(OUTPUT_DIR, output_id)
    comp_path = os.path.join(out_dir, 'composite.png')
    comp_mask_path = os.path.join(out_dir, 'composite_mask.png')

    cv2.imwrite(comp_path, result)

    comp_mask = np.zeros((scene.shape[0], scene.shape[1]), dtype=np.uint8)
    comp_mask[y_off:y_end, x_off:x_end] = mask_region
    cv2.imwrite(comp_mask_path, comp_mask)

    return comp_path, comp_mask_path


@app.route('/api/upload-ply', methods=['POST'])
def upload_ply():
    f = request.files.get('file')
    if not f:
        return jsonify(error='No file'), 400
    fid = str(uuid.uuid4())[:8]
    path = os.path.join(UPLOAD_DIR, f'{fid}.ply')
    f.save(path)
    return jsonify(id=fid, filename=f.filename)


@app.route('/api/upload-scene', methods=['POST'])
def upload_scene():
    f = request.files.get('file')
    if not f:
        return jsonify(error='No file'), 400
    fid = str(uuid.uuid4())[:8]
    ext = os.path.splitext(f.filename)[1]
    path = os.path.join(UPLOAD_DIR, f'{fid}{ext}')
    f.save(path)
    return jsonify(id=fid, filename=f.filename, ext=ext)


@app.route('/api/render', methods=['POST'])
def render():
    data = request.json
    ply_id = data.get('ply_id')
    yaw = data.get('yaw', 0)

    ply_path = os.path.join(UPLOAD_DIR, f'{ply_id}.ply')
    if not os.path.exists(ply_path):
        return jsonify(error='PLY not found'), 404

    output_id = str(uuid.uuid4())[:8]
    rendered_path, mask_path = process_ply(ply_path, yaw, output_id)

    return jsonify(
        output_id=output_id,
        rendered=f'/api/file/{output_id}/rendered.png',
        mask=f'/api/file/{output_id}/mask.png'
    )


@app.route('/api/composite', methods=['POST'])
def composite_route():
    data = request.json
    output_id = data.get('output_id')
    scene_id = data.get('scene_id')
    scene_ext = data.get('scene_ext', '.jpg')
    x_pos = data.get('x_pos', 60)
    y_pos = data.get('y_pos', 80)
    scale_pct = data.get('scale_pct', 30)

    scene_path = os.path.join(UPLOAD_DIR, f'{scene_id}{scene_ext}')
    rendered_path = os.path.join(OUTPUT_DIR, output_id, 'rendered.png')
    mask_path = os.path.join(OUTPUT_DIR, output_id, 'mask.png')

    comp_path, comp_mask_path = composite(
        scene_path, rendered_path, mask_path, x_pos, y_pos, scale_pct, output_id
    )

    return jsonify(
        composite=f'/api/file/{output_id}/composite.png',
        composite_mask=f'/api/file/{output_id}/composite_mask.png'
    )


@app.route('/api/file/<output_id>/<filename>')
def serve_file(output_id, filename):
    path = os.path.join(OUTPUT_DIR, output_id, filename)
    if not os.path.exists(path):
        return jsonify(error='Not found'), 404
    return send_file(path)


if __name__ == '__main__':
    app.run(debug=True, port=5000)