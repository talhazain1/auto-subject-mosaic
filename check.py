import os
import random
import string
import numpy as np
import cv2

from flask import Flask, render_template, request, url_for
from PIL import Image
from shapely.geometry import Point, Polygon, MultiPoint
from shapely.ops import voronoi_diagram, unary_union
from skimage import io

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static'
app.config['SECRET_KEY'] = 'secret-key'  # Replace in production

def auto_saliency_mask(image_path):
    """
    Generate a binary mask (255=foreground, 0=background) using OpenCV's Saliency.
    This is a naive approach; quality can vary by image.
    """
    src = cv2.imread(image_path)
    if src is None:
        return None

    # Convert to BGR if needed
    if len(src.shape) == 2:
        src = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)

    # Saliency using the Fine Grained Saliency method
    # You may try other methods like 'StaticSaliencySpectralResidual'
    saliency = cv2.saliency.StaticSaliencyFineGrained_create()
    success, saliency_map = saliency.computeSaliency(src)
    if not success:
        return None

    # saliency_map is a float32 in [0,1], we threshold it
    # to create a binary mask
    thresh = 0.5  # tweak this as needed
    mask = (saliency_map * 255).astype(np.uint8)
    mask_binary = cv2.threshold(mask, int(thresh*255), 255, cv2.THRESH_BINARY)[1]

    # Optional morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask_clean = cv2.morphologyEx(mask_binary, cv2.MORPH_CLOSE, kernel)

    return mask_clean  # 255=foreground, 0=background

def poisson_disk_sampling(width, height, radius=15, k=30, mask_array=None):
    """
    Poisson-disk sampling in [0..width, 0..height], restricted to mask_array==255 if provided.
    """
    cell_size = radius / np.sqrt(2)
    grid_width = int(np.ceil(width / cell_size))
    grid_height = int(np.ceil(height / cell_size))

    grid = [-1] * (grid_width * grid_height)
    samples = []

    def in_mask(x, y):
        if mask_array is None:
            return True
        ix, iy = int(x), int(y)
        if 0 <= ix < width and 0 <= iy < height:
            return (mask_array[iy, ix] == 255)
        return False

    # random valid start
    tries = 0
    while True:
        x0 = random.uniform(0, width)
        y0 = random.uniform(0, height)
        if in_mask(x0, y0):
            samples.append((x0, y0))
            gx0 = int(x0 // cell_size)
            gy0 = int(y0 // cell_size)
            grid[gy0 * grid_width + gx0] = 0
            break
        tries += 1
        if tries > 10000:
            break

    active = [0]
    while active:
        idx = random.choice(active)
        x, y = samples[idx]
        found = False
        for _ in range(k):
            r = radius * (1 + random.random())
            theta = 2*np.pi*random.random()
            x_new = x + r*np.cos(theta)
            y_new = y + r*np.sin(theta)
            if 0 <= x_new < width and 0 <= y_new < height:
                if in_mask(x_new, y_new):
                    gx = int(x_new // cell_size)
                    gy = int(y_new // cell_size)
                    too_close = False
                    for ny in range(max(0, gy-2), min(grid_height, gy+3)):
                        for nx in range(max(0, gx-2), min(grid_width, gx+3)):
                            neighbor_index = grid[ny*grid_width + nx]
                            if neighbor_index != -1:
                                x_n, y_n = samples[neighbor_index]
                                if (x_n - x_new)**2 + (y_n - y_new)**2 < radius**2:
                                    too_close = True
                                    break
                        if too_close:
                            break
                    if not too_close:
                        samples.append((x_new, y_new))
                        sample_idx = len(samples) - 1
                        active.append(sample_idx)
                        grid[gy*grid_width + gx] = sample_idx
                        found = True
                        break
        if not found:
            active.remove(idx)

    return samples

def dominant_color_in_polygon(poly, img_np):
    """
    Return the dominant (most frequent) color (R,G,B) inside the polygon.
    """
    minx, miny, maxx, maxy = map(int, map(np.floor, poly.bounds))
    h, w = img_np.shape[:2]
    minx = max(0, minx)
    miny = max(0, miny)
    maxx = min(w-1, maxx)
    maxy = min(h-1, maxy)

    color_counts = {}
    for yy in range(miny, maxy+1):
        for xx in range(minx, maxx+1):
            if poly.contains(Point(xx, yy)):
                col = tuple(img_np[yy, xx])  # (R,G,B)
                color_counts[col] = color_counts.get(col, 0) + 1
    if not color_counts:
        return (255,255,255)
    return max(color_counts, key=color_counts.get)

def draw_polygon_fill(poly, image_bgr, color_bgr):
    color_bgr_int = tuple(int(c) for c in color_bgr)
    exterior = np.array(list(poly.exterior.coords), dtype=np.int32)
    cv2.fillPoly(image_bgr, [exterior], color_bgr_int)
    for interior in poly.interiors:
        interior_coords = np.array(list(interior.coords), dtype=np.int32)
        cv2.fillPoly(image_bgr, [interior_coords], (0,0,0))

def draw_polygon_outline(poly, image_bgr, color=(0,0,0), thickness=2):
    color_int = tuple(int(c) for c in color)
    exterior = np.array(list(poly.exterior.coords), dtype=np.int32)
    cv2.polylines(image_bgr, [exterior], isClosed=True, color=color_int, thickness=thickness)
    for interior in poly.interiors:
        coords = np.array(list(interior.coords), dtype=np.int32)
        cv2.polylines(image_bgr, [coords], isClosed=True, color=color_int, thickness=thickness)

def sharpen_image(image_bgr):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]], dtype=np.float32)
    return cv2.filter2D(image_bgr, -1, kernel)

def generate_auto_subject_mosaic(
    input_image_path,
    output_mosaic_path="static/mosaic_result.png",
    output_outline_path="static/mosaic_outlines.svg",
    radius_subject=10,
    radius_background=30,
    outline_thickness=2,
    smoothing=0,
    sharpen=True
):
    """
    1) Create a saliency-based mask => subject region (255) vs. background (0).
    2) Poisson sample subject with smaller radius, background with larger radius.
    3) Voronoi diagram => fill polygons with dominant color.
    4) (Optional) sharpen result.
    """
    # 1) Auto-generate mask from the single image
    mask_array = auto_saliency_mask(input_image_path)
    if mask_array is None:
        print("Saliency mask creation failed; falling back to entire image as 'subject'.")
        # fallback: entire image is subject
        # => single radius
        radius_subject = radius_background
        mask_array = np.ones((1,1), dtype=np.uint8)*255  # trivial

    # load main image
    img_pil = Image.open(input_image_path).convert("RGB")
    width, height = img_pil.size
    img_np = np.array(img_pil)

    # subject seeds
    subject_seeds = poisson_disk_sampling(width, height, radius=radius_subject, mask_array=mask_array)
    # background seeds
    bg_mask = 255 - mask_array  # invert
    background_seeds = poisson_disk_sampling(width, height, radius=radius_background, mask_array=bg_mask)

    # combine
    all_seeds = subject_seeds + background_seeds

    # build Voronoi
    from shapely.ops import voronoi_diagram
    mp = MultiPoint([Point(x, y) for x, y in all_seeds])
    vor = voronoi_diagram(mp)

    # bounding polygon
    bounding_poly = Polygon([(0,0), (width,0), (width,height), (0,height)])
    mosaic_bgr = np.zeros_like(img_np, dtype=np.uint8)

    with open(output_outline_path, "w") as svg_file:
        svg_file.write(f'<svg width="{width}" height="{height}" '
                       f'viewBox="0 0 {width} {height}" '
                       f'xmlns="http://www.w3.org/2000/svg">\n')
        svg_file.write(f'<g fill="none" stroke="black" stroke-width="{outline_thickness}">\n')

        for cell in vor.geoms:
            clipped = cell.intersection(bounding_poly)
            if clipped.is_empty:
                continue
            if clipped.geom_type == 'MultiPolygon':
                sub_polys = clipped.geoms
            else:
                sub_polys = [clipped]

            for spoly in sub_polys:
                if spoly.is_empty:
                    continue
                if smoothing > 0:
                    spoly_simpl = spoly.buffer(smoothing).buffer(-smoothing)
                else:
                    spoly_simpl = spoly
                if spoly_simpl.is_empty:
                    continue

                dom_col = dominant_color_in_polygon(spoly_simpl, img_np)
                color_bgr = dom_col[::-1]
                draw_polygon_fill(spoly_simpl, mosaic_bgr, color_bgr)
                draw_polygon_outline(spoly_simpl, mosaic_bgr, (0,0,0), thickness=outline_thickness)

                # SVG path
                exterior_coords = list(spoly_simpl.exterior.coords)
                path_data = "M " + " L ".join(f"{x:.2f},{y:.2f}" for x, y in exterior_coords) + " Z"
                svg_file.write(f'  <path d="{path_data}" />\n')

        svg_file.write('</g>\n</svg>\n')

    # optional sharpen
    if sharpen:
        mosaic_bgr = sharpen_image(mosaic_bgr)

    mosaic_rgb = mosaic_bgr[..., ::-1]
    out_pil = Image.fromarray(mosaic_rgb)
    out_pil.save(output_mosaic_path)
    print("Auto-subject mosaic saved to:", output_mosaic_path)
    print("SVG outlines saved to:", output_outline_path)

@app.route("/", methods=["GET", "POST"])
def index():
    mosaic_path = None
    outline_path = None

    if request.method == "POST":
        file = request.files.get("image_file")
        if file:
            rand_str = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
            ext = os.path.splitext(file.filename)[1].lower()
            safe_filename = f"upload_{rand_str}{ext}"
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], safe_filename)
            file.save(upload_path)

            try:
                radius_subject = int(request.form.get("radius_subject", 10))
            except:
                radius_subject = 10
            try:
                radius_background = int(request.form.get("radius_background", 30))
            except:
                radius_background = 30
            outline_thickness = int(request.form.get("outline_thickness", 2))
            smoothing = float(request.form.get("smoothing", 0))
            sharpen = (request.form.get("sharpen", "true") == "true")

            mosaic_output_name = f"mosaic_{rand_str}.png"
            outline_output_name = f"outlines_{rand_str}.svg"
            mosaic_output_path = os.path.join(app.config['UPLOAD_FOLDER'], mosaic_output_name)
            outline_output_path = os.path.join(app.config['UPLOAD_FOLDER'], outline_output_name)

            generate_auto_subject_mosaic(
                input_image_path=upload_path,
                output_mosaic_path=mosaic_output_path,
                output_outline_path=outline_output_path,
                radius_subject=radius_subject,
                radius_background=radius_background,
                outline_thickness=outline_thickness,
                smoothing=smoothing,
                sharpen=sharpen
            )

            mosaic_path = url_for('static', filename=mosaic_output_name)
            outline_path = url_for('static', filename=outline_output_name)

    return render_template("home.html", mosaic_path=mosaic_path, outline_path=outline_path)

if __name__ == "__main__":
    if not os.path.isdir(app.config['UPLOAD_FOLDER']):
        os.mkdir(app.config['UPLOAD_FOLDER'])
    # pip install flask pillow shapely>=2.0 scikit-image opencv-python
    # Also need: pip install opencv-contrib-python (for saliency)
    app.run(port=5003, debug=True)
