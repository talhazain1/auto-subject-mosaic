import os
import numpy as np
import cv2
from PIL import Image
from skimage import io, segmentation
from flask import Flask, render_template, request, url_for
import random
import string

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static'
app.config['SECRET_KEY'] = 'secret-key'  # Replace in production

def generate_superpixel_mosaic(
    input_image_path,
    output_mosaic_path="static/mosaic_result.png",
    output_outline_path="static/mosaic_outlines.svg",
    n_segments=1500,
    compactness=20.0,
    sigma=2.0,
    outline_thickness=2
):
    """
    Generate a mosaic using SLIC superpixels, then export both a PNG image and an SVG outline.
    We add steps to reduce grid-like patterns and smooth jagged edges:
      - Lower compactness and add sigma in SLIC
      - Morphological smoothing on each superpixel mask
      - Use a chain approximation method for smoother contours
    """

    # 1) Read the input image
    img = io.imread(input_image_path)  # shape: (H, W, 3) or (H, W, 4)
    if img.shape[-1] == 4:
        img = img[..., :3]  # drop alpha if present

    height, width = img.shape[:2]

    # 2) SLIC superpixels with lower compactness, added sigma, and disabled connectivity
    #    This helps avoid a "grid" effect.
    segments = segmentation.slic(
        img,
        n_segments=n_segments,
        compactness=compactness,
        sigma=sigma,
        start_label=1,
        enforce_connectivity=False  # can reduce blocky shapes
    )
    # 'segments' is a 2D array with superpixel IDs.

    # 3) Compute average color for each superpixel
    segment_ids = np.unique(segments)
    avg_colors = {}
    for seg_id in segment_ids:
        mask = (segments == seg_id)
        seg_pixels = img[mask]
        mean_col = seg_pixels.mean(axis=0)
        avg_colors[seg_id] = tuple(mean_col.astype(np.uint8))

    # 4) Prepare a blank BGR image for the mosaic
    mosaic_bgr = np.zeros_like(img, dtype=np.uint8)

    # 5) Open the SVG file for outlines
    with open(output_outline_path, "w") as svg_file:
        svg_file.write(f'<svg width="{width}" height="{height}" '
                       f'viewBox="0 0 {width} {height}" '
                       f'xmlns="http://www.w3.org/2000/svg">\n')
        svg_file.write(f'<g fill="none" stroke="black" stroke-width="{outline_thickness}">\n')

        # 6) For each superpixel, find & fill contours
        for seg_id in segment_ids:
            # Create a binary mask for this segment
            mask = np.where(segments == seg_id, 255, 0).astype(np.uint8)

            # (a) Morphological smoothing: close small gaps / holes
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
            mask_smooth = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            # (b) Find contours with a chain approximation that tends to be smoother
            #     Alternatively, we can do an additional approxPolyDP step below.
            contours, _ = cv2.findContours(mask_smooth, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)

            # Convert average color from RGB to BGR
            color_bgr = avg_colors[seg_id][::-1]
            color_bgr_int = tuple(map(int, color_bgr))

            # Fill each contour
            for c in contours:
                # Optional further smoothing:
                # approx = cv2.approxPolyDP(c, 0.005 * cv2.arcLength(c, True), True)
                # cv2.fillPoly(mosaic_bgr, [approx], color_bgr_int)
                # cv2.polylines(mosaic_bgr, [approx], isClosed=True, color=(0,0,0), thickness=outline_thickness)

                # or fill directly:
                cv2.fillPoly(mosaic_bgr, [c], color_bgr_int)

                # Draw the dark outline
                cv2.polylines(mosaic_bgr, [c], isClosed=True, color=(0, 0, 0), thickness=outline_thickness)

                # Write the contour to SVG
                points = c.squeeze()
                if len(points.shape) != 2:
                    continue
                path_data = "M " + " L ".join(f"{p[0]},{p[1]}" for p in points) + " Z"
                svg_file.write(f'  <path d="{path_data}" />\n')

        svg_file.write('</g>\n</svg>\n')

    # 7) Save the mosaic image (with outlines)
    mosaic_rgb = mosaic_bgr[..., ::-1]
    mosaic_pil = Image.fromarray(mosaic_rgb)
    mosaic_pil.save(output_mosaic_path)
    print("Superpixel mosaic saved to:", output_mosaic_path)
    print("SVG tile outlines saved to:", output_outline_path)


@app.route("/", methods=["GET", "POST"])
def index():
    mosaic_path = None
    outline_path = None

    if request.method == "POST":
        file = request.files.get("image_file")
        if file:
            random_str = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
            ext = os.path.splitext(file.filename)[1].lower()
            safe_filename = f"upload_{random_str}{ext}"
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], safe_filename)
            file.save(upload_path)

            try:
                outline_thickness = int(request.form.get("outline_thickness", 2))
            except ValueError:
                outline_thickness = 2

            mosaic_output_name = f"mosaic_{random_str}.png"
            outline_output_name = f"outlines_{random_str}.svg"
            mosaic_output_path = os.path.join(app.config['UPLOAD_FOLDER'], mosaic_output_name)
            outline_output_path = os.path.join(app.config['UPLOAD_FOLDER'], outline_output_name)

            # Adjust these parameters to taste
            generate_superpixel_mosaic(
                input_image_path=upload_path,
                output_mosaic_path=mosaic_output_path,
                output_outline_path=outline_output_path,
                n_segments=750,       # Fewer => bigger tiles, More => finer detail
                compactness=79.5,      # Lower => more organic, higher => more grid-like
                sigma=0.0,             # Slight blur helps smooth edges
                outline_thickness=outline_thickness
            )

            mosaic_path = url_for('static', filename=mosaic_output_name)
            outline_path = url_for('static', filename=outline_output_name)

    return render_template("index.html", mosaic_path=mosaic_path, outline_path=outline_path)


if __name__ == "__main__":
    if not os.path.isdir(app.config['UPLOAD_FOLDER']):
        os.mkdir(app.config['UPLOAD_FOLDER'])
    app.run(port=5003, debug=True)
