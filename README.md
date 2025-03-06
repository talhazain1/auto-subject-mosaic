# Auto Subject Mosaic

A Flask web app that automatically detects the main subject in an image and generates a hybrid Poisson-disk Voronoi mosaic—rendering fine detail on the subject and larger, abstract tiles for the background.

## Features

- **Automatic Subject Detection**  
  Uses OpenCV saliency detection to generate a subject mask from a single image.

- **Hybrid Poisson-Disk Tiling**  
  Applies finer tiling (smaller Poisson radius) on the detected subject and coarser tiling (larger radius) on the background.

- **Dominant Color Filling**  
  Each Voronoi cell is filled with the most frequent color within its area for vivid, distinct color patches.

- **SVG Outlines Export**  
  Exports an SVG file containing polygon paths for further editing or laser cutting.

- **Optional Sharpening & Smoothing**  
  Users can adjust smoothing (to round polygon corners) and apply a sharpening filter to enhance edge crispness.

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/talhazain1/auto-subject-mosaic.git
   cd auto-subject-mosaic
