import cv2 as cv
import numpy as np
import os
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend
from math import atan2, degrees
# from scipy.interpolate import splprep, splev

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import base64
from io import BytesIO

def process_image(image_file):
    img_np = np.frombuffer(image_file.read(), np.uint8)
    image = cv.imdecode(img_np, cv.IMREAD_COLOR)
    print(image.shape)
    contours = detect_contours(image)
    data = generate_contourImage_and_otherData(contours, image)
    return data


def detect_contours(image):
    gray = cv.cvtColor(image , cv.COLOR_BGR2GRAY)
    _, contour_binary = cv.threshold(gray, 10, 255, cv.THRESH_BINARY_INV)
    plot_image = cv.cvtColor(contour_binary, cv.COLOR_GRAY2BGR)
    _, contour_binary = cv.threshold(gray, 10, 255, cv.THRESH_BINARY)
    contours, _ = cv.findContours(contour_binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    return contours


def generate_contourImage_and_otherData(contours, image):
    rect_image = image.copy()
    
    length = []
    areas = []
    perimeters = []
    aspect_ratios = []
    compactnesses = []
    solidities = []
    extents = []
    circularities = []

    for cnt in contours:
        area = cv.contourArea(cnt)
        if area ==0:
            area =1
        
        perimeter = cv.arcLength(cnt, True)
        if perimeter ==0:
            perimeter =1    

        x, y, w, h = cv.boundingRect(cnt)
        
        aspect_ratio = w / h if h != 0 else 0
        compactness = (perimeter ** 2) / area if area != 0 else 0

        hull = cv.convexHull(cnt)
        hull_area = cv.contourArea(hull)
        solidity = area / hull_area if hull_area != 0 else 0

        rect_area = w * h
        extent = area / rect_area if rect_area != 0 else 0

        circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter != 0 else 0

        length.append(max(h,w))
        areas.append(area)
        perimeters.append(perimeter)
        aspect_ratios.append(aspect_ratio)
        compactnesses.append(compactness)
        solidities.append(solidity)
        extents.append(extent)
        circularities.append(circularity)

    # Draw rectangle and width on image
    for cnt in contours:
        rect = cv.minAreaRect(cnt)
        width, height = rect[1]
        width = round(width, 2)
        height = round(height, 2)
        box = cv.boxPoints(rect)
        box = np.intp(box)
        cv.drawContours(rect_image, [box], 0, (0, 255, 0), 5)

        M = cv.moments(cnt)
        if M['m00'] == 0:
            continue
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        wd = max(width, height)
        cv.putText(rect_image, str(wd), (cx, cy), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    cv.putText(rect_image, str(len(contours)), (50, 50), cv.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)

    # Convert annotated image to base64
    _, annotated_buf = cv.imencode('.jpg', rect_image)
    annotated_base64 = base64.b64encode(annotated_buf).decode('utf-8')

    # Generate histogram image
    fig, axs = plt.subplots(3, 3, figsize=(20, 15))
    axs = axs.ravel()
    
    parameters = {
        "Length": length,
        "Area": areas,
        "Perimeter": perimeters,
        "Aspect Ratio": aspect_ratios,
        "Compactness": compactnesses,
        "Solidity": solidities,
        "Extent": extents,
        "Circularity": circularities,
    }

    histogram_images = {}
    for i, (param_name, values) in enumerate(parameters.items()):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(values, bins=50, color='green', alpha=0.7, log=True)
        ax.set_title(f"Distribution of {param_name} (Log Scale)")
        ax.set_xlabel(param_name)
        ax.set_ylabel("Frequency (Log Scale)")
        
        buf = BytesIO()
        plt.savefig(buf, format='jpg')
        plt.close(fig)
        buf.seek(0)
        histogram_images[param_name] = base64.b64encode(buf.read()).decode('utf-8')
    
    histogram_base64 = histogram_images

    # Compute stats
    stats = {
        param: {
            "mean": float(np.mean(vals)),
            "median": float(np.median(vals)),
            "std": float(np.std(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals))
        }
        for param, vals in parameters.items()
        
    }

    return {
        "annotated_image": annotated_base64,
        "histogram_images": histogram_base64,
        "contour_count": len(contours),
        "stats": stats,
        "parameters":parameters
    }
    
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology
from io import BytesIO
import base64
from PIL import Image

def process_and_skeletonize_base64(image, blurred_image, threshold=15):
    """
    Skeletonizes an image and returns the result as a base64-encoded PNG image.
    """
    _, binary = cv.threshold(blurred_image, threshold, 255, cv.THRESH_BINARY)
    binary_bool = binary > 0
    skeleton = morphology.skeletonize(binary_bool)
    inverted_skeleton = np.logical_not(skeleton).astype(np.uint8) * 255  # Convert to 0–255 for image

    # Convert to PIL image
    pil_img = Image.fromarray(inverted_skeleton)
    
    # Save to buffer
    buffer = BytesIO()
    pil_img.save(buffer, format="PNG")
    base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')

    return base64_image
