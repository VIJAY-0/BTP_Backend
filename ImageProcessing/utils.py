import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from math import atan2, degrees
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from skimage import morphology
from PIL import Image

matplotlib.use('Agg') 


def process_image(image_file):
    img_np = np.frombuffer(image_file.read(), np.uint8)
    image = cv.imdecode(img_np, cv.IMREAD_COLOR)
    print(image.shape)
    contours = detect_contours(image)
    data = generate_contourImage_and_otherData(contours, image)
    
    skeleton = skeletonize(image)
    
    data2 = generate_branching_dist(image)
    data['branching_dist'] = data2
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
    units ={"Length":'px' , "Area":'px^2' , "Perimeter":'px' , "Aspect Ratio":'' , "Compactness":'' , "Solidity":'' , "Extent":'' , "Circularity":''}  
    
    
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
        "parameters":parameters,
        "units":units
    }
    

def skeletonize(image):
    gray = cv.cvtColor(image , cv.COLOR_BGR2GRAY)
    _, binary = cv.threshold(image,15, 255, cv.THRESH_BINARY )
    binary_bool = binary > 0

    # Perform skeletonization using skimage
    skeleton = morphology.skeletonize(binary_bool)
    return skeleton
    

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

from scipy.ndimage import convolve





def generate_branching_dist(image):

    gray = cv.cvtColor(image , cv.COLOR_BGR2GRAY)
    # blurred_image = cv.GaussianBlur(gray, (5, 5), 10)
    _, binary = cv.threshold(gray,15, 255, cv.THRESH_BINARY )
    binary_bool = binary > 0
    skeleton = morphology.skeletonize(binary_bool)

    _, intersections = detect_endpoints_intersections(skeleton)
    # Convert skeleton to an 8-bit image for contour detection
    binary_8bit = (binary_bool * 255).astype(np.uint8)
    skeleton_8bit = (skeleton * 255).astype(np.uint8)
    cleaned_skeleton = skeleton_8bit.copy()
    # Find contours
    contours, _ = cv.findContours(binary_8bit, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # Create a color image to draw contours and bounding rectangles
    color_skeleton = cv.cvtColor(skeleton_8bit, cv.COLOR_GRAY2BGR)
    
    
    class_masks = {
        'Linear': np.zeros_like(gray, dtype=np.uint8),
        'Less Branched': np.zeros_like(gray, dtype=np.uint8),
        'More Branched': np.zeros_like(gray, dtype=np.uint8),
        'Highly Branched (Mesh)': np.zeros_like(gray, dtype=np.uint8)
    }
    
    for i, contour in enumerate(contours):
        # Get the bounding box of the contour
        x, y, w, h = cv.boundingRect(contour)
        
        # Count the number of intersections within the bounding box
        num_intersections = np.sum(intersections[y:y+h, x:x+w] > 0)
        
        # Classify the contour
        classification = classify_contour(num_intersections)
        
        # Draw the contour on the corresponding class mask
        cv.drawContours(class_masks[classification], [contour], -1, 255, thickness=cv.FILLED)
    
    
    results = {}

    for class_name, class_mask in class_masks.items():
        # Create a masked image
        masked_image = cv.bitwise_and(image, image, mask=class_mask)

        # Invert the masked image for better visibility
        inverted_masked_image = cv.bitwise_not(masked_image)
        
          # Convert to base64-encoded PNG
        pil_img = Image.fromarray(inverted_masked_image)
        buffer = BytesIO()
        pil_img.save(buffer, format="PNG")
        encoded_image = base64.b64encode(buffer.getvalue()).decode("utf-8")

        # Calculate statistics
        num_contours_in_class = sum(
            1 for contour in contours
            if classify_contour(np.sum(intersections[
                cv.boundingRect(contour)[1]:cv.boundingRect(contour)[1] + cv.boundingRect(contour)[3],
                cv.boundingRect(contour)[0]:cv.boundingRect(contour)[0] + cv.boundingRect(contour)[2]
            ] > 0)) == class_name
        )

        # Save results in dictionary
        results[class_name] = {
            "inverted_masked_image": encoded_image,
            "num_contours": int(num_contours_in_class)
        }
        
        
    return results



# Define classification thresholds
def classify_contour(num_intersections):
    if num_intersections == 0:
        return 'Linear'
    elif num_intersections <= 3:
        return 'Less Branched'
    elif num_intersections <= 10:
        return 'More Branched'
    else:
        return 'Highly Branched (Mesh)'


def detect_endpoints_intersections(skeleton):
    # Ensure skeleton is binary and of type uint8
    skeleton = (skeleton > 0).astype(np.uint8)

    # Define a kernel to detect intersections
    kernel = np.array([[1, 1, 1],
                       [1, 1, 1],
                       [1, 1, 1]])

    # Convolve the kernel with the skeleton image
    neighbor_count = cv.filter2D(skeleton, -1, kernel)

    # Detect endpoints (pixels with exactly 2 neighbors)
    endpoints = ((skeleton == 1) & (neighbor_count == 2)).astype(np.uint8)

    # Detect intersections (pixels with more than 3 neighbors)
    intersections = ((skeleton == 1) & (neighbor_count > 3)).astype(np.uint8)
    # if len(intersections.shape) == 3:
    #     intersections = cv.cvtColor(intersections, cv.COLOR_BGR2GRAY)

    # _, intersections_bin = cv.threshold(intersections, 1, 255, cv.THRESH_BINARY)

    # _, labeled_intersections = cv.connectedComponents(intersections_bin, connectivity=8)
    # Ensure that each kernel match corresponds to one intersection point
    _, labeled_intersections = cv.connectedComponents(intersections, connectivity=8)
    return endpoints, labeled_intersections

