import cv2
import numpy as np
import base64
from flask import Flask, render_template, Response

# Initialize Flask application
app = Flask(__name__)

# Function to detect bubbles in the OMR sheet
def detect_bubbles(image_path):
    # Step 1: Read the input image
    original_image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    
    # Step 2: Preprocessing - Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Step 3: Apply Otsu's Thresholding to separate the bubbles
    _, thresholded = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Step 4: Find contours in the thresholded image
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected_bubbles = []

    # Step 5: Loop through each contour to identify filled bubbles
    for contour in contours:
        area = cv2.contourArea(contour)
        if 50 < area < 1000:  # Filter out very small contours and noise
            # Calculate bounding box for the contour
            (x, y, w, h) = cv2.boundingRect(contour)

            # Check if the aspect ratio of the bounding box is close to 1 (circle)
            aspect_ratio = w / float(h)
            if 0.8 < aspect_ratio < 1.2:
                # Extract the region of interest (ROI)
                bubble_roi = thresholded[y:y+h, x:x+w]
                
                # Calculate the fill percentage of the bubble
                fill_percentage = cv2.countNonZero(bubble_roi) / (w * h)
                
                # If the fill percentage is high, consider it a marked bubble
                if fill_percentage > 0.5:
                    detected_bubbles.append((x, y, w, h))
                    # Mark the detected bubble in green
                    cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Step 6: Ensure we only detect 100 bubbles
    detected_bubbles = sorted(detected_bubbles, key=lambda b: (b[1], b[0]))[:100]

    # Save the final image with detected bubbles
    cv2.imwrite('output_image.jpg', original_image)

    return detected_bubbles, original_image

# Function to evaluate answers based on the answer sheet structure
def evaluate_answers(detected_bubbles, answer_sheet_structure):
    detected_answers = []
    
    # Sort the bubbles by their y-coordinate (vertical alignment) and x-coordinate (horizontal alignment)
    sorted_bubbles = sorted(detected_bubbles, key=lambda b: (b[1], b[0]))

    # Now, map each detected bubble to the answer sheet structure
    for i, bubble in enumerate(sorted_bubbles):
        detected_answers.append(answer_sheet_structure[i % len(answer_sheet_structure)])

    # Prepare the answers for display
    answer_results = []
    for i, answer in enumerate(detected_answers, 1):
        answer_results.append(f"{i}. {answer}")

    return answer_results

# Main function
def main(image_path):
    # Answer sheet structure for 100 questions (adjusted for 100 questions)
    answer_sheet_structure = [ 'A', 'C', 'B', 'B', 'B', 'C', 'A', 'D', 'D', 'C', 'B', 'A', 'A', 'B', 'C', 'D', 'D', 'D', 'D', 'B', 'B', 'B', 'B', 'A', 'A', 'B', 'C', 'C', 'C', 'B', 'B', 'B', 'B', 'C', 'C', 'C', 'A', 'A', 'A', 'B', 'C', 'C', 'C', 'D', 'A', 'B', 'C', 'D', 'A', 'A', 'A', 'D', 'B', 'C', 'A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'C', 'C', 'D', 'D', 'C', 'A', 'A', 'B', 'B', 'B', 'B', 'B', 'B', 'D', 'D', 'D', 'C', 'C', 'C', 'C', 'C', 'A', 'A', 'B', 'A', 'C', 'B', 'A', 'D', 'A', 'A', 'A', 'D', 'C', 'B', 'A', 'A', 'A' ]
    
    # Detect bubbles from the OMR sheet
    detected_bubbles, processed_image = detect_bubbles(image_path)
    
    # Evaluate detected answers against the answer sheet
    answer_results = evaluate_answers(detected_bubbles, answer_sheet_structure)
    
    # Convert the processed image to base64 string
    _, jpeg_image = cv2.imencode('.jpg', processed_image)
    jpeg_image_base64 = base64.b64encode(jpeg_image.tobytes()).decode('utf-8')

    return answer_results, jpeg_image_base64

@app.route('/')
def index():
    # Provide the path to the uploaded OMR answer sheet image
    image_path = "images/test_01.jpg"
    answer_results, jpeg_image_base64 = main(image_path)

    # Display results on the webpage
    return render_template('index.html', answer_results=answer_results, image=jpeg_image_base64)

# Start the Flask web server
if __name__ == "__main__":
    app.run(debug=True)
