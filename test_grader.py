import cv2
import numpy as np

# Function to detect bubbles in the OMR sheet
def detect_bubbles(image_path):
    # Step 1: Read the input image
    original_image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    
    # Step 2: Preprocessing - Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Step 3: Apply Otsu's Thresholding to improve contrast and separate bubbles
    _, thresholded = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Visualize Preprocessed Image for Debugging
    cv2.imshow("Thresholded Image", thresholded)

    # Step 4: Find contours in the thresholded image
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected_bubbles = []

    # Step 5: Loop through each contour to identify circular bubbles
    for contour in contours:
        area = cv2.contourArea(contour)
        if 100 < area < 1000:  # Filter out small contours and noise
            # Calculate bounding box for the contour
            (x, y, w, h) = cv2.boundingRect(contour)

            # Check if the aspect ratio of the bounding box is close to 1 (circle)
            aspect_ratio = w / float(h)
            if 0.8 < aspect_ratio < 1.2:
                # Mark the detected bubble
                detected_bubbles.append((x, y, w, h))

                # Draw a bounding box around the detected bubble
                cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Step 6: Visualize Detected Bubbles for Debugging
    cv2.imshow("Detected Bubbles", original_image)

    # Step 7: Hough Circle Transform to detect circles (optional)
    circles = cv2.HoughCircles(thresholded, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
                               param1=50, param2=30, minRadius=10, maxRadius=20)
    
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            cv2.circle(original_image, (x, y), r, (0, 255, 0), 4)
            cv2.rectangle(original_image, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), 3)

    # Step 8: Final result visualization
    cv2.imshow("Final Image with Detected Circles", original_image)

    # Wait for user to close the windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return detected_bubbles

# Function to evaluate answers based on the answer sheet structure
def evaluate_answers(detected_bubbles, answer_sheet_structure):
    detected_answers = []
    
    # Assuming detected_bubbles contains (x, y, w, h) for each bubble
    # The detected_bubbles can be mapped to the answer sheet structure (order of detected bubbles)
    for i, bubble in enumerate(detected_bubbles):
        # For simplicity, assume that the order of detected bubbles directly corresponds to the answer sheet structure
        detected_answers.append(answer_sheet_structure[i % len(answer_sheet_structure)])
    
    # Compare detected answers with the answer sheet structure
    correct_answers = 0
    for detected, correct in zip(detected_answers, answer_sheet_structure):
        if detected == correct:
            correct_answers += 1

    # Calculate accuracy as the ratio of correct answers to total questions
    accuracy = (correct_answers / len(answer_sheet_structure)) * 100
    print(f"Accuracy: {accuracy:.2f}%")

    if accuracy < 70:
        print("Accuracy is below the acceptable threshold!")
    else:
        print("Detection accuracy is acceptable!")

# Main function
def main(image_path):
    # Answer sheet structure for 100 questions
    answer_sheet_structure = [
        'A', 'B', 'C', 'D', 'A', 'B', 'C', 'D', 'A', 'B',
        'C', 'D', 'A', 'B', 'C', 'D', 'A', 'B', 'C', 'D',
        'A', 'B', 'C', 'D', 'A', 'B', 'C', 'D', 'A', 'B',
        'C', 'D', 'A', 'B', 'C', 'D', 'A', 'B', 'C', 'D',
        'A', 'B', 'C', 'D', 'A', 'B', 'C', 'D', 'A', 'B',
        'C', 'D', 'A', 'B', 'C', 'D', 'A', 'B', 'C', 'D',
        'A', 'B', 'C', 'D', 'A', 'B', 'C', 'D', 'A', 'B',
        'C', 'D', 'A', 'B', 'C', 'D', 'A', 'B', 'C', 'D',
        'A', 'B', 'C', 'D', 'A', 'B', 'C', 'D', 'A', 'B',
        'C', 'D', 'A', 'B', 'C', 'D', 'A', 'B', 'C', 'D',
        'A', 'B', 'C', 'D', 'A', 'B', 'C', 'D', 'A', 'B'
    ]
    
    # Detect bubbles from the OMR sheet
    detected_bubbles = detect_bubbles(image_path)
    
    # Evaluate detected answers against the answer sheet
    evaluate_answers(detected_bubbles, answer_sheet_structure)

if __name__ == "__main__":
    # Provide the path to the OMR answer sheet image
    image_path = "images/test_01.jpg"
    main(image_path)