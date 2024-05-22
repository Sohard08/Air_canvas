import numpy as np
import cv2
from collections import deque

# Default trackbar function
def set_values(x):
    print("")

# Creating the trackbars needed for adjusting the marker colour (GREEN)
# cv2.namedWindow("Color detectors")
# cv2.createTrackbar("Upper Hue", "Color detectors", 153, 180,set_values)
# cv2.createTrackbar("Upper Saturation", "Color detectors", 255, 255,set_values)
# cv2.createTrackbar("Upper Value", "Color detectors", 255, 255,set_values)
# cv2.createTrackbar("Lower Hue", "Color detectors", 64, 180,set_values)
# cv2.createTrackbar("Lower Saturation", "Color detectors", 72, 255,set_values)
# cv2.createTrackbar("Lower Value", "Color detectors", 49, 255,set_values)

# Creating the trackbars needed for adjusting the marker colour (BLUE)
# cv2.namedWindow("Color detectors")
# cv2.createTrackbar("Upper Hue", "Color detectors", 120, 180, set_values)  # Adjusted for blue
# cv2.createTrackbar("Lower Hue", "Color detectors", 100, 180, set_values)  # Adjusted for blue
# cv2.createTrackbar("Upper Saturation", "Color detectors", 255, 255, set_values)
# cv2.createTrackbar("Lower Saturation", "Color detectors", 50, 255, set_values)  # Adjusted for blue
# cv2.createTrackbar("Upper Value", "Color detectors", 255, 255, set_values)
# cv2.createTrackbar("Lower Value", "Color detectors", 50, 255, set_values)  # Adjusted for blue

# Creating the trackbars needed for adjusting the marker colour (PINK)
cv2.namedWindow("Color detectors")
cv2.createTrackbar("Upper Hue", "Color detectors", 170, 180, set_values)  # Adjusted for pink
cv2.createTrackbar("Lower Hue", "Color detectors", 140, 180, set_values)  # Adjusted for pink
cv2.createTrackbar("Upper Saturation", "Color detectors", 255, 255, set_values)
cv2.createTrackbar("Lower Saturation", "Color detectors", 50, 255, set_values)  # Adjusted for pink
cv2.createTrackbar("Upper Value", "Color detectors", 255, 255, set_values)
cv2.createTrackbar("Lower Value", "Color detectors", 50, 255, set_values)  # Adjusted for pink

# Creating the trackbars needed for adjusting the marker colour (ORANGE)
# cv2.namedWindow("Color detectors")
# cv2.createTrackbar("Upper Hue", "Color detectors", 22, 180, set_values)  # Adjusted for orange
# cv2.createTrackbar("Lower Hue", "Color detectors", 8, 180, set_values)  # Adjusted for orange
# cv2.createTrackbar("Upper Saturation", "Color detectors", 255, 255, set_values)
# cv2.createTrackbar("Lower Saturation", "Color detectors", 50, 255, set_values)  # Adjusted for orange
# cv2.createTrackbar("Upper Value", "Color detectors", 255, 255, set_values)
# cv2.createTrackbar("Lower Value", "Color detectors", 50, 255, set_values)  # Adjusted for orange


# # Creating the trackbars needed for adjusting the marker colour(YELLOW)
# cv2.namedWindow("Color detectors")
# cv2.createTrackbar("Upper Hue", "Color detectors", 30, 180, set_values)  # Adjusted for yellow
# cv2.createTrackbar("Upper Saturation", "Color detectors", 255, 255, set_values)
# cv2.createTrackbar("Upper Value", "Color detectors", 255, 255, set_values)
# cv2.createTrackbar("Lower Hue", "Color detectors", 20, 180, set_values)  # Adjusted for yellow
# cv2.createTrackbar("Lower Saturation", "Color detectors", 100, 255, set_values)  # Adjusted for yellow
# cv2.createTrackbar("Lower Value", "Color detectors", 100, 255, set_values)  # Adjusted for yellow

# Creating the trackbars needed for adjusting the marker colour (RED)
# cv2.namedWindow("Color detectors")
# cv2.createTrackbar("Upper Hue", "Color detectors", 10, 180, set_values)  # Adjusted for red
# cv2.createTrackbar("Lower Hue", "Color detectors", 170, 180, set_values)  # Adjusted for red
# cv2.createTrackbar("Upper Saturation", "Color detectors", 255, 255, set_values)
# cv2.createTrackbar("Lower Saturation", "Color detectors", 50, 255, set_values)  # Adjusted for red
# cv2.createTrackbar("Upper Value", "Color detectors", 255, 255, set_values)
# cv2.createTrackbar("Lower Value", "Color detectors", 50, 255, set_values)  # Adjusted for red


# Giving different arrays to handle colour points of different colour
blue_points = [deque(maxlen=1024)]
green_points = [deque(maxlen=1024)]
red_points = [deque(maxlen=1024)]
yellow_points = [deque(maxlen=1024)]

# These indexes will be used to mark the points in particular arrays of specific colour
blue_index = 0
green_index = 0
red_index = 0
yellow_index = 0

# The kernel to be used for dilation purpose
kernel = np.ones((5, 5), np.uint8)

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
color_index = 0

# Canvas setup
paint_window = np.zeros((471, 636, 3), dtype=np.uint8)  # Black canvas
paint_window = cv2.rectangle(paint_window, (40, 1), (140, 65), (0, 0, 0), 2)
paint_window = cv2.rectangle(paint_window, (160, 1), (255, 65), colors[0], -1)
paint_window = cv2.rectangle(paint_window, (275, 1), (370, 65), colors[1], -1)
paint_window = cv2.rectangle(paint_window, (390, 1), (485, 65), colors[2], -1)
paint_window = cv2.rectangle(paint_window, (505, 1), (600, 65), colors[3], -1)

cv2.putText(paint_window, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paint_window, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(paint_window, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(paint_window, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(paint_window, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 2, cv2.LINE_AA)
cv2.namedWindow('Paint', cv2.WINDOW_AUTOSIZE)

# Loading the default webcam
cap = cv2.VideoCapture(0)

# Main loop
while True:
    ret, frame = cap.read()  # Reading the frame from the camera
    frame = cv2.flip(frame, 1)  # Flipping the frame
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # Converting to HSV

    # Getting trackbar values
    upper_hue = cv2.getTrackbarPos("Upper Hue", "Color detectors")
    upper_saturation = cv2.getTrackbarPos("Upper Saturation", "Color detectors")
    upper_value = cv2.getTrackbarPos("Upper Value", "Color detectors")
    lower_hue = cv2.getTrackbarPos("Lower Hue", "Color detectors")
    lower_saturation = cv2.getTrackbarPos("Lower Saturation", "Color detectors")
    lower_value = cv2.getTrackbarPos("Lower Value", "Color detectors")
    upper_hsv = np.array([upper_hue, upper_saturation, upper_value])
    lower_hsv = np.array([lower_hue, lower_saturation, lower_value])

    # Adding colour buttons to the frame
    frame = cv2.rectangle(frame, (40, 1), (140, 65), (122, 122, 122), -1)
    frame = cv2.rectangle(frame, (160, 1), (255, 65), colors[0], -1)
    frame = cv2.rectangle(frame, (275, 1), (370, 65), colors[1], -1)
    frame = cv2.rectangle(frame, (390, 1), (485, 65), colors[2], -1)
    frame = cv2.rectangle(frame, (505, 1), (600, 65), colors[3], -1)
    cv2.putText(frame, "CLEAR ALL", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 2, cv2.LINE_AA)

    # Identifying the pointer by creating its mask
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)

    # Finding contours for the pointer
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    center = None

    # If contours are found
    if len(contours) > 0:
        cnt = sorted(contours, key=cv2.contourArea, reverse=True)[0]  # Finding the largest contour
        (x, y), radius = cv2.minEnclosingCircle(cnt)  # Finding the enclosing circle
        cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)  # Drawing the circle
        M = cv2.moments(cnt)
        center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))  # Calculating the centroid

        # Checking if the user clicked on any button
        if center[1] <= 65:
            if 40 <= center[0] <= 140:  # Clear Button
                blue_points = [deque(maxlen=512)]
                green_points = [deque(maxlen=512)]
                red_points = [deque(maxlen=512)]
                yellow_points = [deque(maxlen=512)]
                blue_index = 0
                green_index = 0
                red_index = 0
                yellow_index = 0
                paint_window[67:, :, :] = 0  # Resetting the canvas to black
            elif 160 <= center[0] <= 255:
                color_index = 0  # Blue
            elif 275 <= center[0] <= 370:
                color_index = 1  # Green
            elif 390 <= center[0] <= 485:
                color_index = 2  # Red
            elif 505 <= center[0] <= 600:
                color_index = 3  # Yellow
        else:
            if color_index == 0:
                blue_points[blue_index].appendleft(center)
            elif color_index == 1:
                green_points[green_index].appendleft(center)
            elif color_index == 2:
                red_points[red_index].appendleft(center)
            elif color_index == 3:
                yellow_points[yellow_index].appendleft(center)
    # Append the next deques when nothing is detected to avoid messing up
    else:
        blue_points.append(deque(maxlen=512))
        blue_index += 1
        green_points.append(deque(maxlen=512))
        green_index += 1
        red_points.append(deque(maxlen=512))
        red_index += 1
        yellow_points.append(deque(maxlen=512))
        yellow_index += 1

    # Draw lines of all the colors on the canvas and frame
    points = [blue_points, green_points, red_points, yellow_points]
    for i in range(len(points)):
        for j in range(len(points[i])):
            for k in range(1, len(points[i][j])):
                if points[i][j][k - 1] is None or points[i][j][k] is None:
                    continue
                cv2.line(frame, points[i][j][k - 1], points[i][j][k], colors[i], 2)
                cv2.line(paint_window, points[i][j][k - 1], points[i][j][k], colors[i], 2)

    # Show all the windows
    cv2.imshow("Tracking", frame)
    cv2.imshow("Paint", paint_window)
    cv2.imshow("mask", mask)

    # If the 'q' key is pressed then stop the application
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the camera and all resources
cap.release()
cv2.destroyAllWindows()
