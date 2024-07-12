from ultralytics import YOLO
import cv2
import cvzone
import math
import time
import pygame

# Initialize pygame mixer
pygame.mixer.init()

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

model = YOLO("../Yolo-Weights/yolov8n.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

myColor = (0,0,255)
countdown_start = time.time()  # Start the countdown when the app starts
countdown_duration = 516  # Initial countdown duration in seconds (8.6 minutes)

bottle_frame_count = 0  # Counter for consecutive frames with bottle detected
bottle_detection_counter = 0  # Counter out of 105
first_detection_time = None  # Track the time of the first detection
last_counter_increase_time = None  # Track the time of the last counter increase
sound_played_time = None  # Track the time when the sound was played
bottle_detected_after_sound = False  # Track if a bottle was detected after the sound

while True:
    success, img = cap.read()
    results = model.predict(source=img, classes=[39], verbose=False)  # Only detect "bottle"
    bottle_detected = False
    for r in results:
        boxes = r.boxes
        for box in boxes:

            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            w, h = x2-x1, y2-y1
            # cvzone.cornerRect(img,(x1,y1,w,h))

            # Confidence
            conf = math.ceil((box.conf[0]*100))/100

            # Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]
            if conf > 0.2:
                if currentClass == "bottle":
                    myColor = (0,255,0)
                    bottle_detected = True
                else:
                    myColor = (255,0,0)
                cvzone.putTextRect(img,f'{classNames[cls]} {conf}',(max(0,x1),max(35,y1)), scale=1, thickness=1, colorB=myColor, colorT=(255,255,255), colorR=myColor, offset=5)
                cv2.rectangle(img, (x1, y1), (x2, y2), myColor, 3)

    if bottle_detected:
        if first_detection_time is None:
            first_detection_time = time.time()  # Record the time of the first detection
        bottle_frame_count += 1
        current_time = time.time()
        if current_time - first_detection_time <= 5:  # Check if within 5 seconds of the first detection
            if bottle_frame_count >= 2:  # Check if bottle is detected at least twice
                if last_counter_increase_time is None or current_time - last_counter_increase_time >= 10:
                    if sound_played_time is None or current_time - sound_played_time >= 10:  # Check if 10 seconds have passed since the sound was played
                        bottle_detection_counter = min(bottle_detection_counter + 1, 105)
                        last_counter_increase_time = current_time  # Update the last counter increase time
                        if countdown_duration - (current_time - countdown_start) > 0:  # Only increase the timer if it hasn't reached 0
                            countdown_duration += 516  # Increase the countdown duration by 10 seconds
                bottle_frame_count = 0  # Reset the frame count after increasing the counter
                first_detection_time = None  # Reset the first detection time
        else:
            bottle_frame_count = 0  # Reset the frame count if more than 5 seconds have passed
            first_detection_time = None  # Reset the first detection time
    else:
        bottle_frame_count = 0  # Reset the frame count if bottle is not detected
        first_detection_time = None  # Reset the first detection time

    if countdown_start is not None:
        elapsed_time = time.time() - countdown_start
        remaining_time = max(0, countdown_duration - elapsed_time)
        cvzone.putTextRect(img, f'Countdown: {remaining_time:.1f}s', (50, 50), scale=2, thickness=2, colorB=(0, 0, 255), colorT=(255, 255, 255), colorR=(0, 0, 255), offset=10)
        if remaining_time == 0:
            pygame.mixer.music.load('../Test/note.wav')  # Load sound
            pygame.mixer.music.play()  # Play sound when countdown ends
            countdown_start = time.time()  # Restart the countdown timer
            countdown_duration = 516  # Reset the countdown duration to 8.6 minutes
            sound_played_time = time.time()  # Record the time when the sound was played
            bottle_detected_after_sound = False  # Reset the flag for bottle detection after sound

    if bottle_detection_counter >= 105:
        countdown_start = None  # Stop the countdown
        pygame.mixer.music.stop()  # Stop playing the noise

    # Check if bottle is detected within 10 seconds after the sound is played
    if sound_played_time is not None and time.time() - sound_played_time <= 10:
        if bottle_detected and not bottle_detected_after_sound:
            bottle_detection_counter = min(bottle_detection_counter + 1, 105)
            bottle_detected_after_sound = True  # Set the flag to indicate a bottle was detected after the sound

    cvzone.putTextRect(img, f'Bottle Detection Counter: {bottle_detection_counter}/105', (50, 100), scale=2, thickness=2, colorB=(0, 255, 0), colorT=(255, 255, 255), colorR=(0, 255, 0), offset=10)

    cv2.imshow("Image",img)
    cv2.waitKey(1)

