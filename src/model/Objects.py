import cv2
from ultralytics import YOLO

model = YOLO('yolov8s.pt')
# Path to your video file
video_path = "D:/1/Hacka/project/input.mp4"
fixed_size = (640, 480) #fixed window size
cap = cv2.VideoCapture(video_path)

# Check successfully opened video
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while True:
    ret, frame = cap.read()  
    if not ret:
        print("Reached the end of the video.")
        break
    results = model(frame)
    annotated_frame = results[0].plot()
    resized_frame = cv2.resize(annotated_frame, fixed_size) #resize
    
    # Display 
    cv2.imshow("Video", resized_frame)
    
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release close any open windows
cap.release()
cv2.destroyAllWindows()
