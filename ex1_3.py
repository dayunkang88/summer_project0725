from ultralytics import solutions
import cv2

video_path = "ex2.mp4"
cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)            
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

counter = solutions.ObjectCounter(
  show=True,
  region=[(430, 150), (550, 150)],    
  model = "yolo11l.pt",
)

while cap.isOpened():
  ret, img = cap.read()
  if ret:
    results = counter(img)

    if cv2.waitKey(30) == ord('q'):
      break
  
  else:
    break

cap.release()
cv2.destroyAllWindows()