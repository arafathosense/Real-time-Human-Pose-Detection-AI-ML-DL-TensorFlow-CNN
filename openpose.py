import cv2 as cv
import numpy as np

# ------------------- User-configurable paths -------------------
# Input video path
input_video_path = r"C:\Users\iTparK\Desktop\HOSEN ARAFAT Projects\Real-time-Human-Pose-Detection\dance.mp4"
# Output video path
output_video_path = r"C:\Users\iTparK\Desktop\HOSEN ARAFAT Projects\Real-time-Human-Pose-Detection\output.mp4"

# Threshold for confidence
thr = 0.2
# Resize input frames (width, height)
inWidth = 368
inHeight = 368

# ------------------- Body parts and pose pairs -------------------
BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
               "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
               "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
               "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }

POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
               ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
               ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
               ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
               ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]

# ------------------- Load OpenPose TensorFlow model -------------------
net = cv.dnn.readNetFromTensorflow("graph_opt.pb")

# ------------------- Video processing -------------------
cap = cv.VideoCapture(input_video_path)
if not cap.isOpened():
    print("Error: Cannot open video:", input_video_path)
    exit()

# Prepare video writer
frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv.CAP_PROP_FPS))

fourcc = cv.VideoWriter_fourcc(*'mp4v')
out = cv.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

while True:
    hasFrame, frame = cap.read()
    if not hasFrame:
        break

    # Prepare input blob
    net.setInput(cv.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight),
                                     (127.5, 127.5, 127.5), swapRB=True, crop=False))
    out_net = net.forward()
    out_net = out_net[:, :19, :, :]  # Only first 19 channels (body parts)

    points = []
    for i in range(len(BODY_PARTS)):
        heatMap = out_net[0, i, :, :]
        _, conf, _, point = cv.minMaxLoc(heatMap)
        x = (frame_width * point[0]) / out_net.shape[3]
        y = (frame_height * point[1]) / out_net.shape[2]
        points.append((int(x), int(y)) if conf > thr else None)

    # Draw skeleton
    for pair in POSE_PAIRS:
        partFrom = pair[0]
        partTo = pair[1]
        idFrom = BODY_PARTS[partFrom]
        idTo = BODY_PARTS[partTo]

        if points[idFrom] and points[idTo]:
            cv.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
            cv.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
            cv.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)

    # Display inference time
    t, _ = net.getPerfProfile()
    freq = cv.getTickFrequency() / 1000
    cv.putText(frame, '%.2fms' % (t / freq), (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    # Show frame
    cv.imshow('Human Pose Detection', frame)
    out.write(frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv.destroyAllWindows()
print("Output saved to:", output_video_path)
