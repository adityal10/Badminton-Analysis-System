from ultralytics import YOLO
import cv2

class CourtLineDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def predict(self, image):
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        outputs = self.model(img_rgb)
        keypoints = outputs[0].keypoints.xy.squeeze().cpu().numpy().flatten()
        original_h, original_w = img_rgb.shape[:2]

        return keypoints
    
    def draw_keypoints(self, image, keypoints):
        for i in range(0, len(keypoints), 2):
            x = int(keypoints[i])
            y = int(keypoints[i+1])

            cv2.putText(image, str(i//2), (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.circle(image, (x,y), 5, (0, 0, 255), -1)
        
        return image
    
    def draw_keypoints_on_video(self, video_frames, keypoints):
        output_video_frames = []
        for frame in video_frames:
            frame = self.draw_keypoints(frame, keypoints)
            output_video_frames.append(frame)

        return output_video_frames

