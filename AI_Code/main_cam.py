import cv2
import cv2
from SafeZone import Safe_Zone 
import numpy as np
from ultralytics import YOLO

CONFIDENCE_THRESHOLD = 0.6
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)

class_list = ["bicycle", "car", "bollard", "pole", "tree_trunk", "scooter", "movable_signage"]

def extract_masked_region(image, mask):
    # Mask 값이 1인 픽셀만 추출하여 새로운 이미지 생성
    masked_image = np.copy(image)
    masked_image[mask != 1] = 0

    return masked_image

webcam = cv2.VideoCapture(0)

if not webcam.isOpened():
    print("Could not open webcam")
    exit()

model = YOLO('yolo_pt/best.pt')

consecutive_directions = []
max_consecutive_directions = 5

while webcam.isOpened():
    status, frame = webcam.read()

    frame_index = 0

    if status:
        safe = Safe_Zone(frame)
        save_mask, masks2 = safe.SAM()

        vs = safe.VanishingPoint_Triangle(masks2)

        if vs is None or vs[0] == 0 or vs[1] == 0:
            try:
                vs = vs_previous
            except:
                vs = [int(frame.shape[0]/2),int(frame.shape[1]/2)]
        else:
            vs_previous = vs
            
        if frame_index == 0:
            frame, pr_mask, pr_x1, pr_x2, y1, y2, x3, x4, y3, y4 = safe.Angular_Bisector(masks2,vs)
        else:
            frame, pr_mask, pr_x1, pr_x2, y1, y2, x3, x4, y3, y4 = safe.Angular_Bisector(masks2,vs,pr_mask,pr_x1,pr_x2)
        
        masks_save = np.expand_dims(masks2, axis=2)
        masks_save = np.squeeze(masks_save)
        masked_region = extract_masked_region(frame, masks_save)

        detection = model(frame)[0]
        for data in detection.boxes.data.tolist():
            confidence = float(data[4])
            if confidence < CONFIDENCE_THRESHOLD:
                continue
        
            xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
            label = int(data[5])
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), GREEN, 2)
            cv2.putText(frame, class_list[label]+' '+str(round(confidence, 2))+'%', (xmin, ymin), cv2.FONT_ITALIC, 1, WHITE, 2)

        
        if frame.shape[1]/2 < pr_x1:
            #print('right')
            dirg = 'right'
        elif frame.shape[1]/2 > pr_x2:
            #print('left')
            dirg = 'left'
        else:
            #print('Normal')
            dirg = 'Normal'
        
        rt_dirg = ""
        consecutive_directions.append(dirg)
        if len(consecutive_directions) > max_consecutive_directions:
            consecutive_directions.pop(0)  # Remove the oldest direction

        # Check if there are 5 consecutive "right" or "left" directions
        if consecutive_directions.count('right') >= max_consecutive_directions:
            print('Right')
            rt_dirg = 'Right'
        elif consecutive_directions.count('left') >= max_consecutive_directions:
            print('Left')
            rt_dirg = 'Left'
        else:
            print('Normal')
            rt_dirg = 'Normal'
        
        cv2.circle(masked_region, (int(vs[0]), int(vs[1])), 10, (0, 0, 255), -1)
        image_rgb = cv2.cvtColor(masked_region, cv2.COLOR_BGR2RGB)
        
        cv2.imshow("test", frame)
        frame_index += 1

        f = open('./temp.txt','w')
        f.write(f'{pr_x1} {y1} {pr_x2} {y2} {x3} {y3} {x4} {y4} {rt_dirg}')
        f.close()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()