import cv2
from SafeZone import Safe_Zone 
import argparse
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

def extract_frames(video_path, output_folder, frame_rate=1):
    # 비디오 캡처 객체 생성
    cap = cv2.VideoCapture(video_path)

    # 프레임 수 정보
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate_actual = int(cap.get(cv2.CAP_PROP_FPS))

    # 요청된 프레임레이트보다 낮으면 오류 반환
    if frame_rate_actual < frame_rate:
        cap.release()
        raise ValueError("The frame rate of the video is lower than the requested frame rate.")

    # 프레임 인덱스 초기화
    frame_index = 0
    model = YOLO('yolo_pt/best_new.pt')

    consecutive_directions = []
    max_consecutive_directions = 5

    while True:
        # 프레임 읽기
        ret, frame = cap.read()

        #frame_filename = f"frame/frame_{str(frame_index).zfill(6)}.jpg"
        #cv2.imwrite(frame_filename,frame)

        # 비디오의 끝에 도달하면 반복 종료
        if not ret:
            break
        
        #try:
        safe = Safe_Zone(frame)
        save_mask, masks2 = safe.SAM()
        #vs = safe.VanishingPoint(masks2)
        vs = safe.VanishingPoint_Triangle(masks2)
        #print(vs)

        if vs is None or vs[0] == 0 or vs[1] == 0:
            try:
                vs = vs_previous
            except:
                vs = [int(frame.shape[0]/2),int(frame.shape[1]/2)]
        else:
            vs_previous = vs
            
        if frame_index == 0:
            frame, pr_mask, pr_x1, pr_x2 = safe.Angular_Bisector(masks2,vs)
        else:
            frame, pr_mask, pr_x1, pr_x2 = safe.Angular_Bisector(masks2,vs,pr_mask,pr_x1,pr_x2)
        
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

        consecutive_directions.append(dirg)
        if len(consecutive_directions) > max_consecutive_directions:
            consecutive_directions.pop(0)  # Remove the oldest direction

        # Check if there are 5 consecutive "right" or "left" directions
        if consecutive_directions.count('right') >= max_consecutive_directions:
            print('Right direction detected for 5 consecutive frames!')
        elif consecutive_directions.count('left') >= max_consecutive_directions:
            print('Left direction detected for 5 consecutive frames!')
        
        

        cv2.circle(masked_region, (int(vs[0]), int(vs[1])), 10, (0, 0, 255), -1)
        image_rgb = cv2.cvtColor(masked_region, cv2.COLOR_BGR2RGB)
        frame_filename = f"seg_out/frame_{str(frame_index).zfill(6)}_{dirg}.jpg"
        cv2.imwrite(frame_filename, image_rgb)

        # 요청된 프레임레이트로 프레임 저장
        #if frame_index % (frame_rate_actual // frame_rate) == 0:
            # 이미지로 저장 (파일명 예시: frame_000001.jpg)
        frame_filename = f"{output_folder}/frame_{str(frame_index).zfill(6)}_{dirg}.jpg"
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.imwrite(frame_filename, frame)
        #except:
            #pass

        frame_index += 1

    # 캡처 객체 해제
    cap.release()

    print(f"Total frames in the video: {total_frames}")
    print(f"Extracted frames at {frame_rate}fps and saved to '{output_folder}'")


# 30fps로 이미지 추출

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path')
    parser.add_argument('--output_folder')
    parser.add_argument('--frame_rate',default=1,type=int)

    args = parser.parse_args()

    video_path = args.video_path
    output_folder = args.output_folder
    frame_rate = args.frame_rate

    extract_frames(video_path, output_folder, frame_rate=frame_rate)


if __name__ == "__main__":
    main()


#python main.py --video_path "dataset_video_sidewalk/20230805_150411.mp4" --output_folder "output"
#python main.py --video_path "dataset_video_sidewalk/sample_2.mp4" --output_folder "output"
# python main.py --video_path "Sample/video_sample.mp4" --output_folder "output"
