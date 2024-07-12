import os
import cv2

def create_video_from_images(image_folder, output_video_path, frame_rate):
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg') or f.endswith('.png')]
    image_files.sort()  # 숫자 순서대로 정렬
    
    if len(image_files) == 0:
        raise ValueError("폴더 안에 이미지 파일이 없습니다!")

    # 첫 번째 이미지를 기준으로 동영상 크기 설정
    first_image = cv2.imread(os.path.join(image_folder, image_files[0]))
    height, width, _ = first_image.shape

    # 동영상 생성기 초기화
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 코덱 설정 (XVID를 사용하면 AVI 형식으로 저장)
    out = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))

    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        frame = cv2.imread(image_path)
        out.write(frame)

    out.release()
    print("동영상 생성 완료!")

if __name__ == "__main__":
    image_folder_path = "output"  # 폴더 경로를 실제로 바꾸세요
    output_video_path = "output_vedio/output_vedio_sample_3.mp4"  # 동영상 출력 경로와 파일명을 지정하세요
    frame_rate = 24  # 초당 프레임 수 (예: 24, 30, 60 등)

    create_video_from_images(image_folder_path, output_video_path, frame_rate)

#python Video.py