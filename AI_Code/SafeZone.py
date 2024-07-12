import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from PIL import ImageDraw, Image
import sys
sys.path.append("..")
from MobileSAM.mobile_sam import sam_model_registry, SamPredictor
from VanishingPoint.main import GetLines,GetVanishingPoint
import math
from ultralytics import YOLO

# SAM
sam_checkpoint = "MobileSAM/weights/mobile_sam.pt"
model_type = "vit_t"
device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
det_model = YOLO('yolo_pt/best.pt')

def show_mask2(mask):
    color = np.array([30, 144, 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    return mask_image[:,:,:3]

def fline(x1, y1, x2, y2):
    # 두 점을 지나는 직선의 기울기를 구합니다.
    if x2 - x1 == 0:
        x2=x2+1
        #raise ValueError('오류')
    m = (y2 - y1) / (x2 - x1)
    # y절편을 구합니다. (한 점을 대입하여 b를 구할 수 있습니다.)
    b = y1 - m * x1
    return m, b
    
def cross(m1,b1,m2,b2):
    x = (b2-b1)/(m1-m2)
    y = m1*x+b1
    return x, y
    
def bisecting_line(m1, m2, b1, b2, x, y):
    # 두 직선의 각을 구합니다
    angle1 = math.atan2(1,m1)
    angle2 = math.atan2(1,m2)
            
    # 두 각의 평균을 구해 이등분하는 각의 크기를 계산합니다
    bisecting_angle = (angle1 + angle2)/2

    # 이등분하는 각의 크기를 이용하여 이등분하는 직선의 기울기를 구합니다
    bisecting_slope = math.tan(bisecting_angle)

    # 이등분하는 직선의 y절편을 계산합니다
    bisecting_y_intercept = y - bisecting_slope * x

    # 이등분하는 직선의 방정식 문자열 생성

    return bisecting_slope, bisecting_y_intercept

def extract_masked_region(image, mask):
    # Mask 값이 1인 픽셀만 추출하여 새로운 이미지 생성
    masked_image = np.copy(image)
    masked_image[mask != 1] = 0

    return masked_image

def detection(image):
    pred = det_model.predict(image)
    x_lst = []
    for i in pred[0].boxes:
        if i.xyxy[0][3]>= image.shape[0]/2:
            x_lst.append((i.xyxy[0][0]+i.xyxy[0][2])/2)
    return x_lst

class Safe_Zone():

    def __init__(self,image):
        self.image = image
    
    def SAM(self):
        image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        sam.to(device=device)
        sam.eval()
        predictor = SamPredictor(sam)

        predictor.set_image(image)

        input_point = np.array([[image.shape[1]/2-100, +image.shape[0]-10],[image.shape[1]/2+100, image.shape[0]-10],[image.shape[1]/2, image.shape[0]-100]])
        input_label = np.array([1, 1, 1])

        masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,)

        mask_input = logits[np.argmax(scores), :, :]  # Choose the model's best mask

        masks2, _, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        mask_input=mask_input[None, :, :],
        multimask_output=False,)
        
        save_mask = show_mask2(masks[2]).astype(np.uint8)

        return save_mask, masks2

    def VanishingPoint(self,masks2):
        masks2 = np.expand_dims(masks2, axis=2)
        masks2 = np.squeeze(masks2)
        masked_region = extract_masked_region(self.image, masks2)

        Lines = GetLines(masked_region)
        VanishingPoint = GetVanishingPoint(Lines)
        return VanishingPoint
    
    def VanishingPoint_Triangle(self,masks2):
        masks2 = np.expand_dims(masks2, axis=2)
        masks2 = np.squeeze(masks2)

        image = np.uint8(masks2) * 255
        # 가장 큰 contour를 찾습니다.
        contours, hierarchy = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        max_contour = max(contours, key=cv2.contourArea)

        # contour를 근사화하여 삼각형을 구합니다.
        approx = cv2.approxPolyDP(max_contour, 0.1 * cv2.arcLength(max_contour, True), True)

        if len(approx) < 3:
            return None
        else:
            lowest_y = None
            lowest_y_coord = None

            for coord in approx:
                y = coord[0][1]
                if lowest_y is None or y < lowest_y:
                    lowest_y = y
                    lowest_y_coord = coord

            return lowest_y_coord[0]

    
    def Angular_Bisector(self,masks2,VanishingPoint,pr_mask = None, pr_x1 = None, pr_x2 = None):
        
        x_lst = detection(self.image)
        
        mask_x = np.sum(masks2,axis = 1)[0]
        for i in range(len(mask_x[::-1])):
            if mask_x[::-1][i] !=0:
                max_x = self.image.shape[1]-1-i
                break
        for i in range(len(mask_x)):
            if mask_x[i] !=0:
                min_x = i
                break
        for i in range(len(masks2[0,:,max_x])):
            if masks2[0,:,max_x][i] != False:
                max_y = i
                break
        for i in range(len(masks2[0,:,min_x])):
            if masks2[0,:,min_x][i] != False:
                min_y = i
                break
        
        image = self.image

        mid_x, mid_y =  int(VanishingPoint[0]), max(0,int(VanishingPoint[1])-300)

        m1,b1 = fline(mid_x,mid_y,min_x,min_y)
        m2,b2 = fline(mid_x,mid_y,max_x,max_y)
        box_left, box_right = [], []
        for i in x_lst:
            if i<mid_x:
                box_left.append(i)
            else:
                box_right.append(i)
        box_left.append(0)
        box_right.append(self.image.shape[0])
        
        box_left_max = max(box_left)
        box_right_min = min(box_right)

        mm, bm = bisecting_line(m1, m2, b1, b2,mid_x,mid_y)
        x_mid = (int(masks2.shape[1])-bm)/mm
        x1 = (image.shape[0]-b1)/(m1+1e-12)#max((image.shape[0]-b1)/(m1+1e-12),box_left_max)
        x2 = (image.shape[0]-b2)/(m2+1e-12)#min((image.shape[0]-b2)/(m2+1e-12),box_right_min)

        color = (255, 0, 255)
        alpha = 0.2
        bgra_color = (*color, int(255 * alpha))
        height, width, _ = image.shape
        bgra_image = np.zeros((height, width, 3), dtype=np.uint8)

        if pr_mask is None:
            pts = np.array([[min(int((x1 + x_mid) // 2+50),int((x2 + x_mid) // 2-50)), int(masks2.shape[1] - 200)],[max(int((x1 + x_mid) // 2+50),int((x2 + x_mid) // 2-50)), int(masks2.shape[1] - 200)],
                            [int((x2 + x_mid) // 2), int(masks2.shape[1])],[int((x1 + x_mid) // 2), int(masks2.shape[1])]])
            cv2.fillConvexPoly(bgra_image, pts, bgra_color)
            pr_mask = masks2
            pr_x1 = max(0, ((x1 + x_mid) // 2))
            pr_x2 = min(image.shape[0], ((x2 + x_mid) // 2))
            image = cv2.addWeighted(image, 1, bgra_image, 1, 0)
        elif (pr_mask * masks2).sum() / pr_mask.sum() < 0.9:
            pts = np.array([[min(int(pr_x1+50),int(pr_x2-50)), int(masks2.shape[1] - 200)],[max(int(pr_x1+50),int(pr_x2-50)), int(masks2.shape[1] - 200)],
                            [int(pr_x2), int(masks2.shape[1])],[int(pr_x1), int(masks2.shape[1])]])
            cv2.fillConvexPoly(bgra_image, pts, bgra_color)
            pr_mask = masks2
            image = cv2.addWeighted(image, 1, bgra_image, 1, 0)
        else:
            n_x1 = max(0, int(((x1 + x_mid) // 2) * 0.2 + pr_x1 * 0.8))
            n_x2 = min((image.shape[0]), int(((x2 + x_mid) // 2) * 0.2 + pr_x2 * 0.8))
            pts = np.array([[min(n_x1+50,n_x2-50), int(masks2.shape[1] - 200)],[max(n_x1+50,n_x2-50), int(masks2.shape[1] - 200)],
                            [n_x2, int(masks2.shape[1])],[n_x1, int(masks2.shape[1])]])
            cv2.fillConvexPoly(bgra_image, pts, bgra_color)
            pr_mask = masks2
            pr_x1 = n_x1
            pr_x2 = n_x2
            image = cv2.addWeighted(image, 1, bgra_image, 1, 0)

        return image, pr_mask, int(pr_x1), int(pr_x2), int(masks2.shape[1]), int(masks2.shape[1]), int(min(pr_x1+50,pr_x2-50)), int(max(pr_x1+50,pr_x2-50)), int(masks2.shape[1] - 200), int(masks2.shape[1] - 200)