# BODA
시각장애인을 위한 보행 방향 및 길 안내 서비스 제안

<br/>

## 1. 배경 & 목적
 
- 대회명 : 2023 장애인 분야 해커톤 대회(장애 플러스 기술) - 장애인을 위한 APP 개발
- 디지털 기술을 활용하여 장애인의 삶의 질 향상에 기여할 수 있는 APP 개발

<br/>

## 2. 주최/주관 & 성과

- 주최/주관: 한국장애인재단
- 후원: 보건복지부
- 참가 대상: 장애인의 삶의 질 향상을 위한 디지털 기술을 활용한 아이디어를 갖고 있거나 관련 APP 개발에 관심 있는 자
- 성과 : 2023 장애인 분야 해커톤 대회(장애 플러스 기술) 메이커톤 부분 입선(4등)

<br/>

![KakaoTalk_20231121_182600767](https://github.com/user-attachments/assets/4348cfd4-f97c-40bc-9aae-36fd5c85b35a)

<br/>

## 3. 프로젝트 기간

- 1차 서류 심사 : 2023.05 - 2023.06
- 본선 : 2023.06 - 2023.09
- 본선 발표 : 2023.10.30

<br/>

## 4. 프로젝트 소개

&nbsp;&nbsp;&nbsp;&nbsp; 국내에 많은 시각장애인분들이 계심. 그러나 시각장애인분들을 위한 점자 블럭과 같은 것들이 미관상의 이유로 많이 사람짐. 이러한 이유로 특히 후천적 시각장애인분들이 도보 보행에 많은 어려움을 겪고있어서 이를 해결하고자 다음과 같은 서비스를 제안함
<br/>

![스크린샷 2024-07-12 230648](https://github.com/user-attachments/assets/cb915fac-9df3-48aa-891c-f2cdafbf1055)

<br/>

1. 보행 방향 안내 서비스  
시각장애인이 도로의 가운데로 보행할 수 있도록 보행로의 안전 구역을 탐지, 안전구역에서 벗어날 경우 알림. 안전구역 내 위험한 객체가 탐지 되었을 경우 피해서 가도록 안내함

<br/>

![스크린샷 2024-07-12 230745](https://github.com/user-attachments/assets/eb526561-248e-4a19-8d98-0d45e656c8f5)

<br/>

2. 길 안내 서비스  
음성으로 목적지를 설정하면 현재 위치에서 목적지까지 길 안내 서비스 제공. 각 코너나 신호등마다 정보를 제공해 갈림길에서도 올바른 방향으로 안내함

<br/>

![스크린샷 2024-07-12 231016](https://github.com/user-attachments/assets/82e29204-6e76-40a5-9d07-b9fa804704f0)

<br/>

&nbsp;&nbsp;&nbsp;&nbsp; 인도 구분을 위한 Segmentation, 안전 구역 설정을 위한 Vanishing Point Detection, 위험 객체 탐지를 위한 Object Detection 기술을 활용함. 보행로를 segmentation하고 길의 중앙을 Vanishing point detection을 활용하여 안전구역으로 설정함. 기준점을 중심으로 안전 구역을 벗어나면 다시 안전 구역 안으로 들어오게 유도. 위험 객체가 탐지될 경우 해당 객체를 피하여 보행하도록 방향 조정. 최종 Flow는 다음과 같음

<br/>

![스크린샷 2024-07-12 231345](https://github.com/user-attachments/assets/9660e3a9-c770-4b55-970a-2029f8d21b8e)

<br/>

&nbsp;&nbsp;&nbsp;&nbsp; AI를 제외한 Backend,Frontend는 [Boda organization](https://github.com/BO-DA) 참고

<br/>

## 5. Process

### ch.1 Data Preparation 

- 보행 영상 데이터 처리
- 장애물 데이터 처리

---

### ch.2 AI Modeling  

- Object Detection(Yolo V8)
- Vanishing point detection
- Segmentation(SAM)

---

### ch.3 Backend

- Kakao 지도 API
- 구글 STT 및 TTS
- T MAP API
- RTMP Stream

<br/>

## 6. 프로젝트 팀원 및 담당 역할

**[팀원]**

- 학부생 5명

**[담당 역할]**

- 기획
- AI 모델링
- Backend, AI 통신 구축

<br/>

## 7. 발표 자료&참고 자료

[2023 장애인해커톤 대회 Team BODA 발표자료]()  
