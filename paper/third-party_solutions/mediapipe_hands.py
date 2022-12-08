import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

class MediaPipePredictor:
    def __init__(self):
      self.model = mp_hands.Hands(
          static_image_mode=True,
          max_num_hands=4,
          min_detection_confidence=0.1)

    def eval(self):
        pass

    def __call__(self, return_loss, rescale, img_meta, img):
        image_bgr = cv2.imread(img_meta[0].data[0][0]["filename"], cv2.IMREAD_UNCHANGED)
        return self.predict(image_bgr)

    def predict(self, image_bgr):
        # Convert the BGR image to RGB before processing.
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results = self.model.process(img_rgb)

        bboxes = []
        image_height, image_width, _ = img_rgb.shape
        annotated_image = img_rgb.copy()
        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,results.multi_handedness):
                #print('hand_landmarks:', hand_landmarks)
                mp_drawing.draw_landmarks(
                    annotated_image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
                bbox = self.calc_bbox(image_height, image_width, hand_landmarks) 
                bbox_with_score = np.append(bbox, handedness.classification[0].score)
                annotated_image = cv2.rectangle(annotated_image, bbox[0:2], bbox[2:4], (255,255,255), 3)
                bboxes.append(bbox_with_score)

        # import matplotlib.pyplot as plt
        # fig = plt.figure(1)
        # ax1 = fig.add_subplot(221)
        # ax1.imshow(annotated_image)
        # plt.show()
        return [bboxes, []]


    def calc_bbox(self, image_height, image_width, hand_landmarks):
        x_max = max(i.x for i in hand_landmarks.landmark) 
        x_min = min(i.x for i in hand_landmarks.landmark)
        y_max = max(i.y for i in hand_landmarks.landmark)
        y_min = min(i.y for i in hand_landmarks.landmark)
        
        return np.array([x_min * image_width, y_min * image_height, x_max * image_width, y_max * image_height]).astype(int)
     
        
