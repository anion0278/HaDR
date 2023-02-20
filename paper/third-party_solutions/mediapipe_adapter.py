import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

class MediaPipePredictor:
    def __init__(self, min_confidence_score):
      self.model = mp_hands.Hands(
          static_image_mode=True,
          max_num_hands=4,
          min_tracking_confidence=min_confidence_score,
          min_detection_confidence=min_confidence_score)

    def eval(self):
        pass

    def __call__(self, return_loss, rescale, img_meta, img):
        image_bgr = cv2.imread(img_meta[0].data[0][0]["filename"], cv2.IMREAD_UNCHANGED)
        return self.predict(image_bgr)

    def predict(self, image_bgr):
        # Convert the BGR image to RGB before processing.
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results = self.model.process(img_rgb)

        masks = []
        bboxes = []
        image_height, image_width, _ = img_rgb.shape
        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,results.multi_handedness):
                #print('hand_landmarks:', hand_landmarks)
                self.draw_landmarks(img_rgb, hand_landmarks)
                bbox = self.calc_bbox(image_height, image_width, hand_landmarks) 
                bbox_with_score = np.append(bbox, handedness.classification[0].score)
                bboxes.append(bbox_with_score)
                masks.append({"counts":"", "size":0})

        # import matplotlib.pyplot as plt
        # fig = plt.figure(1)
        # ax1 = fig.add_subplot(221)
        # ax1.imshow(img_rgb)
        # plt.show()

        # this is required to run the same evaluation
        return (np.array([bboxes]), np.array([masks]),), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    def draw_landmarks(self, annotated_image, hand_landmarks):
        mp_drawing.draw_landmarks(
                    annotated_image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())


    def calc_bbox(self, image_height, image_width, hand_landmarks):
        x_max = np.clip(max(i.x * image_width for i in hand_landmarks.landmark), 0, image_width)
        x_min = np.clip(min(i.x * image_width for i in hand_landmarks.landmark), 0, image_width)
        y_max = np.clip(max(i.y * image_height for i in hand_landmarks.landmark), 0, image_height)
        y_min = np.clip(min(i.y * image_height for i in hand_landmarks.landmark), 0, image_height)
        
        return np.array([x_min, y_min, x_max, y_max]).astype(int)
     
        
