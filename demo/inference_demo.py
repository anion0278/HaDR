from mmdet.apis import init_detector, inference_detector, show_result_pyplot
# config_file = '../configs/mask_rcnn_r101_fpn_custom.py'
config_file = '../SOLO/configs/solov2/solov2_r101_fpn_custom.py'
# download the checkpoint from model zoo and put it in `checkpoints/`
# checkpoint_file = '../checkpoints/DECOUPLED_SOLO_R50_3x.pth'
checkpoint_file = '../SOLO/checkpoints/epoch_6.pth'
# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')
print("test")


from mmdet.apis import inference_detector, show_result_ins
import cv2
model.CLASSES = ["hand"]

def detect(img_path):
    import time
    start = time.time()
    result = inference_detector(model, img_path)
    text = "Inference time: %.2f s" % (time.time() - start)
    img_res = show_result_ins(img, result, model.CLASSES, score_thr=0.3)
    cv2.imshow(text, img_res)
    cv2.waitKey(0)

img = r"C:\Users\Stefan\source\repos\HGR_CNN\datasets\rgbd_joined_dataset\ruka_2/color/0000022_gest1_X-182.6_Y189.1_Z386.6_hand1_date27-02-2022_20#27#32.png"
detect(img)


for f in ["425", "399", "292", "315", "11_140", "13_150", "7_240", "5_209", "497", "10_124","458", "302",  "9_152",  "1", "2", "3", "4", "5", "6",  "282", "372", ]:
# for f in ["12_131", "7_240", "8_196", "9_152", "12_131"]:
    import os
    img = os.path.abspath('../../HGR_CNN/datasets/real_cam/2_ruce_rukavice_250/color') + "/" + f+".png"
    detect(img)
