from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv

config_file = '../configs/solov2/solov2_light_448_r50_fpn_custom.py'
# download the checkpoint from model zoo and put it in `checkpoints/`
# checkpoint_file = '../checkpoints/DECOUPLED_SOLO_R50_3x.pth'
checkpoint_file = '../checkpoints/epoch_4.pth'
# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')


from mmdet.apis import inference_detector, show_result_ins
model.CLASSES = ["hand", "dummy_class"]

def detect(img_path):
    result = inference_detector(model, img_path)

    out_path = os.path.abspath('../..') + "/demo_hand_out.jpg"

    show_result_ins(img, result, model.CLASSES, score_thr=0.25, out_file=out_path)
    from IPython.display import display
    from PIL import Image   

    display(Image.open(out_path))

# img = "G:/datasety/new/color/0000035_gest1_X-138.2_Y239.3_Z380.8_hand1_date01-03-2022_08#56#08.png"
# detect(img)


for f in ["312", "520", "521", "4", "5", "282", "372", "497"]:
    import os
    img = os.path.abspath('../../datasets/real_cam/2_ruce_rukavice_250/color') + "/" + f+".png"
    detect(img)
