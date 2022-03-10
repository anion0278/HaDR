from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv
config_file = '../configs/solov2/solov2_light_448_r50_fpn_8gpu_3x.py'
# download the checkpoint from model zoo and put it in `checkpoints/`
# checkpoint_file = '../checkpoints/DECOUPLED_SOLO_R50_3x.pth'
checkpoint_file = '../checkpoints/epoch_1.pth'
# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')
# test a single image
import os
# img = os.path.abspath('../../HGR_CNN/datasets/real_cam/2_ruce_rukavice_250/RGB') + "/1.png"
img = os.path.abspath('../../HGR_CNN/datasets/rgbd_joined_dataset/ruka_2/color') + "/0000001_gest1_X429.4_Y295.1_Z301.2_hand3_date27-02-2022_20#04#11.png"
model.CLASSES = ["hand", "dummy_class"]
model.cfg.score_thr = 0.0
result = inference_detector(model, img)
print(result)

from mmdet.apis import init_detector, inference_detector, show_result_pyplot, show_result_ins
out_path = os.path.abspath('../../HGR_CNN') + "/demo_hand_out.jpg"

show_result_ins(img, result, model.CLASSES, score_thr=0.25, out_file=out_path)
result = [[
    tuple(t.cpu().data.numpy() for t in result[0][0]),
    tuple(t.cpu().data.numpy() for t in result[0][1]),
    tuple(t.cpu().data.numpy() for t in result[0][2]),
    ]]
show_result_pyplot(img, result, model.CLASSES)