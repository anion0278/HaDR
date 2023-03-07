import cv2
import os
import numpy as np
import shutil

import ws_specific_settings as wss
import model_utils as utils
import image_loader

import common_settings as s
s.add_packages_paths()

from mmdet.apis import init_detector, show_result_ins, predict_image, show_result_pyplot

#threshold = 0.01
tested_models={
"2X-solov2_r101_fpn_1ch-CocoPretrained=True-DS=sim_train_320x256_full-Aug=False-BS=4-BNfixed=False-FrozenEP=0+LR=0.01-UnfrozenEP=20_+LR=0.01-LRConfig=step[7-18]-Tue_20_Dec_17h_37m":0.55,
"1Y-solov2_light_448_r50_fpn_3ch-CocoPretrained=True-DS=sim_train_320x256_full-Aug=False-BS=4-BNfixed=False-FrozenEP=0+LR=0.01-UnfrozenEP=20_+LR=0.01-LRConfig=step[7-18]-Thu_22_Dec_13h_45m":0.6,
"2Z-solov2_r101_fpn_4ch-CocoPretrained=True-DS=sim_train_320x256_full-Aug=False-BS=4-BNfixed=False-FrozenEP=0+LR=0.01-UnfrozenEP=20_+LR=0.01-LRConfig=step[7-18]-Sun_25_Dec_22h_59m":0.5,
"4X-mask_rcnn_r101_fpn_1ch-CocoPretrained=True-DS=sim_train_320x256_full-Aug=False-BS=4-BNfixed=False-FrozenEP=0+LR=0.01-UnfrozenEP=20_+LR=0.01-LRConfig=step[7-18]-Thu_05_Jan_01h_16m":0.825,
"3Y-mask_rcnn_r50_fpn_3ch-CocoPretrained=True-DS=sim_train_320x256_full-Aug=False-BS=4-BNfixed=False-FrozenEP=0+LR=0.01-UnfrozenEP=20_+LR=0.01-LRConfig=step[7-18]-Fri_23_Dec_20h_55m":0.975,
"3Z-mask_rcnn_r50_fpn_4ch-CocoPretrained=True-DS=sim_train_320x256_full-Aug=False-BS=4-BNfixed=False-FrozenEP=0+LR=0.01-UnfrozenEP=20_+LR=0.01-LRConfig=step[7-18]-Mon_26_Dec_18h_41m":0.9
}

dataset_dir = os.path.join(s.path_to_datasets,"qualitative_test")
gt_dir = os.path.join(dataset_dir,"mask")
height, width = 480,640

def get_tested_image(input_channels, img_bgrd):
    options = { 1: img_bgrd[:,:,3:4], 
                3: img_bgrd[:,:,0:3],
                4: img_bgrd }
    return options[input_channels]

def detect(img_bgrd, arch):
    tested_image = get_tested_image(cfg.model.backbone.in_channels, img_bgrd)
    result = predict_image(model, tested_image)
    #result = np.zeros((height,width,1), np.uint8)
    image = np.zeros((height,width,3), np.uint8)
    ins_visualization = show_result_pyplot if "mask" in arch else show_result_ins # mask rcnn requires different visualization
    res_img_bgr = ins_visualization(image, result, model.CLASSES, score_thr=threshold, show_bbox=False)
    #res_img_bgr = result
    _,binary_mask = cv2.threshold(res_img_bgr,1,255,cv2.THRESH_BINARY)
    return binary_mask

def compare_single_image(img_prediction,filename):
    label_filename = os.path.join(gt_dir,filename)
    label = cv2.imread(label_filename,0)
    #img_prediction = cv2.cvtColor(img_prediction,cv2.COLOR_GRAY2BGR)
    FP = 0
    TP = 0
    FN = 0
    for i in range (img_prediction.shape[0]):
        for j in range (img_prediction.shape[1]):
            if img_prediction[i,j][0]>label[i,j]:
                img_prediction[i,j]=(0,0,255)
                FP = FP + 1
            elif img_prediction[i,j][0]<label[i,j]:
                img_prediction[i,j]=(255,0,0)
                FN = FN + 1
            elif img_prediction[i,j][0]==255:
                img_prediction[i,j]=(255,255,255)
                TP = TP + 1
    
    return img_prediction,[str(TP),str(FP),str(FN)]

if __name__ == "__main__":
    test_dest = os.path.join(dataset_dir,"test")
    os.mkdir(test_dest)

    for model_dir,threshold in tested_models.items():
               
        checkpoint_path_full = os.path.join(s.path_to_models,"FINAL TRAIN - ours",model_dir,s.tested_checkpoint_file_name)
        arch, channels = utils.parse_config_and_channels_from_checkpoint_path(model_dir)
        cfg = utils.get_config(arch, channels)
        model = init_detector(cfg, checkpoint_path_full, device='cuda:0')
        if len(model.CLASSES) > 1: 
            print(f"Overrinding the model classes! Current classes: {len(model.CLASSES)}")
            model.CLASSES = ["person"] if cfg.model.bbox_head.num_classes == 81 else ["hand"]
    
        

        log_filename = os.path.join(test_dest,str(channels)+"ch-"+arch+"-"+"log.txt")
        f = open(log_filename,"w")    
        stats = []
        log = []
    
        files_count = len([entry for entry in os.listdir(gt_dir) if os.path.isfile(os.path.join(gt_dir, entry))])

        loader = image_loader.ImageLoader(dataset_dir)
        for i in range(files_count):
            img = detect(loader.get_rgbd_image(), arch)
            filename = loader.get_current_image_name()
            img_prediction,stats=compare_single_image(img,filename)
            stats.insert(0,filename)
            output_filename = os.path.join(test_dest,str(channels)+"ch-"+arch+"-"+filename.split("-")[-1])
            log.append(stats)
            cv2.imwrite(output_filename,img_prediction)
            if (not os.path.exists(os.path.join(test_dest,"color-"+filename))):
                shutil.copy(os.path.join(dataset_dir,"color",filename),os.path.join(test_dest,"color-"+filename))
                shutil.copy(os.path.join(dataset_dir,"depth",filename),os.path.join(test_dest,"depth-"+filename))
    
        for l in log:
            f.write("{} TP: {},FP: {}, FN: {}\n".format(l[0],l[1],l[2],l[3]))
        f.close()
           
