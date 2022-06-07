import cv2
from pathlib import Path
import os
import numpy as np

current_dir_path = os.path.dirname(os.path.realpath(__file__))
color_dir_path = os.path.join(current_dir_path, "color")
depth_dir_path = os.path.join(current_dir_path, "depth")

mean = []
stdev = []
imagesR = []
imagesG = []
imagesB = []
imagesD = []

for filename in os.listdir(color_dir_path):

    name = os.path.join(color_dir_path,filename)
    file = cv2.imread(name)

    name = os.path.join(depth_dir_path,filename)
    depth = cv2.imread(name,0)
    
    #ms = cv2.meanStdDev(file)
    #msd = cv2.meanStdDev(depth)
    
    #mean.append([*ms[0].tolist(),*msd[0].tolist()])
    #stdev.append([*ms[1].tolist(),*msd[1].tolist()])
    
    transposed = file.transpose(2,0,1).reshape(3,-1)
    #imagesR.append(transposed[2].astype("uint8"))
    #imagesG.append(transposed[1].astype("uint8"))
    #imagesB.append(transposed[0].astype("uint8"))
    imagesD.append(depth.astype("uint8"))

#pixelsR = np.array(imagesR).reshape(-1)
#pixelsG = np.array(imagesG).reshape(-1)
#pixelsB = np.array(imagesB).reshape(-1)
pixelsD = np.array(imagesD).reshape(-1)

#print("result mean:\nR: {}\nG: {}\nB: {}\nD: {}\n".format(np.mean(mean,axis = 0)[2][0],np.mean(mean,axis = 0)[1][0],np.mean(mean,axis = 0)[0][0],np.mean(mean,axis = 0)[3][0]))
#print("result std from means:\nR: {}\nG: {}\nB: {}\nD: {}\n".format(np.std(mean,axis = 0)[2][0],np.std(mean,axis = 0)[1][0],np.std(mean,axis = 0)[0][0],np.std(mean,axis = 0)[3][0]))
#print("result mean from stds:\nR: {}\nG: {}\nB: {}\nD: {}\n".format(np.mean(stdev,axis = 0)[2][0],np.mean(stdev,axis = 0)[1][0],np.mean(stdev,axis = 0)[0][0],np.mean(stdev,axis = 0)[3][0]))
#print("result std:\nR: {}\nG: {}\nB: {}\nD: {}\n".format(np.std(pixelsR),np.std(pixelsG),np.std(pixelsB),np.std(pixelsD)))
print("result std:\n{}".format(np.std(pixelsD)))

