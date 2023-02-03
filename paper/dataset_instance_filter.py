from os import listdir
from os.path import join,exists,splitext
import cv2
from shutil import move

dataset_dir_path = "C:/datasets/"
number_of_instances = 3

color_dir_path = join(dataset_dir_path, "color")
mask_dir_path = join(dataset_dir_path, "mask2")
depth_dir_path = join(dataset_dir_path, "depth")
dest_dir = join(dataset_dir_path, "two_instances")
dest_color = join(dest_dir, "color")
dest_mask = join(dest_dir, "mask2")
dest_depth = join(dest_dir, "depth")

for filename in listdir(color_dir_path):

    mask = join(mask_dir_path,filename)
    instance_name = splitext(filename)[0]
    
    files = []
    instances = 0
    if(exists(mask)):
        files.append(cv2.imread(mask,0))
    else:
        for i in range(1,number_of_instances,1):
            files.append(cv2.imread(join(mask_dir_path,instance_name + "_i"+str(i)+".png"),0))
            
    for file in files:
        M = cv2.moments(file)
        if (M["m00"]!=0):
            cx=int(M["m10"]/M["m00"])
            cy=int(M["m01"]/M["m00"])
            instances +=1
        
    if (instances==2):
       move(join(color_dir_path,instance_name+".png"),join(dest_color,instance_name+".png")) 
       move(join(depth_dir_path,instance_name+".png"),join(dest_depth,instance_name+".png"))
       for i in range(1,number_of_instances,1):
        move(join(mask_dir_path,instance_name + "_i"+str(i)+".png"),join(dest_mask,instance_name + "_i"+str(i)+".png")) 

