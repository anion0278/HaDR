import cv2
import os,random
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('_mpl-gallery')
plt.rcParams.update({'font.size': 18})

dataset_dir_path = "C:/datasets/egohands_data"
color_dir_path = os.path.join(dataset_dir_path, "color")
mask_dir_path = os.path.join(dataset_dir_path, "mask2")

centroids = []
#counts = [0,0,0]
#egohands
counts = [0,0,0,0,0]

name =random.choice(os.listdir(mask_dir_path))
file = cv2.imread(os.path.join(mask_dir_path,name))
dims = file.shape

for filename in os.listdir(color_dir_path):

    name = os.path.join(color_dir_path,filename)
    mask = os.path.join(mask_dir_path,filename)
    instance_name = os.path.splitext(filename)[0]
   
    files = []
    instances = 0
    if(os.path.exists(mask)):
        files.append(cv2.imread(mask,0))
    else:
        files.append(cv2.imread(os.path.join(mask_dir_path,instance_name + "_i1.png"),0))
        files.append(cv2.imread(os.path.join(mask_dir_path,instance_name + "_i2.png"),0))
        #egohands
        files.append(cv2.imread(os.path.join(mask_dir_path,instance_name + "_i3.png"),0))
        files.append(cv2.imread(os.path.join(mask_dir_path,instance_name + "_i4.png"),0))

    for file in files:
        M = cv2.moments(file)
        if (M["m00"]!=0):
            centroids.append((int(M["m10"]/M["m00"]),int(M["m01"]/M["m00"])))
            instances +=1
    
    counts[instances]+=1

fig = plt.figure()
gs = fig.add_gridspec(1,2,wspace=0.2)
gs1=gs[0].subgridspec(1,1)
gs2=gs[1].subgridspec(2,2,height_ratios=[1,8],width_ratios=[8,1],wspace=0.06,hspace=0.06)
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.9)
x,y=zip(*centroids)

bar = fig.add_subplot(gs1[0,0])
#bar.bar([0,1,2],counts)
#egohands
bar.bar([0,1,2,3,4],counts)
bar.grid(visible=None)
#bar.set_xticks([0,1,2])
#egohands
bar.set_xticks([0,1,2,3,4])
bar.set_xlabel("number of instances",labelpad = 10)
bar.set_ylabel("number of images",labelpad = 10)

scatter = fig.add_subplot(gs2[1,0])
scatter.grid(visible=None)
scatter.hexbin(x,y)
scatter.set_xticks(np.arange(0,dims[1]+32,dims[1]/5))
scatter.set_yticks(np.arange(0,dims[0]+32,dims[1]/5))
scatter.set_xlabel("x axis (pixels)",labelpad = 10)
scatter.set_ylabel("y axis (pixels)",labelpad = 10)

y_hist = fig.add_subplot(gs2[1,1], sharey = scatter)
y_hist.hist(y,np.arange(0,dims[0]+1,1), orientation="horizontal")
y_hist.set_xticks(np.arange(0,250,200))
y_hist.grid(visible=None)
y_hist.tick_params(axis="y", labelleft=False)

x_hist = fig.add_subplot(gs2[0,0], sharex = scatter)
x_hist.hist(x,np.arange(0,dims[1]+1,1))
x_hist.set_yticks(np.arange(0,250,200))
x_hist.tick_params(axis="x", labelbottom=False)
x_hist.grid(visible=None)
plt.show()
