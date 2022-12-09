import cv2
import os,random
import matplotlib.pyplot as plt
import numpy as np
import mpl_scatter_density
from ast import literal_eval

#plt.style.use('_mpl-gallery')
plt.rcParams.update({'font.size': 18})

dataset_dir_path = "C:/dataset/sim_val_320x256"
number_of_instances = 3

centroids_file = os.path.join(dataset_dir_path,"centroid.txt")
color_dir_path = os.path.join(dataset_dir_path, "color")
mask_dir_path = os.path.join(dataset_dir_path, "mask2")

def main():
    if(os.path.exists(centroids_file)):
        x,y,dims,counts = read_centroids()
    else:
        x,y,dims,counts = find_centroids()
    show_graph(x,y,dims,counts)


def find_centroids():
    print("Processing dataset to find centroids")
    centroids = []
    counts = [0]*number_of_instances

    name = random.choice(os.listdir(mask_dir_path))
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
            for i in range(1,number_of_instances,1):
                files.append(cv2.imread(os.path.join(mask_dir_path,instance_name + "_i"+str(i)+".png"),0))
            
        for file in files:
            M = cv2.moments(file)
            if (M["m00"]!=0):
                cx=int(M["m10"]/M["m00"])
                cy=int(M["m01"]/M["m00"])
                centroids.append((cx,cy))
                instances +=1
                #cv2.circle(file, (cx, cy), 10, (255, 255, 255), -1)
                #cv2.imshow("win"+str(len(counts)),file)
        
        counts[instances]+=1
        if(len(centroids)%10000):
            print("{} files processed".format(str(len(centroids))))

    f = open(centroids_file,'w')
    f.write(str(dims)+"\n")
    f.write(str(counts)+"\n")
    for c in centroids:
        f.write(str(c)+"\n")
    f.close()
    x,y = zip(*centroids)
    return x,y,dims,counts

def read_centroids():
    print("Reading centroids from file")
    f = open(centroids_file,'r')
    dims_s = f.readline()
    dims = literal_eval(dims_s)
    counts_s = f.readline()
    counts = literal_eval(counts_s)
    centroids = []
    for line in f:
        centroids.append(literal_eval(line))

    x,y = zip(*centroids)
    return x,y,dims,counts

def show_graph(x,y,dims,counts):
    fig = plt.figure()
    gs = fig.add_gridspec(1,2,wspace=0.2)
    gs1=gs[0].subgridspec(1,1)
    gs2=gs[1].subgridspec(2,2,height_ratios=[1,8],width_ratios=[8,1],wspace=0.12,hspace=0.06)
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.9)

 
    bar = fig.add_subplot(gs1[0,0])
    bar.bar(list(range(number_of_instances)),counts,color = 'blue')
    bar.grid(visible=None)
    bar.set_xticks(list(range(number_of_instances)))

    bar.set_xlabel("number of instances",labelpad = 10)
    bar.set_ylabel("number of images",labelpad = 10)

    scatter = fig.add_subplot(gs2[1,0],projection='scatter_density')
    scatter.grid(visible=None)
    #scatter.hexbin(x,y,gridsize=(dims[1],dims[0]))
    density = scatter.scatter_density(x,y, color = 'blue',dpi=10)
    scatter.set_xticks(np.arange(0,dims[1]+32,dims[1]/5))
    scatter.set_yticks(np.arange(0,dims[0]+32,dims[1]/5))
    scatter.set_xlabel("x axis (pixels)",labelpad = 10)
    scatter.set_ylabel("y axis (pixels)",labelpad = 10)

    

    y_hist = fig.add_subplot(gs2[1,1], sharey = scatter)
    hy,_,_=y_hist.hist(y,np.arange(0,dims[0]+1,1), orientation="horizontal",color = 'blue')
    max_hist = np.round(max(hy),-2)+50
    y_hist.set_xticks([0,max_hist])
    y_hist.grid(visible=None)
    y_hist.tick_params(axis="y", labelleft=False)

    x_hist = fig.add_subplot(gs2[0,0], sharex = scatter)
    hy,_,_=x_hist.hist(x,np.arange(0,dims[1]+1,1),color = 'blue')
    max_hist = np.round(max(hy),-2)+50
    x_hist.set_yticks([0,max_hist])
    x_hist.tick_params(axis="x", labelbottom=False)
    x_hist.grid(visible=None)
    plt.show()


if __name__ == "__main__":
    main()
