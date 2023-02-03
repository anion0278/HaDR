import cv2
import os

spacing = 4

def load_images(folder):
    filelist = [file for file in os.listdir(folder) if file.endswith('.png')]
    images = []
    for file in filelist:
        im_path = os.path.join(folder,file)
        images.append(im_path)
    
    return images

def save_result(images):
    names = []
    results = []
    for image in images:
        name = image.split("-")[-1]
        if not name in names:
            names.append(name)
            results.append([])
            results[len(names)-1].append(image)
        else:
            results[names.index(name)].append(image)

    
    for j in range(len(results[0])):
        images = []
        for i in range(len(results)-1):
            images.append(cv2.copyMakeBorder(cv2.imread(results[i][j]),0,spacing,0,0,cv2.BORDER_CONSTANT,value=[255,255,255]))
        images.append(cv2.imread(results[i+1][j]))
        name = os.path.basename(results[i][j]).split("-")
        cv2.imwrite(os.path.join(source_dir,"result",name[-3]+"-"+name[-2]+".png"),cv2.vconcat(images))    

    
if __name__ == "__main__":
    source_dir = "E:/datasets/qualitative_test/Threshold0.01"
    res_dir = os.path.join(source_dir,"result")
    image_names = load_images(source_dir)
    os.mkdir(res_dir)
    save_result(image_names)