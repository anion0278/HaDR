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
    imgs = []
    for j in range(len(images)-1):
       imgs.append(cv2.copyMakeBorder(cv2.imread(images[j]),0,0,0,spacing,cv2.BORDER_CONSTANT,value=[255,255,255]))
    imgs.append(cv2.imread(images[j+1]))
    cv2.imwrite(os.path.join(source_dir,"result.png"),cv2.hconcat(imgs))    

if __name__ == "__main__":
    source_dir = "E:/datasets/EVALmedia"
    image_names = load_images(source_dir)
    save_result(image_names)