import numpy as np
import os
import Utils
import Morfology
import Cluster
import Features
import threading

def message(text):
    pass#print(text)

lock = threading.Lock()
def preprocess(mask,h,temp_dir,img,num_clusters,prefix):
    binary_img = None
    Utils.saveImg(os.path.join(temp_dir, "0_"+str(prefix)+"_img.png"), img)

    mask_reconstructed = Morfology.reconstructionAdaptative(mask, h)
    Utils.saveImg(os.path.join(temp_dir, "0_"+str(prefix)+"_mask_reconstructed.png"), mask_reconstructed)

    message("mask_reconstructed")

    markers = Morfology.seeds(img, mask_reconstructed)
    message("seeds")
    mean = Features.mean(img, markers)
    message("mean")

    if len(mean) > 0 and mean is not None:
        if (mean.shape[0] >= 2):
            groups = Cluster.cluster(mean, min(num_clusters, mean.shape[0]))
        else:
            groups = markers
        num_seeds = np.max(markers)

        nImgs = img.copy()
        for i in range(0, num_seeds):
            nImgs[markers == i] = [255 // (groups[i] + 1), 255 // (groups[i] + 1), 255 // (groups[i] + 1)]
        message("new_mask")

        # new_mask = Morfology.dilatate(new_mask, Utils.getKernel((3, 3)))
        message("Em " + str(h) + " há " + str(num_seeds) + " grupos")
        Utils.saveImg(os.path.join(temp_dir, str(prefix)+"_new_mask.png"), nImgs)

        message("nImgs")
        grayImage = Utils.to_gray(nImgs)
        binary_img = Utils.binarize_otsu(grayImage)
        print(binary_img.dtype)
        Utils.saveImg(os.path.join(temp_dir, "1_"+str(prefix-2)+"_binary_img.png"), binary_img)

    return binary_img
def pipeline(h,img,mask):
    print("Thread", h)
    global rImg
    global lock

    temp_dir = Utils.getDir(dir, str(h))
    binary_img = preprocess(mask,h,temp_dir,img,2,2)
    if binary_img is not None:

        Utils.saveImg(os.path.join(temp_dir, "2_binary_img.png"), binary_img)


        nImgs = Morfology.opening(binary_img, Utils.getKernel((3, 3)))
        nImgs = Morfology.close(nImgs, Utils.getKernel((3, 3)))
        Utils.saveImg(os.path.join(temp_dir, "3_nImgs.png"), nImgs)
        message("nImgs")

        nImgs_fill = Utils.multiFill(nImgs)
        Utils.saveImg(os.path.join(temp_dir, "5_img_fill.png"), nImgs_fill)

        nImgs_border = Utils.sub(nImgs, nImgs_fill)
        Utils.saveImg(os.path.join(temp_dir, "6_img_fill_xor.png"), nImgs_border)

        imgs_filtred = Utils.removeSmallComponents(nImgs_border)
        Utils.saveImg(os.path.join(temp_dir, "8_clear.png"), imgs_filtred)
        with lock:
            img_normalized = Utils.normalize(imgs_filtred)
            rImg = Utils.img_and(rImg, img_normalized)
            Utils.saveImg(os.path.join(temp_dir, "9_clear.png"), rImg * 255)

    print("Fim thread",h)

backgrounds = Utils.loadAll('/home/calazans/Documents/Images_pdi/imgs/img/')
classifier = Cluster.loadKNN()
has = 0

parameters = [2, 3]#, 5,6,7,8,9,10,11,12,13,14,15,16]
for img_path in backgrounds:
    print(img_path)
    has += 1
    #print(img_path)
    img = Utils.loadImg(img_path)
    result_image = img.copy()
    img_resize = Utils.resize(img)
#    img_resize, pos = Utils.mkborder(img_resize)

    mask = Utils.getMask(img_resize)
    #print(mask.shape)
  #  mask = Utils.cleanEdges(mask)
    dir = Utils.getDir("/home/calazans/Documents/Images_pdi/results/", os.path.splitext(os.path.basename(img_path))[0])

    #print(has,len(backgrounds),img_resize.shape)

    rImg = np.ones(img_resize.shape[:2])
    r_Img = np.zeros(img.shape)

    Utils.saveImg(os.path.join(dir, "mask.png"), mask)

    threads = []

    for h in parameters:
        thread = threading.Thread(target=pipeline, args=(h,img_resize,mask,))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    imgs_connected = Utils.getConnectedObjects(rImg)
    i = 0
    features = Features.all_features_batch(imgs_connected)
    if(features.shape[0] > 0):
        predicted = classifier.predict(features)
    else:
        predicted = False
    for img_connected in imgs_connected:
        i+=1
        doc_dir = Utils.getDir(dir,"docs")
        others_dir = Utils.getDir(dir,"others")
        h,w  = img.shape[:2]
        canvas = Features.findContours(img_connected)
        Utils.saveImg(os.path.join(dir, "finalIMsG.png"), canvas)
        finalImage = Utils.resize_fixed_size(canvas,(w,h) )
        if predicted[i-1] == 1:
            Utils.saveImg(os.path.join(doc_dir, "img_connected" + str(i) + ".png"), img_connected)
            Utils.saveImg(os.path.join(doc_dir, "finalIMG" + str(i) + ".png"), canvas)
        else:
            Utils.saveImg(os.path.join(doc_dir, "img_connected" + str(i) + ".png"), img_connected)
            Utils.saveImg(os.path.join(doc_dir, "finalIMG" + str(i) + ".png"), canvas)
        result_image = Utils.add(result_image,finalImage)
        Utils.saveImg(os.path.join(dir, "finalIMG.png"), result_image)

        nImgs_fill = Utils.sub(canvas,Utils.multiFill(canvas))
        Utils.saveImg(os.path.join(dir, "finaslIMG.png"), nImgs_fill)

        r_Img = Utils.add(r_Img, Utils.resize_fixed_size(nImgs_fill,(w,h) ))
    finalResult = os.path.join("/home/calazans/Documents/Images_pdi/results/",os.path.splitext(os.path.basename(img_path))[0] + ".png")
    Utils.saveImg(finalResult, r_Img)
