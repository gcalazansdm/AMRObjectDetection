import numpy as np
import os
import Utils
import Morfology
import Cluster
import Features
import threading

def message(text):
    print(text)

lock = threading.Lock()

def pipeline(h,img,mask):
    print("Thread", h)
    global rImg
    global lock
    temp_dir = Utils.getDir(dir, str(h))
    mask_reconstructed = Morfology.reconstructionAdaptative(mask, h)
    message("mask_reconstructed")

    markers = Morfology.seeds(img, mask_reconstructed)
    mean = Features.mean(img, markers)
    message("mean")

    if len(mean) > 0 and mean is not None:
        groups = Cluster.cluster(mean, min(5, mean.shape[0]))
        img_dil = Morfology.dilatate(img, Utils.getKernel((3, 3)))
        message("dil")

        new_mask = img_dil.copy()

        num_seeds = np.max(markers)
        for i in range(0, num_seeds):
            new_mask[markers == i] *= groups[i]
        message("new_mask")

        mask2 = Utils.getMask(new_mask)
        mask2_reconstructed = Morfology.reconstructionAdaptative(mask2, h)
        markers = Morfology.seeds(img, mask2_reconstructed)
        message("markers")

        message("Em " + str(h)+ " há "+str(num_seeds)+ " sementes")
        nImgs = Utils.makeMarkers(img, markers, num_seeds)
        Utils.saveImg(os.path.join(temp_dir, "1_binary_img.png"), nImgs)
        message("nImgs")
        binary_img = Utils.binarize(nImgs)
        Utils.saveImg(os.path.join(temp_dir, "2_binary_img.png"), binary_img)
        message("binary_img")
        if True:#Features.density(binary_img) < 0.7:
            nImgs = Morfology.opening(binary_img, Utils.getKernel((3, 3)))
            Utils.saveImg(os.path.join(temp_dir, "3_nImgs.png"), nImgs)
            message("nImgs")

            nImgs_border, pos = Utils.mkborder(nImgs)
            Utils.saveImg(os.path.join(temp_dir, "4_nImgs_border.png"), nImgs_border)

            nImgs_fill = Utils.fill(nImgs_border)
            Utils.saveImg(os.path.join(temp_dir, "5_img_fill.png"), nImgs_fill)

            nImgs_border = Utils.sub(nImgs_border, nImgs_fill)
            Utils.saveImg(os.path.join(temp_dir, "6_img_fill_xor.png"), nImgs_border)

            nImgs_crop = Utils.crop(nImgs_border, pos)
            Utils.saveImg(os.path.join(temp_dir, "7_nImgs_crop.png"), nImgs_crop)

            imgs_filtred = Utils.removeSmallComponents(nImgs_crop)
            Utils.saveImg(os.path.join(temp_dir, "8_clear.png"), imgs_filtred)
            with lock:
                rImg = Utils.img_and(rImg, imgs_filtred)
                Utils.saveImg(os.path.join(temp_dir, "9_clear.png"), rImg)
    print("Fim thread",h)

backgrounds = Utils.loadAll('/home/calazans/Downloads/lol/imgs')
classifier = Cluster.loadKNN()
has = 0

parameters = [2,3, 4, 5,6,7,8,9,10,11,12,13,14,15,16]
for img_path in backgrounds:
    has += 1

    img = Utils.loadImg(img_path)

    img_resize = Utils.resize(img)
#    img_resize, pos = Utils.mkborder(img_resize)

    mask = Utils.getMask(img_resize)

    dir = Utils.getDir("/home/calazans/Downloads/lol/results", os.path.splitext(os.path.basename(img_path))[0])

    print(has,len(backgrounds),img_resize.shape)

    rImg = np.ones(img_resize.shape[:2])

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
    predicted = classifier.predict(features)
    for img_connected in imgs_connected:
        i+=1
        doc_dir = Utils.getDir(dir,"docs")
        others_dir = Utils.getDir(dir,"others")

        if predicted[i-1] == 1:
            Utils.saveImg(os.path.join(doc_dir, "img_connected" + str(i) + ".png"), img_connected)
            canvas = Features.findContours(img_connected)
            Utils.saveImg(os.path.join(doc_dir, "finalIMG" + str(i) + ".png"), canvas)
        else:
            Utils.saveImg(os.path.join(doc_dir, "img_connected" + str(i) + ".png"), img_connected)
            canvas = Features.findContours(img_connected)
            Utils.saveImg(os.path.join(doc_dir, "finalIMG" + str(i) + ".png"), canvas)

