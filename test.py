import numpy as np
import os
import Utils
import Morfology
import Cluster

backgrounds = Utils.loadAll('/home/calazans/Downloads/temp/0/')

has = 0

for img_path in backgrounds:
    has += 1
    img = Utils.loadImg(img_path)

    print("mask")
    mask = Utils.getMask(img)

    dir = os.path.join("/home/calazans/Documents/lista 08/", os.path.splitext(os.path.basename(img_path))[0])

    if not os.path.exists(dir):
        os.makedirs(dir)

    print(has,len(backgrounds),img.shape)

    Utils.saveImg(os.path.join(dir,"mask.png"),mask)
    for h in range(3,14):
        mask_reconstructed = Morfology.reconstructionAdaptative(mask,h)

        markers = Morfology.seeds(img,mask_reconstructed )
        mean = Cluster.feature_extract(img,markers)

        grups = Cluster.cluster(mean,2)
        img_dil = Morfology.dilatate(img,Utils.getKernel((3,3)))

        new_mask = img_dil.copy()

        num_seeds = np.max(markers)
        for i in range(0,num_seeds):
            new_mask[markers == i] *= grups[i]

        mask2 = Utils.getMask(new_mask)
        mask2_reconstructed = Morfology.reconstructionAdaptative(mask2,h)
        markers = Morfology.seeds(img,mask2_reconstructed)

        nImgs = Utils.makeMarkers(img, markers, num_seeds)
        Utils.saveImg(os.path.join(dir,str(h)+"_img"+".png"),nImgs)