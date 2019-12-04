import cv2
import numpy as np
import os
import random

bkground_folder = '/home/calazans/Downloads/Background/'
bkground = []
for r, d, f in os.walk(bkground_folder):
	for file in f:
		bkground.append(os.path.join(r, file))

imgs_folder = '/home/calazans/Downloads/Documentos/'
imgs = []
for r, d, f in os.walk(imgs_folder):
	for file in f:
		imgs.append(os.path.join(r, file))
res_folder = '/home/calazans/Downloads/temp/'

def resize(img,w,h,prop):
    height, width = img.shape[:2]
    ex = max(min(w/width,h/height)*prop,0.5)
    res = cv2.resize(img,(round(ex*width), round(ex*height)), interpolation = cv2.INTER_CUBIC)
    return res

def to_RGBA(img):
    if (len(img.shape) == 2):
        b_channel =r_channel=g_channel= img
    elif(img.shape[2] == 3):
        b_channel, g_channel, r_channel = cv2.split(img)
    else:
        return img
    alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 255 #creating a dummy alpha channel image.
    return cv2.merge((b_channel, g_channel, r_channel, alpha_channel))

def rotate_bound(image, angle):
	(h, w) = image.shape[:2]
	(cX, cY) = (w / 2, h / 2)

	M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
	cos = np.abs(M[0, 0])
	sin = np.abs(M[0, 1])

	# compute the new bounding dimensions of the image
	nW = int((h * sin) + (w * cos))
	nH = int((h * cos) + (w * sin))

	# adjust the rotation matrix to take into account translation
	M[0, 2] += (nW / 2) - cX
	M[1, 2] += (nH / 2) - cY

	# perform the actual rotation and return the image
	return cv2.warpAffine(image, M, (nW, nH), borderValue=(0,0,0,0))

def add(imageA,imageB, pos):
    (x,y) = pos


    (h, w) = imageB.shape[:2]
    (Ay, Ax) = imageA.shape[:2]
    imageC = np.ones(imageA.shape[:3])*255
    ImageD = np.zeros(imageB.shape)
    maxX= min(Ax-x,w)
    maxY= min(Ay-y,h)
    test = maxX == Ax-x and maxY == Ay-y

    alpha_s = imageB[:maxY, :maxX, 3] / 255.0
    alpha_l = 1.0 - alpha_s

    for c in range(0, 3):
        imageC[y:maxY+y, x:maxX+x,  c] = alpha_s * ImageD[:maxY, :maxX, c] + alpha_l*imageC[y:maxY+y, x:maxX+x, c]
        imageA[y:maxY+y, x:maxX+x,  c] = alpha_s * imageB[:maxY, :maxX, c] + alpha_l*imageA[y:maxY+y, x:maxX+x, c]
    return imageA,imageC, test
i =0
total = len(imgs)*len(bkground)
for img_path in imgs:
    img = cv2.imread(img_path)
    for back in bkground:
        print(i,total)
        i+=1
        bk = cv2.imread(back)
        (h, w) = bk.shape[:2]
        img = rotate_bound(to_RGBA(img), random.random()*360)
        border = max(w,h)*0.1
        img = resize(img, w-border, h-border, max(min(random.random(),0.5),1.0))
        (h2, w2) = img.shape[:2]
        pos = (max(round(random.random()*(w - w2)),1),max(round(random.random()*(h- h2)),1))
        img2,gt,test = add(bk,img,pos)
        if(not test):
            act = os.path.join(res_folder,str(i//100))
        else:
            act = os.path.join(os.path.join(res_folder,"erro"),str(i//10))
        if not os.path.exists(act):
            os.makedirs(act)
        thres, gt = cv2.threshold(gt, 127, 255, cv2.THRESH_BINARY_INV)
        cv2.imwrite(os.path.join(act,str(i)+'.png'),to_RGBA(img2))
        cv2.imwrite(os.path.join(act,str(i)+'_gt.png'),to_RGBA(gt))
