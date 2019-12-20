import numpy as np
import Utils
def verify(value,x):
    a = x
    res = a + 0.1
    return value > res
def calculateDensityInBoarders(image):
    temp_image = Utils.normalize(image)

    rValue = False

    sum = np.sum(temp_image,axis=0)
    sum_1 = np.sum(temp_image,axis=1)
    max_density = np.sum(sum,axis=0)

    sum = sum / max_density
    sum_1 = sum_1 / max_density

    x = sum.shape[0]
    y = sum_1.shape[0]

    density_x = 0
    density_y = 0

    for i in range(0,x//2):
        density_x += sum[i]
        if verify(density_x,i/x):
            rValue = True
            break
    if rValue:
        for j in range(0,y//2):
            density_y += sum_1[j]
            if verify(density_y, j/y):
                rValue = True
                break
    else:
        density_x = 0
        for i in range(x // 2,x):
            density_x += sum[x-i]
            if verify(density_x, i / x):
                rValue = True
                break
        if rValue:
            for j in range(y // 2,y):
                density_y += sum_1[y - j]
                if verify(density_y, j / y):
                    rValue = True
                    break
   # print(type(density_x),type(density_y))
   # print(rValue)
   # exit()

    return rValue

def calculateDensity(image):
    temp_image = Utils.normalize(image)
    x,y = temp_image.shape[:2]
    sum = np.sum(temp_image,axis=1)
    sum = np.sum(sum,axis=0)
    sum = sum /(x*y)
    return sum
