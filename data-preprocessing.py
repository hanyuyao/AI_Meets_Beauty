import os
import numpy as np
import random
import cv2

def delete(path):
    # print(os.path.getsize(path))
    # delete files whose size is smaller than 2000 bytes
    if os.path.getsize(path) < 2000:
        os.remove(path)


def delete_file():
    for dirpath, dirnames, filenames in os.walk('./data_10000'):
        for x in filenames:
            path = os.path.join(dirpath, x)
            delete(path)


def sp_noise(image,prob):
    # add salt and pepper noise to image
    # add to pixel whose rgb value >= (250, 250, 250)
    # prob: Probability of the noise
    white = np.array([250, 250, 250])
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob 
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if (image[i][j] >= white).all():
                for k in range(image.shape[2]):
                    rdn = random.random()
                    if rdn < prob:
                        output[i][j][k] = 0
                    elif rdn > thres:
                        output[i][j][k] = 255
                    else:
                        output[i][j][k] = image[i][j][k]
            else:
                output[i][j] = image[i][j]
    return output


def process_file():
    base_path = './data_10000_processed'
    for dirpath, dirnames, filenames in os.walk('./data_10000'):
        for x in filenames:
            source_path = os.path.join(dirpath, x)
            dest_path = os.path.join(base_path, x)
            img = cv2.imread(source_path)
            img = cv2.resize(img, (224,224), interpolation=cv2.INTER_LINEAR)
            noise_img = sp_noise(img,0.1)
            cv2.imwrite(dest_path, noise_img)

if __name__ == '__main__':
    # delete_file()
    # process_file()
    print('preprocessing...')