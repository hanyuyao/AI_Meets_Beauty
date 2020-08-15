import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# features
db = np.load('./features/features_10000.npy')
target = np.load('./features/features_test.npy')

# a list of file names
db_list = np.load('./features/list_10000.npy')
target_list = np.load('./features/list_test.npy')

db_list.shape   # (10000, )
db.shape    # (10000, 181888)

pred = cosine_similarity(target, db)

pred.shape  # (#of input images, 10000)

pred = np.argmax(pred, axis=1)

for i in range(target_list.shape[0]):
    test_img_name = target_list[i]
    result_img_name = db_list[pred[i]]
    test_img = os.path.join('./data_test', test_img_name)
    result_img = os.path.join('./data_10000_processed', result_img_name)

    plt.subplot(1, 2, 1)
    img = mpimg.imread(test_img)
    plt.title(test_img_name)
    plt.imshow(img)

    plt.subplot(1, 2, 2)
    img = mpimg.imread(result_img)
    plt.title(result_img_name)
    plt.imshow(img)

    # plt.show()
    plt.savefig('./result/{}.jpg'.format(test_img_name))
