import csv
import numpy as np
import urllib.request

def read_csv():
    train_csv_dir = './training_public.csv'

    train_label = []
    with open(train_csv_dir, 'r', encoding="utf-8") as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        next(csv_reader)
        for line in csv_reader:
            train_label.append([line[0], line[1], line[2]])
    train_label = np.array(train_label)         # train_label.shape = (540506, 3)

    np.save('training.npy', train_label)


def download_file():
    f = open("fail.txt", "a")

    x = np.load('training.npy')
    for line in x:
        path = './data/' + line[0] + '.jpg'
        url = line[1]
        try:
            urllib.request.urlretrieve(url, path)
        except:
            f.write(line[0] + ' failed...')
    
    f.close()


if __name__ == '__main__':
    download_file()