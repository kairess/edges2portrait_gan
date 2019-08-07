from skimage.io import imread
from skimage.transform import resize
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import os

class DataGenerator():
  def __init__(self):
    pass

  def process(self, batch_path, is_train):
    imgs_A, imgs_B = [], []

    for img_path in batch_path:
      img_A = imread(img_path, as_gray=True)
      img_B = imread(os.path.join('edges2portrait/trainB', os.path.basename(img_path)), as_gray=True)

      if is_train and np.random.random() < 0.5:
        img_A = np.fliplr(img_A)
        img_B = np.fliplr(img_B)

      imgs_A.append(np.expand_dims(img_A, axis=-1))
      imgs_B.append(np.expand_dims(img_B, axis=-1))

    imgs_A = np.array(imgs_A) / 127.5 - 1.
    imgs_B = np.array(imgs_B) / 127.5 - 1.

    return imgs_A, imgs_B

  def load_data(self, batch_size=1, is_train=True):
    listA = glob('edges2portrait/trainA/*.jpg')

    batch_path = np.random.choice(listA, size=batch_size)

    imgs_A, imgs_B = self.process(batch_path, is_train)

    return imgs_A, imgs_B

  def load_batch(self, batch_size=1, is_train=True):
    listA = glob('edges2portrait/trainA/*.jpg')

    self.n_batches = int(len(listA) / batch_size)

    for i in range(self.n_batches-1):
      batch_path = listA[i*batch_size:(i+1)*batch_size]
      
      imgs_A, imgs_B = self.process(batch_path, is_train)

      yield imgs_A, imgs_B

if __name__ == '__main__':
  dg = DataGenerator()
  a = dg.load_data(batch_size=3, is_train=True)

  print(a)