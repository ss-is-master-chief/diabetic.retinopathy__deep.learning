import os
import shutil
import glob
import argparse
from tqdm import tqdm

import matplotlib as mpl
mpl.use('TkAgg')

import cv2
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, default=".", help='/path/to/images/')
parser.add_argument('--type', type=str, default="png", help="png, jpg, etc.")
parser.add_argument('--out', type=str, default=".", help="/path/to/images/")

FLAGS = parser.parse_args()
BASE_DIR = os.path.abspath(FLAGS.dir)
IMG_EXT = FLAGS.type
OUT_DIR = os.path.abspath(FLAGS.out)

try:
  print("Removing legacy directory...")
  shutil.rmtree(OUT_DIR+"/normalized_images")
  print("Normalized Images in directory: ", OUT_DIR+"/normalized_images")
  os.mkdir(OUT_DIR+"/normalized_images")
except:
  print("Normalized Images in directory: ", OUT_DIR+"/normalized_images")
  os.mkdir(OUT_DIR+"/normalized_images")

def get_images():
  images = list()
  for image in glob.glob(BASE_DIR + "/*." + IMG_EXT, recursive=True):
      images.append(image)
  return images

def perform_normalization(images):
  for image in tqdm(images):
      color_normalize(image)

def color_normalize(image):
  
  img = cv2.imread(image)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  
  b = img.copy()
  # set green and red channels to 0
  b[:, :, 1] = 0
  b[:, :, 2] = 0

  g = img.copy()
  # set blue and red channels to 0
  g[:, :, 0] = 0
  g[:, :, 2] = 0

  r = img.copy()
  # set blue and green channels to 0
  r[:, :, 0] = 0
  r[:, :, 1] = 0

  img = (b/b.max() + g/g.max() + r/r.max())
  img = cv2.resize(img, (512, 512)) 
  img = (img * 255).astype(np.uint8)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  cv2.imwrite(OUT_DIR+"/normalized_images/"+os.path.basename(image), img)

if __name__=="__main__":
  images = get_images()
  perform_normalization(images)
  


