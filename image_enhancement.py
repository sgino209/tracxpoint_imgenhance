import cv2
import colorsys
import numpy as np
import tensorflow as tf
from pathlib import Path
from os import path, makedirs
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
from tensorflow.python.ops.numpy_ops import np_config

data_dir = '/Users/shahargino/Downloads/TracxPoint'
img_files_list = [str(x) for x in Path(data_dir).rglob('*.tif')]
print('%d images found' % len(img_files_list))

# -----------------------------------------------------------------------

def img_equalization(img):
  """ Apply Image Eualization to the converted image in LAB format to
      only Lightness component and convert back the image to RGB """

  # Color input:
  if len(img.shape) == 3:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab_planes = list(cv2.split(lab))
    lab_planes[0] = cv2.equalizeHist(lab_planes[0])
    lab = cv2.merge(lab_planes)
    res = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
  
  # Gray input:
  else:
    res = cv2.equalizeHist(img.astype(np.uint8))

  return res

# -----------------------------------------------------------------------

def img_clahe(img, grid_size=(8,8)):
  """ Apply CLAHE to the converted image in LAB format to 
      only Lightness component and convert back the image to RGB """
  
  clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=grid_size)
  
  # Color input:
  if len(img.shape) == 3:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab_planes = list(cv2.split(lab))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    res = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

  # Gray input:
  else:
    res = clahe.apply(img)

  return res

# -----------------------------------------------------------------------

def gamma_correction(img, gamma=2.2):
  """ Apply Gamma Correction, so that Out = MaxVal*(In/MaxVal)**(1/gamma)"""

  max_val = np.iinfo(img.dtype).max

  invGamma = 1 / gamma

  table = [np.clip(pow(k/max_val, invGamma) * max_val, 0, max_val) for k in range(max_val+1)]
  table = np.array(table, img.dtype)

  if img.dtype == np.uint8:
    res = cv2.LUT(img, table)
  
  else:
    res = tf.image.adjust_gamma(img/max_val, gamma=gamma, gain=max_val).numpy()

  return res

# -----------------------------------------------------------------------

def colorize(image, hue):
  """ Colorize PIL image `original` with the given `hue` (hue within 0-360), returns another PIL image """

  def shift_hue(arr, hout):
    rgb_to_hsv = np.vectorize(colorsys.rgb_to_hsv)
    hsv_to_rgb = np.vectorize(colorsys.hsv_to_rgb)

    r, g, b, a = np.rollaxis(arr, axis=-1)
    h, s, v = rgb_to_hsv(r, g, b)
    h = hout
    r, g, b = hsv_to_rgb(h, s, v)
    arr = np.dstack((r, g, b, a))
    return arr

  img = image.convert('RGBA')
  arr = np.array(np.asarray(img).astype('float'))
  res = Image.fromarray(shift_hue(arr, hue/360.).astype('uint8'), 'RGBA')

  return res

# -----------------------------------------------------------------------

def image_enhance(img, gamma=0.001, saturation=3):
  """ Gamma Correction + CLAHE + Histogram Equalization + Saturation """

  gamma_img = gamma_correction(img, 0.001)
  clahe_img = img_clahe(gamma_img.astype(np.uint8))
  histeq_img = img_equalization(clahe_img)
  pil_img = Image.fromarray(histeq_img)
  sat_img = np.array(ImageEnhance.Color(pil_img).enhance(3))
  sat_img_color = cv2.cvtColor(sat_img, cv2.COLOR_GRAY2BGR)

  return sat_img_color

# -----------------------------------------------------------------------

print('Started')

for k, img_file in enumerate(img_files_list):
  
  print('Processing (%d/%d): %s' % (k+1, len(img_files_list), img_file))

  img = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)
  ref_img = cv2.imread(img_file.replace('.tif', '.jpg'), cv2.IMREAD_UNCHANGED)

  res_img = image_enhance(img)

  out_file = img_file.replace(data_dir, 'results')
  out_dir = path.dirname(out_file)
  if not path.exists(out_dir):
    makedirs(out_dir)  
  cv2.imwrite(out_file, res_img)

# -----------------------------------------------------------------------

print('completed successfully')

