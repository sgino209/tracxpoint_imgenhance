import cv2
import colorsys
import numpy as np
import tensorflow as tf
from pathlib import Path
from os import path, makedirs
import matplotlib.pyplot as plt
from warnings import simplefilter
from PIL import Image, ImageEnhance
import imquality.brisque as brisque
from tensorflow.python.ops.numpy_ops import np_config
print('OpenCV version: %s' % cv2.__version__)
print('Tensorflow version: %s' % tf.__version__)

# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

data_dir = '/Users/shahargino/data/tracxpoint__data'
img_files_list = [str(x) for x in Path(data_dir).rglob('*.tif')]
print('%d images found' % len(img_files_list))

# -----------------------------------------------------------------------

def histeq(img):
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

def clahe(img, grid_size=(8,8)):
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

def gamma_correction(img, gamma=0.001):
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

def saturation(img, saturation_factor=1.1):
  """ Image Saturation, based on PIL library """

  pil_img = Image.fromarray(img)
  res = np.array(ImageEnhance.Color(pil_img).enhance(saturation_factor))

  return res

# -----------------------------------------------------------------------

def sharpening(img):
  """ Image sharpening by a kernel operation """
  
  kernel = np.array([[0, -1, 0],
                     [-1, 5,-1],
                     [0, -1, 0]])
  
  res = cv2.filter2D(src=img, ddepth=-1, kernel=kernel)

  return res

# -----------------------------------------------------------------------

def denoise(img, mode='bilateral', median_kernel=11, d=9, sigmaColor=75, sigmaSpace=75):
  """ Applies the bilateral filter to an image, highly effective in noise removal while keeping edges sharp """

  if mode == 'median':
    res = cv2.medianBlur(img, median_kernel)
  
  elif mode == 'bilateral':
    res = cv2.bilateralFilter(img, d, sigmaColor, sigmaSpace)

  else:
    res = img.copy()

  return res

# -----------------------------------------------------------------------

def nl_denoise(img, h=10, template_win=7, search_win=21, temporal_index=2, temporal_window=3):
  """ Non-local Means Denoising algorithm to remove noise in the image
      If img is a list of images, then temporal information will be exploit.
      For example, if img is a list of 5 frames, and temporal_index=2 and 
      temopral_window=3 then frame-1, frame-2 and frame-3 are used to denoise frame-2 """

  if type(img) == list:
    res = cv2.fastNlMeansDenoisingColoredMulti(img, temporal_index, temporal_window, None, h, h, template_win, search_win)

  else:
    res = cv2.fastNlMeansDenoisingColored(img, None, h, h, template_win, search_win)

  return res

# -----------------------------------------------------------------------

def iqa_score(img, resize=(380, 507)):
  """ Image Quality Assessment with no-reference, return the BRISQUE score """
  
  resized_img = cv2.resize(img, resize)
  iqa_score = brisque.score(resized_img)

  return iqa_score

# -----------------------------------------------------------------------

def image_enhance(img, gamma=0.001, sat=1.1):
  """ Gamma Correction + Histogram Equalization + CLAHE + Local Denoise + Sharpening + NonLocal Denoise + Saturation """

  gamma_img = gamma_correction(img, gamma)
  histeq_img = histeq(gamma_img.astype(np.uint8))
  clahe_img = clahe(histeq_img)
  denoise_img = denoise(clahe_img)  
  nl_denoise_img = nl_denoise(denoise_img)
  sharp_img = sharpening(nl_denoise_img)
  sat_img = saturation(sharp_img, sat)

  return sat_img

# -----------------------------------------------------------------------

print('Started')

for k, img_file in enumerate(img_files_list):

  print('Processing (%d/%d): %s' % (k+1, len(img_files_list), img_file))

  bayer_img = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)
  tif_img = cv2.cvtColor(bayer_img, cv2.COLOR_BAYER_BG2BGR)

  res_img = image_enhance(tif_img)

  tif_score = iqa_score(tif_img.astype(np.uint8))
  res_score = iqa_score(res_img)

  out_file = img_file.replace(data_dir, 'results').replace('.tif', '_iqa_%.2f_to_%.2f.tif' % (tif_score, res_score))
  out_dir = path.dirname(out_file)
  if not path.exists(out_dir):
    makedirs(out_dir)
  cv2.imwrite(out_file, res_img)

# -----------------------------------------------------------------------

print('completed successfully')

