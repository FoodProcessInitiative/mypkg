import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
def least_sq(sample_spectrum, components):
   # sample_spectrum (unknown spectrum): array of w values.
   # components (known spectra): array of n (number of components) columns with w values.
   # This def returns an array of n values. Each value is the similarity score for the sample_spectrum and a component spectrum.
   similarity = np.dot(inv(np.dot(components, components.T)) , np.dot(components, sample_spectrum))
   return similarity

def CLSscore(components,np_mapdata):
  np_mean = np_mapdata.mean(axis=0) #np_mean: mean intensity of np_mapdata
  np_csR = np_mean.copy()
  np_csG = np_mean.copy()
  np_csB = np_mean.copy()
  for i in np.arange(len(np_mean)):
    query_spectra = np_mapdata[:,i]
    # Apply Least squares
    cs = least_sq(query_spectra, components)
    np_csR[i] = cs[0]
    np_csG[i] = cs[1]
    np_csB[i] = cs[2]
  return np_csR, np_csG, np_csB

from matplotlib.colors import LinearSegmentedColormap
def CLSim(im_csR,im_csG,im_csB):
  #from matplotlib.colors import LinearSegmentedColormap
  # カスタムカラーマップ定義
  cmap_red   = LinearSegmentedColormap.from_list("black_red",   [(0, 0, 0), (1, 0, 0)])
  cmap_green = LinearSegmentedColormap.from_list("black_green", [(0, 0, 0), (0, 1, 0)])
  cmap_blue  = LinearSegmentedColormap.from_list("black_blue",  [(0, 0, 0), (0, 0, 1)])

  plt.figure(figsize=(15, 5))
  plt.subplot(1, 3, 1)
  plt.imshow(im_csR, cmap=cmap_red)
  plt.title('Component R (Red Gradient)')
  plt.colorbar(label='Intensity')
  plt.axis('off')

  plt.subplot(1, 3, 2)
  plt.imshow(im_csG, cmap=cmap_green)
  plt.title('Component G (Green Gradient)')
  plt.colorbar(label='Intensity')
  plt.axis('off')

  plt.subplot(1, 3, 3)
  plt.imshow(im_csB, cmap=cmap_blue)
  plt.title('Component B (Blue Gradient)')
  plt.colorbar(label='Intensity')
  plt.axis('off')

  plt.tight_layout()
  plt.show()

  # Normalize each component to the range [0, 1] for proper RGB visualization
  im_csR_norm = (im_csR - np.min(im_csR)) / (np.max(im_csR) - np.min(im_csR))
  im_csG_norm = (im_csG - np.min(im_csG)) / (np.max(im_csG) - np.min(im_csG))
  im_csB_norm = (im_csB - np.min(im_csB)) / (np.max(im_csB) - np.min(im_csB))

  # Stack the normalized components to create an RGB image
  rgb_image = np.stack([im_csR_norm, im_csG_norm, im_csB_norm], axis=-1)
  #rgb_image = np.stack([im_csR, im_csG, im_csB], axis=-1)
  plt.imshow(rgb_image)
  plt.title('RGB Composite Image of normalized im_csR, im_CcsG, and im_csB')
  plt.axis('off')
  plt.show()
