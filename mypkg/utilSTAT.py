"""
Created on Sat Aug 29 10:26:10 2026
Statistical analysis(CLS,PCA,MCR)
def least_sq(sample_spectrum, components)
def CLSscore(components,np_mapdata)
def CLSim(im_csR,im_csG,im_csB)
def calc_lof_vaf(X, C, S)
def calc_contribution(C, S)
def summary_table(X, C, S)
@author: Yasushi Nakata
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv

def least_sq(sample_spectrum, components):
   """
   sample_spectrum (unknown spectrum): array of w values.
   components (known spectra): array of n (number of components) columns with w values.
   This def returns an array of n values. Each value is the similarity score for the sample_spectrum and a component spectrum.
   """
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

#評価パラメータLOF、VAF、contribution
def calc_lof_vaf(X, C, S):
    """
    X : 元データ行列 (samples × variables)
    C : 濃度プロファイル (samples × components)
    S : スペクトル (components × variables)
    """
    # 再構成
    X_hat = C @ S  #@は行列積、Cの列数=Sの行数,np.dot（）はベクトル内積をもとめる時が良い
    # 残差
    residual = X - X_hat
    # LOF (Lack of Fit)
    #linalg.norm:行列のnorm大きさ
    #行列のすべての要素を二乗して、全部足し合わせて、最後に平方根を取る
    lof = np.linalg.norm(residual) / np.linalg.norm(X) * 100
    # VAF (Variance Accounted For)
    vaf = 100 * (1 - (np.linalg.norm(residual)**2 / np.linalg.norm(X)**2))
    return lof, vaf

def calc_contribution(C, S):
    """
    成分ごとの寄与率（Contribution）を計算
    """
    # 各成分の再構成量 ||C_i S_i||
    comp_power = np.array([np.linalg.norm(np.outer(C[:, i], S[i, :])) 
                           for i in range(C.shape[1])])
    # 寄与率（正規化）
    contribution = comp_power / comp_power.sum()
    return contribution

def summary_table(X, C, S):
    lof, vaf = calc_lof_vaf(X, C, S)
    contrib = calc_contribution(C, S)

    df = pd.DataFrame({
        "Component": [f"Comp {i+1}" for i in range(len(contrib))],
        "Contribution (%)": contrib * 100
    })

    print("=== MCR-ALS Summary ===")
    print(f"LOF: {lof:.3f} %")
    print(f"VAF: {vaf:.3f} %")
    print("\nComponent Contributions:")
    print(df)
