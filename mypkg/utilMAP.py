# mypkg/utils.py の内容例
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def loadWN_maptxt(path):
    f0 = open(path,'r',encoding='UTF-8')
    datalist = f0.readlines()
    #print(datalist[0])
    f0.close()
    la = datalist[0].strip().split('\t')  #最初の空白（ｘ、ｙ）を除き（stripし）、tabで区切る(split)
    WN = []
    for a in la:
        WN.append(float(a))
    print('number of wavenumber channels',len(WN))
    return WN

def loadSP_maptxt(path):
    df1 = pd.read_csv(path, sep=r'[,\t]', encoding='utf-8', engine='python') #Using r string literals for regular expressions
    # Indexを0列目に挿入し、インデックスをリセット
    df_reset = df1.reset_index()  # インデックスが「index」という名前の新しい列になる
    df_reset.rename(columns={'level_0': 'x'}, inplace=True)  # 列名を変更
    df_reset.rename(columns={'level_1': 'y'}, inplace=True)  # 列名を変更
    df11 = df_reset
    print('\n','>>> df11 >>>>')
    print(df11.head())
    df_xy = df11[['x','y']]
    print('\n','>>> df_xy >>>')
    print(df_xy.head())
    df111 = df11.drop(['x','y'],axis='columns')
    df111T = df111.T
    # Indexを0列目に挿入し、インデックスをリセット
    df_reset = df111T.reset_index()  # インデックスが「index」という名前の新しい列になる
    df_reset.rename(columns={'index': 'WN'}, inplace=True)  # 列名を変更
    mapdata = df_reset
    print('\n','>>> mapdata : dataframe >>>')
    print(mapdata.head())
    print(mapdata.shape)
    print(mapdata.info())
    return df11, df_xy, mapdata

def plot_mapsp4(np_WN2, np_mapdata):
    #import matplotlib.pyplot as plt
    #numpy plot: min, max, averaged spectra in np_mapdata
    # 列方向（axis=0）の平均を計算
    # averaged spectrum plot
    np_mean = np.mean(np_mapdata, axis=1)
    #plot_SP(np_WN2, np_mean,'averaged spectrum for mapdata')
    x = np_WN2
    y = np_mean
    plt.figure(figsize=(8, 3))  # 横長サイズを指定    
    plt.plot(x, y,linestyle="-",
             color = "black", alpha = 1.0,
             linewidth = 1)
    # min. and max. spectrum plot
    np_min = np.min(np_mapdata, axis=1)
    np_max = np.max(np_mapdata, axis=1)
    #x = np_WN2
    y = np.stack([np_min,np_max],axis=1)
    #'-': Solid line.'--': Dashed line.'-.': Dash-dot line.':': Dotted line.
    plt.plot(x, y,linestyle=":",
             color = "black", alpha = 0.5,
             linewidth = 1)
    title = 'min., max., and averaged spectra'
    plt.title(title)
    plt.xlabel('Wavenumvers ($cm^-$$^1$)')
    plt.ylabel('Counts')
    #plt.legend()
    #plt.grid(True)
    plt.show()

def df_to_numpy(mapdata,WN):
    #import numpy as np
    # transform from dataframe to numpy
    lc2 = mapdata.columns.to_list()
    print('lc2 size(mapdata.columns):',len(lc2))
    print(lc2[:10], '...', lc2[-5:])
    np_WN = np.array(WN)
    print('\n','>>> np_WN >>>')
    print(np_WN.shape)
    print(np_WN[:5],' ... ',np_WN[-5:])
    #mapdata: (1024, 1016)
    print('\n','>>> np_mapdata wthout WN: droped WN column')
    np_mapdata = np.array(mapdata.drop('WN',axis='columns'))
    print('\n','>>> np_mapdata >>>','\n',np_mapdata.shape)
    #np_mapdata (1024, 1015): chnnels of spectrum is 1024. 1015 datapoints
    print("5 rows × 3 columns in np_mapdata")
    print(np_mapdata[:5,:3]) #np_mapdata[specrrum方向,datapoint方向]
    return np_WN, np_mapdata

def map_im(np_mapdata,x_dim, y_dim, f):
    #image with sum area
    np_suma = np_mapdata.sum(axis=0)   #各スペクトルの面積
    im2 = np_suma.reshape(y_dim, x_dim)   #(Y,X)
    im = plt.imshow(im2, cmap='gray')   #imshowはnumpy配列を画像表示するメソッド
    plt.colorbar(im, label='Area intensity')
    plt.title(f)
    plt.show()
    return np_suma

def map_imjet(np_mapdata,x_dim, y_dim, f):
    #image with sum area
    np_suma = np_mapdata.sum(axis=0)   #各スペクトルの面積
    im2 = np_suma.reshape(y_dim, x_dim)   #(Y,X)
    im = plt.imshow(im2, cmap='jet')   #imshowはnumpy配列を画像表示するメソッド
    plt.colorbar(im, label='Area intensity')
    plt.title(f)
    plt.show()
    return np_suma

def im_extract(np_mapdata,x_dim, y_dim, fn, params):
  #import matplotlib.pyplot as plt
  #import numpy as np
  #params[0]: np_WN(wavenumbers) index from wn1 to wn2.
  #params[1]: x_start, x_end
  #params[2]: y_start, y_end
  #===== example code for preparing params
  #===== specify the wavenumbers for the OH strech region 
  #wn1,wn2 = 3000,3800   #input the same wavenumbers for spectral extraction.
  #w1id,w2id = WNid(np_WN,wn1,wn2)
  #if wn1==wn2:
  #  w1id = int((w1id+w2id)/2)
  #  w2id = w1id
  #print(f"{'w1id,w2id':<25}: {w1id:15.0f}\t{w2id:15.0f}")
  #print(f"{'np_WN[w1id],np_WN[w2id]':<25}: {np_WN[w1id]:15.1f}\t{np_WN[w2id]:15.1f}")
  #x_start, x_end = 6,20   #input x region from x_start to x_end
  #y_start, y_end = 7,13   #input y region from y_start to x_end
  #print(f"{'x_start, x_end':<25}: {x_start:15.0f}\t{x_end:15.0f}")
  #print(f"{'y_start, y_end':<25}: {y_start:15.0f}\t{y_end:15.0f}")
  #params = []
  #params.append((w1id,w2id)) #extraxt np_WN(wavenumbers) index from wn1 to wn2.
  #params.append((x_start, x_end)) #extraxt x region from x_start to x_end
  #params.append((y_start, y_end)) #extraxt y region from y_start to y_end
  #print(params)
  #======
  w1id,w2id = params[0]
  #w1id,w2id = WNid(np_WN,wn1,wn2)
  #image with sum area
  if w1id==w2id:
    npa = np_mapdata[w1id,:]  #intensity at a specified wavenumber指定波数の強度
  else:
    npa = np_mapdata[w1id:w2id,:].sum(axis=0)   #area intensity in a specified spectral region指定範囲波数の面積強度
  
   # Display the base image
  image = npa.reshape(y_dim, x_dim) 
  plt.imshow(image, cmap='gray', aspect='equal')

  # Define the region for image2
  # image2 = image[7:13,6:20] means:
  # y-axis from 7 (inclusive) to 13 (exclusive)
  # x-axis from 6 (inclusive) to 20 (exclusive)
  x_start, x_end = params[1]
  y_start, y_end = params[2]
  image2 = image[y_start:y_end, x_start:x_end]

  # Overlay image2 using extent to position it correctly
  im_overlay = plt.imshow(image2, cmap='jet', aspect='equal', alpha=0.7,
                          extent=[x_start, x_end, y_end, y_start])
  plt.colorbar(im_overlay, label='intensity')
  plt.xticks(range(0, x_dim, 5)) # Set x-axis ticks intervals
  plt.yticks(range(0, y_dim, 5)) # Set y-axis ticks intervals
  plt.show()
  return image2

def np_SPs(np_mapdata,x_start,x_end,y_start,y_end,y_dim):
    #np_SPs;spectra in a specified spectral region of image2
    for i in range(x_start,x_end):
          SP_start = y_dim*(i)+(y_start)
          npaa = np_mapdata[:,SP_start:(SP_start+(y_end-y_start))]
          if i==x_start:
                np_SPs = npaa
          else:
                np_SPs = np.column_stack((np_SPs,npaa))
    print('np_SPs.shape',np_SPs.shape)
    return np_SPs

def base(npx,sx,sy,ex,ey):
    #base as line from a start to a end datapoint
    y = (ey - sy) / (ex - sx) * (npx - sx) + sy
    return y

def basecorr1(np_WN_m,np_m):
    sa = 0
    ea = len(np_WN_m)-1     #w2id-w1id-1
    i = 0   # P0 intensity(counts)
    sx = np_WN_m[sa]
    ex = np_WN_m[ea]
    x = np_WN_m  #numpy.ndarray
    #start to end baseleine correction
    tpl_b = np_m.shape  #tuple(WNチャネル数,datapoint数)
    np_mb = np_m.copy()
    for i in range(tpl_b[1]):
        sy = np_m[sa,i]
        ey = np_m[ea,i]
        np_mb[:,i] = np_m[:,i] - base(x,sx,sy,ex,ey)
    return np_mb
