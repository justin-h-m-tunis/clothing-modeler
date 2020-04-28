import cv2
import numpy as np
import scipy.optimize
import os

def getBkgDistRGB(img, bkg, value_weight=0,rgb_weight = .7,gaussian_kernel=5):
    im = np.dstack((cv2.cvtColor(img,cv2.COLOR_RGB2HSV),img.astype(float)*rgb_weight))
    bk = np.dstack((cv2.cvtColor(bkg,cv2.COLOR_RGB2HSV),bkg.astype(float)*rgb_weight))
    im[:,:,1:3] = im[:,:,1:3].astype(np.float) * value_weight
    bk[:,:,1:3] = bk[:,:,1:3].astype(np.int16) * value_weight
    im = cv2.GaussianBlur(im,(gaussian_kernel,gaussian_kernel),0)
    bk = cv2.GaussianBlur(bk,(gaussian_kernel,gaussian_kernel),0)
    return np.linalg.norm(im - bk,axis=2)

def getBkgDistDepth(img, bkg, value_weight=.1,gaussian_kernel=3):
    im = cv2.GaussianBlur(img,(gaussian_kernel,gaussian_kernel),0)
    bk = cv2.GaussianBlur(bkg,(gaussian_kernel,gaussian_kernel),0)
    return np.linalg.norm(im - bk,axis=2)

def removeBkg(img,dist,thresh,combine='or',kernel=5):
    if combine == 'or':
        m = np.zeros((np.shape(dist[0,:,:]))).astype(bool)
        for i in range(len(thresh)):
            m = m | (dist[i,:,:] < thresh[i])
    elif callable(combine):
        m = np.zeros((np.shape(dist[0, :, :]))).astype(bool)
        for i in range(len(thresh)):
            m = combine(m,(dist[i, :, :] < thresh[i]))
    else:
        m = np.ones((np.shape(dist[0,:, :]))).astype(bool)
        for i in range(len(thresh)):
            m = m & ~(dist[i,:, :] > thresh[i])
    m = ~(cv2.GaussianBlur((~m).astype(np.uint8),(kernel,kernel),0).astype(bool))
    obj = np.ma.array(img,mask=np.repeat(m,np.size(img,axis=2)))
    return obj.filled(0)

'''
if __name__ == '__main__':
    img_path = 'data/color/'
    depth_path = 'data/depth/'
    bkg_path = 'data/color_bkg/'
    depth_bkg_path = 'data/depth_bkg/'
    img_names = [os.listdir(img_path), os.listdir(depth_path)]
    depth_thresh = 30
    rgb_thresh = 50
    for i in range(40,len(img_names[0])):
        print(i)
        print(type(removeBkg))
        f = [img_names[0][i],img_names[1][i]]
        img = cv2.imread(img_path + f[0])
        rgb_dist = getBkgDistRGB(img,cv2.imread(bkg_path + f[0]))
        depth = cv2.imread(depth_path + f[1])
        depth_dist = getBkgDistDepth(depth,cv2.imread(depth_bkg_path + f[1]))
        cv2.imshow(' ',(rgb_dist*np.sqrt(1/3)).astype(np.uint8))
        bkg_thresh_rgb = removeBkg(img,np.array([rgb_dist,depth_dist]),[rgb_thresh,depth_thresh],'or')
        bkg_thresh_depth = removeBkg(depth,np.array([rgb_dist,depth_dist]),[rgb_thresh,depth_thresh],'or')
        cv2.imshow('',bkg_thresh_rgb)
        cv2.imwrite('data/thresholded/' + f[0],bkg_thresh_rgb)
        cv2.imwrite('data/thresholded/' + f[1], bkg_thresh_depth)
        cv2.imwrite('data/pretty/' + f[0],(rgb_dist*np.sqrt(1/3)).astype(np.uint8))
        cv2.waitKey(10)
'''