import cv2
import numpy as np
import scipy.optimize
from scipy.ndimage.filters import *
import os
import pickle
from numpy import tensordot as td

def getBkgDistances(img, im_depth, bkg, bkg_depth,gaussian_kernel=(5,5)):
    im_blur = cv2.GaussianBlur(img,gaussian_kernel,0)
    bkg_blur = cv2.GaussianBlur(bkg, gaussian_kernel, 0)
    rgb_dist = np.linalg.norm(im_blur - bkg_blur,axis=2)/(255*np.sqrt(3))
    hue_dist = abs(cv2.cvtColor(im_blur,cv2.COLOR_BGR2HSV)[:,:,0].astype(float) - cv2.cvtColor(bkg_blur,cv2.COLOR_BGR2HSV)[:,:,0].astype(float))/179.0
    cv2.imshow("rgby",rgb_dist)
    depth_dist = abs(im_depth.astype(float) - bkg_depth.astype(float))/(np.max([np.max(im_depth),np.max(bkg_depth)]))
    print(np.max(rgb_dist),np.max(hue_dist),np.max(depth_dist))
    return np.stack((rgb_dist, hue_dist, depth_dist),axis=2)

def distanceFn(I1, I2):
    return np.linalg.norm(I1.astpye(float)-I2.astype(float),axis=2)/(np.max([np.max(I1),np.max(I1)]))

def thresholdFn(I1, lb, ub, direction="inclusive"):
    if direction=="inclusive":
        if(len(np.shape(I1))==3):
            return ((I1 > lb) & (I1 < ub)).all(axis=2).astype(float)
        else:
            return ((I1 > lb) & (I1 < ub)).astype(float)
    else:
        if(len(np.shape(I1))==3):
            return ((I1 < lb) | (I1 > ub)).all(axis=2).astype(float)
        else:
            return ((I1 < lb) | (I1 > ub)).astype(float)


def convolutionFn(I1,h):
    l = len(I1)
    w = len(I1[0])
    h_half = ((h-1)/2).astype(int)
    I2 = np.zeros((l+2*h_half,w+2*h_half))
    for i in range(len(h)):
        for j in range(len(h[0])):
            for k in range(len(h[0,0])):
                I2[i:l-2*h_half+i,j:w-2*h_half+j] += h[i,j,k]*I1[:,:,k]
    return I2[h_half:l-h_half,h_half:w-h_half]

# I realize now I accidentally just imeplemented a convolutional NN, that you tune by hand, I didn't realize this at
# 5am when I wrote it -__-
# but hey at least realizing this will make it better

#inputs = hdist, sdist, vdist, ddist, depthmask,backdropmask, filters = gaussian* (x,y,z grad, 3d laplacian)
def getResponses(ims,depths,bkgs,bkgdepths,depthrange=(700,1270),backdrop_hsv_range=((136,47,18),(198,58,94))):
    for i in range(3):
        ims[:,:,i,:] = gaussian_filter(ims[:, :, i, :], 1)
        bkgs[:,:,i,:] = gaussian_filter(bkgs[:,:,i,:],1)
    depths[:, :, :] = gaussian_filter(depths[:, :, :], 1)
    bkgdepths[:, :, :] = gaussian_filter(bkgdepths[:, :, :], 1)

    depth_scale = np.max([np.max(depths),np.max(bkgdepths)])
    responses = np.zeros((len(ims),len(ims[0]),6,np.shape(ims)[3]))
    print("computing responses")
    for i in range(np.shape(ims)[3]):
        hsv_ims = cv2.cvtColor(ims[:, :, :, i], cv2.COLOR_BGR2HSV)
        hsvdist = np.tanh(2*abs(hsv_ims - cv2.cvtColor(bkgs[:,:,:,i],cv2.COLOR_BGR2HSV))/255.0)
        ddist = np.tanh(2*abs(depths[:,:,i].astype(float) - bkgdepths[:,:,i].astype(float))/depth_scale)
        depth_mask = thresholdFn(depths[:,:,i],depthrange[0],depthrange[1])
        backdrop_mask = thresholdFn(hsv_ims,np.array(backdrop_hsv_range[0]),np.array(backdrop_hsv_range[1]))
        responses[:,:,:,i] = np.dstack((hsvdist,
                                       ddist,
                                       backdrop_mask,
                                       depth_mask))
    print(np.shape(responses))
    save_resp = responses
    pickle.dump(save_resp, open("responses.pkl", "wb"))
    return responses

'''def createFilteredResponse(input, gaussian_size):
    input = np.swapaxis(input,2,3)
    responses = np.zeros((len(input),len(input[0],size(ims,1),40)))
    for n in range(np.size(input,3)):
        responses[:,:,:,5*n] = gaussian_filter(input[:,:,:,n], 1)
        responses[:,:,:,5*n+1] = gaussian_filter(input[:,:,:,n], 1, order=2)
        responses[:,:,:,5*n+2] = sobel(input[:,:,:,n], axis=0)
        responses[:,:,:,5*n+3] = sobel(input[:,:,:,n], axis=1)
        responses[:,:,:,5*n+4] = sobel(input[:,:,:,n], axis=2)
    responses = swapaxis(responses,2,3)
    pickle.dump(responses, open("responses.pkl", "wb"))
    return responses
'''
def getOutput(responses,weights,biases):
    print("Getting Weighted Outputs")
    n_images = np.shape(responses)[3]
    n_filters = np.shape(responses)[2]
    x_ = np.sum(responses,axis=(2,3))/(n_images*n_filters)
    W_ = np.sum(weights,axis=1)
    obj_means = np.dstack((np.sum((responses[:,:,:,i] + biases) * weights[0,:],axis=2) for i in range(np.shape(responses)[3])))/W_[0]
    bkg_means = np.dstack((np.sum((responses[:, :, :, i] + biases) * weights[1, :], axis=2) for i in range(np.shape(responses)[3])))/W_[1]
    obj_uncertainties = np.dstack((np.sqrt(n_filters/(n_filters-1))*np.linalg.norm(np.moveaxis(np.moveaxis(responses[:,:,:,i],2,0)-x_,0,2)**2 * weights[0,:] ** 2,axis=2) for i in range(np.shape(responses)[3])))/W_[0]
    bkg_uncertainties = np.dstack((np.sqrt(n_filters / (n_filters - 1)) * np.linalg.norm(np.moveaxis(np.moveaxis(responses[:, :, :, i],2,0) - x_,0,2)**2 * weights[1, :] ** 2, axis=2) for i in range(np.shape(responses)[3])))/W_[1]

    output = np.stack((obj_means,bkg_means),axis=3)
    output_uncertainties = np.stack((obj_uncertainties,bkg_uncertainties),axis=3)
    save_out = np.stack((output,output_uncertainties),axis=4)
    pickle.dump(save_out, open("mean_uncertainties.pkl", "wb"))
    return output,output_uncertainties

def sigmoid(x):
    return np.power(1+np.power(np.e,-x),-1)

def calculateLoss(output,uncertainties):
        #number of uncertain pixels
    loss = sigmoid(abs(output[:,:,:,0]-output[:,:,:,1]) - np.sum(uncertainties,axis=3))
    return loss

def getDerivs(weights, biases, responses, output, uncertainties):
    print("Determining Gradients")
    n = np.shape(responses)[3]
    h = np.shape(responses)[0]
    w = np.shape(responses)[1]
    W_ = np.sum(weights,axis=1)
    channels = np.shape(responses)[2]
    L = calculateLoss(output,uncertainties)
    print(np.max(responses),np.max(output),np.max(uncertainties),np.max(L))

    dLdf = L*(1-L) #1xDDn #1x1
    dfdu = 2*np.equal(output,np.tile(np.expand_dims(np.max((output),axis=3),3),(1,1,1,2)),dtype=float)-1 #DDnx2 #1x2
    dfds = np.array([-1,-1]) # 1x2
    x = np.swapaxes(responses,2,3) + biases
    x_ = np.sum(x,axis=(3))/channels
    x_diff = np.moveaxis(np.tile(np.moveaxis(np.moveaxis(x,3,0) - x_,0,3),(2,1,1,1,1)),0,-1)
    dudw = (x+biases)/n #DDnxf #1xf
    dsdw = np.moveaxis(n/(n-1)*np.moveaxis(weights.T * (x_diff ** 2),3,0)/(uncertainties*W_**2),0,3) #DDnx2xf, 2xf
    dudb = weights #2xf
    dsdb = np.moveaxis(n/(n-1)*np.moveaxis(weights.T**2 * x_diff * (1-1/channels),3,0)/(uncertainties*W_**2),0,3) #DDnx2xf, 2xf

    dLdf = np.reshape(dLdf,(1,h*w*n))
    dfdu = np.reshape(dfdu,(h*w*n,2))
    dudw = np.reshape(dudw, (h * w * n, channels))
    dsdw = np.reshape(np.swapaxes(dsdw,3,4), (h * w * n, 2, channels))
    dsdb = np.reshape(np.swapaxes(dsdb,3,4), (h*w*n,2,channels))

    #print(np.max(dLdf),np.max(dfdu),np.max(dfds),np.max(dudw),np.max(dsdw),np.max(dudb),np.max(dsdb))

    grad_w = ((dLdf.T*dfdu).T @ dudw + np.sum((dLdf.T @ np.expand_dims(dfds,0)) * np.moveaxis(dsdw,2,0),axis=1).T)/np.size(dLdf)
    grad_b = (dLdf @ (dfdu @ dudb) - dLdf @ np.sum(dsdb,axis=1))/np.size(dLdf)
    return np.reshape(grad_w,(2,6)),np.reshape(grad_b,(6))

 ## not terrible weights: [[ 0.63047816  0.05227725 -0.46316327  0.62588872 -0.43291542  0.73123184]
 #[-0.75273859 -0.17357586 -0.80692281  0.48178903 -0.76356912 -0.76175928]] [ 0.02398244 -0.00168372  0.00998617 -0.06937236 -0.00429752 -0.08155093]
def optimizeWeights(ims,depths,bkgs,bkgdepths, epochs=10,learning_rate = .1, starting_weights=None, starting_biases=None):
    in_size = 6
    out_size = 2
    if starting_weights is None:
        W = np.random.uniform(0,np.sqrt(6/(in_size + out_size)),out_size*in_size)
        W = np.reshape(W,(out_size,in_size))
        #W[0,:] *= -1
    else:
        W = starting_weights
    if starting_biases is None:
        b = np.zeros(in_size) #might need to init to .5
    else:
        b = starting_biases
    print(W,b)
    for i in range(epochs):
        print(i)
        responses = getResponses(ims, depths, bkgs, bkgdepths)
        [means, uncertainties] = getOutput(responses, W, b)
        print((np.argmin(means[:,:,0,:],axis=2)))
        cv2.imshow("mask",((np.argmin(means[:,:,0,:],axis=2))*255).astype(np.uint8))
        cv2.waitKey(1)
        [dW, db] = getDerivs(W,b,responses,means,uncertainties)
        print(dW,db)
        W -= dW*learning_rate
        #W /= .5*np.sum(abs(W))
        b -= db*learning_rate
        print(W,b)
    return W,b

def removeBkg(ims,depths,bkgs,bkgdepths,optimize=True):
    if optimize:
        weights, biases = optimizeWeights(ims,depths,bkgs,bkgdepths)
    else:
        weights = np.zeros((2,6))
        biases = np.zeros((1,6))
    responses = getResponses(ims, depths, bkgs, bkgdepths)
    [means, uncertainties] = getOutput(responses, weights, biases)
    mask = (1 - np.argmin(means, axis=2)).astype(bool)
    cv2.imshow("mask", (1 - np.argmin(means, axis=2)) * 255)
    '''tanh_dist = np.tanh(dist)
    obj_certainties = np.multiply(tanh_dist,np.array(obj_weights))
    bkg_certainties = np.multiply(tanh_dist,np.array(bkg_weights))
    cv2.imshow("positive certainties",cv2.cvtColor((obj_certainties*255).clip(min=0).astype(np.uint8),cv2.COLOR_BGR2RGB))
    cv2.imshow("negative certainties",cv2.cvtColor((bkg_certainties*255).clip(min=0).astype(np.uint8),cv2.COLOR_BGR2RGB))
    #cv2.imwrite("certainties5.png",cv2.cvtColor(((certainties).clip(min=0)*255).astype(np.uint8),cv2.COLOR_BGR2RGB))
    mask = (np.sum(obj_certainties,axis=2) > np.sum(bkg_certainties,axis=2)).astype(bool)
    mask = ~(cv2.GaussianBlur((mask).astype(np.uint8),gaussian_kernel,0).astype(bool))
    cv2.imshow("mask", mask.astype(np.uint8)*255)
    '''
    obj = np.ma.array(img,mask=np.repeat(mask,np.shape(img)[2]))
    cv2.waitKey(1)
    return obj.filled(0)
'''
def getLoss(certainties):
    u = np.sum(certainties,axis=2)
    stddev = np.linalg.norm()

def optimizeOffsets(img,dist,weights):
'''


if __name__ == "__main__":
    rgb_output_path = 'data/preview/'
    depth_output_path = None
    img_ind = 1
    curr_dirname = os.path.dirname(__file__)
    img_path = 'data/color/'
    depth_path = 'data/depth/'
    bkg_path = 'data/bkg/color/'
    depth_bkg_path = 'data/bkg/depth/'
    dirs = [img_path, depth_path, bkg_path, depth_bkg_path]
    for dir in dirs:
        if (len(os.listdir(dir)) == 0):  # exit when empty folder found
            exit()
    print("Processing image " + str(img_ind))

    img_names = [os.listdir(img_path), os.listdir(depth_path)]
    f = ['color_' + str(img_ind) + '.png', 'Depth_' + str(img_ind) + '.png']
    img = cv2.imread(img_path + f[0])
    depth = cv2.imread(depth_path + f[1])
    dist = getBkgDistances(img, depth, cv2.imread(bkg_path + f[0]), cv2.imread(depth_bkg_path + f[1]))
    bkg_thresh_rgb = removeBkg(img, dist)
    bkg_thresh_depth = removeBkg(depth, dist)
    '''
    take bkg_thresh_rgb and bkg_depth_rgb and do static crop/color filtering
    '''

    crop_left = 0
    crop_right = 1000
    crop_top = 0
    crop_bottom = 1000

    if rgb_output_path is not None:
        cv2.imwrite(rgb_output_path + f[0], bkg_thresh_rgb)
        cv2.imshow("bla",bkg_thresh_rgb)
        cv2.waitKey()
    if depth_output_path is not None:
        cv2.imwrite(depth_output_path + f[0], bkg_thresh_depth)
        im = Image.open(depth_output_path + f[0])
        if (not self.is_crop_error(crop_top, crop_left, crop_bottom, crop_right, im)):
            im2 = im.crop((crop_left, crop_top, crop_right, crop_bottom))
            im2.save(depth_output_path + f[0])
        else:
            print("crop dimension error! showing original image")
    print("done!")