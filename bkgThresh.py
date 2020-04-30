import cv2
import numpy as np
import scipy.optimize
from scipy.ndimage.filters import *
import os
import pickle

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
    depth_scale = np.max([np.max(depths),np.max(bkgdepths)])
    responses = np.zeros((len(ims),len(ims[0]),8,np.size(ims,3)))
    for i in range(np.size(ims,3)):
        print(ims[:,:,:,i])
        hsv_ims = cv2.cvtColor(ims[:, :, :, i], cv2.COLOR_BGR2HSV)
        hsvdist = np.tanh(2*abs(hsv_ims - cv2.cvtColor(bkgs[:,:,:,i],cv2.COLOR_BGR2HSV))/255.0)
        ddist = np.tanh(2*abs(depths[:,:,i].astype(float) - bkgdepths[:,:,i].astype(float))/depth_scale)
        depth_mask = thresholdFn(depths[:,:,i],depthrange[0],depthrange[1])
        backdrop_mask = thresholdFn(hsv_ims,np.array(backdrop_hsv_range[0]),np.array(backdrop_hsv_range[1]))
        responses[:,:,:,i] = np.array([hsvdist,ddist,backdrop_mask,depth_mask])
    save_resp = [output, output_uncertainties]
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
    n_images = np.size(responses,3)
    n_filters = np.size(responses,2)
    obj_means = [np.sum(responses[:,:,:,i] * weights[0,:],axis=2) + biases for i in range(size(responses,2))]
    obj_uncertainties = [np.linalg.norm(sqrt(n_filters/(n_filters-1))*np.linalg.norm((responses[:,:,:,i]-obj_means) * weights[0,:]),axis=2) for i in range(size(responses,2))]
    bkg_means = [np.sum(responses[:, :, :, i] * weights[1, :], axis=2) + biases for i in range(size(responses, 2))]
    bkg_uncertainties = [np.linalg.norm(sqrt(n_filters / (n_filters - 1)) * np.linalg.norm((responses[:, :, :, i] - obj_means) * weights[1, :]), axis=2) for i in range(size(responses, 2))]
    output = np.array([obj_means,bkg_means])
    output_uncertainties =np.array([obj_uncertainties,bkg_uncertainties])
    save_out = [output,output_uncertainties]
    pickle.dump(save_out, open("mean_uncertainties.pkl", "wb"))
    return output,output_uncertainties

def sigmoid(x):
    np.power(1-np.power(e,-x),-1)

def calculateLoss(output,uncertainties):
        #number of uncertain pixels
    loss = np.tile(sigmoid(abs(output[:,:,:,0]-output[:,:,:,1]) - np.sum(uncertainties,axis=2)),(1,1,1,2))
    return loss

def getDerivs(weights, biases, responses, output, uncertainties):
    n = np.size(responses,3)
    h = np.size(responses,0)
    w = np.size(responses,1)
    channels = np.size(responses,2)
    grad_w = np.zeros(np.shape(weights)) #2xf
    grad_b = np.zeros(np.shape(biases)) #f
    L = calculateLoss(output,uncertainties)
    dLdx = L*(1-L) #DDnx2
    dxdo = -2*((output == np.max(output,axis=2)).astype(float)-.5) #DDnx2
    dxds = np.array([-1,-1])
    c = np.sum(responses,axis=3)
    dodw = c/n #DDnxf
    dsdw = n/(n+1)*((c-output) @ (dodw*weights ** 2) + weights*(c-output) ** 2)/uncertainties #DDnxf
    dodb = np.ones(np.shape(dodw))
    dsdb = n/(n+1)*((c-output) @ weights ** 2)/uncertainties #DDnx1

    grad_w = np.reshape(np.swapaxis(dLdx * dxdo,2,3),(h*w*n,2)) @ np.reshape(np.swapaxis(dodw,2,3),(h*w*n,channels)).T + np.reshape(np.swapaxis(dLdx * dxds,2,3),(h*w*n,2)) @ np.reshape(np.swapaxis(dsdw,2,3),(h*w*n,channels)).T
    grad_b = np.reshape(np.swapaxis(dLdx * dxdo,2,3),(h*w*n,2)) @ np.reshape(np.swapaxis(dodb,2,3),(h*w*n,channels)).T + np.reshape(np.swapaxis(dLdx * dxds,2,3),(h*w*n,2)) @ np.reshape(np.swapaxis(dsdb,2,3),(h*w*n,channels)).T
    return grad_w,grad_b

def optimizeWeights(ims,depths,bkgs,bkgdepths, epochs=100,learning_rate = .001):
    in_size = 8
    out_size = 2
    W = np.random.uniform(-np.sqrt(6/(in_size+out_size)),np.sqrt(6/(in_size + out_size)),out_size*in_size)
    W = np.reshape(W,(in_size,out_size))
    b = np.zeros(out_size) #might need to init to .5
    print(W,b)
    for i in range(epochs):
        print(i)
        responses = getResponses(ims, depths, bkgs, bkgdepths)
        [means, uncertainties] = getOutput(responses, W, b)
        cv2.imshow("mask",(1-np.argmin(means,axis=2))*255)
        [dW, db] = getDerivs(W,b,responses,means,uncertainties)
        W -= dW*learning_rate
        b -= db*learning_rate
        print(W,b)
        cv2.waitKey(1)
    return W,b

def removeBkg(ims,depths,bkgs,bkgdepths,optimize=True):
    if optimize:
        weights, biases = optimizeWeights(ims,depths,bkgs,bkgdepths)
    else:
        weights = np.zeros((2,8))
        biases = np.zeros((1,8))
    responses = getResponses(ims, depths, bkgs, bkgdepths)
    [means, uncertainties] = getOutput(responses, W, b)
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
    obj = np.ma.array(img,mask=np.repeat(mask,np.size(img,axis=2)))
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