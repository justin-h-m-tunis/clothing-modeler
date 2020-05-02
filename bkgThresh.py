import cv2
import numpy as np
import scipy.optimize
from scipy.ndimage.filters import *
from scipy.stats import *
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

def mahalaobisDist(im, mean, cov):
    h = np.shape(im)[0]
    w = np.shape(im)[1]
    im = np.reshape(im,(h*w,3))
    probs = np.sqrt(np.sum((im-mean) * (np.linalg.inv(cov) @ (im-mean).T).T,axis=1))
    return np.reshape(probs,(h,w))

def thresholdFn(I1, lb=0, ub=0,mean=None,cov=None, direction="inclusive", dist='linear'):
    if direction=="inclusive":
        '''if(len(np.shape(I1))==3):
            return ((I1 > lb) & (I1 < ub)).all(axis=2).astype(float)
        else:
            return ((I1 > lb) & (I1 < ub)).astype(float)'''
        if dist=='linear':
            return sigmoid(np.min(np.dstack((I1-lb,ub-I1)),axis=2))
        else:
            return multivariate_normal.pdf(I1,mean,cov)
    else:
        '''if(len(np.shape(I1))==3):
            return ((I1 < lb) | (I1 > ub)).all(axis=2).astype(float)
        else:
            return ((I1 < lb) | (I1 > ub)).astype(float)
            '''
        if dist=='linear':
            return sigmoid(np.max(np.dstack((lb-I1,I1-ub)),axis=2))
        else:
            I1 = I1/255
            #probs = multivariate_normal.pdf(I1, mean, cov)
            return chi2.cdf(mahalaobisDist(I1,mean,cov),3)

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
def getResponses(ims,depths,bkgs,bkgdepths,depthrange=(700,1270),bk_mean=None,bk_cov=None):
    for i in range(3):
        ims[:,:,i,:] = gaussian_filter(ims[:, :, i, :], (1,1,0))
        bkgs[:,:,i,:] = gaussian_filter(bkgs[:,:,i,:],(1,1,0))
    depths[:, :, :] = gaussian_filter(depths[:, :, :], (1,1,0))
    bkgdepths[:, :, :] = gaussian_filter(bkgdepths[:, :, :], (1,1,0.1))

    depth_scale = np.max([np.max(depths),np.max(bkgdepths)])
    responses = np.zeros((len(ims),len(ims[0]),6,np.shape(ims)[3]))
    print("computing responses")
    for i in range(np.shape(ims)[3]):
        cv2.imshow("ims",ims[:,:,:,i])
        cv2.imshow("bks",bkgs[:,:,:,i])
        hsv_ims = cv2.cvtColor(ims[:, :, :, i], cv2.COLOR_BGR2HSV)
        hsvdist = np.tanh(2*abs(hsv_ims - cv2.cvtColor(bkgs[:,:,:,i],cv2.COLOR_BGR2HSV))/255.0)
        cv2.imshow("hsv",(255*hsvdist).astype(np.uint8))
        ddist = np.tanh(2*abs(depths[:,:,i].astype(float) - bkgdepths[:,:,i].astype(float))/depth_scale)
        cv2.imshow("ddist",(255*ddist).astype(np.uint8))
        depth_mask = thresholdFn(depths[:,:,i],lb=depthrange[0],ub=depthrange[1])
        cv2.imshow("d mask",(255*depth_mask).astype(np.uint8))
        backdrop_mask = thresholdFn(hsv_ims,mean=bk_mean,cov=bk_cov, direction='exclusive', dist='normal')
        cv2.imshow("bkg mask",(255*backdrop_mask).astype(np.uint8))
        responses[:,:,:,i] = np.dstack((hsvdist,
                                       ddist,
                                       backdrop_mask,
                                       depth_mask))
        cv2.waitKey(1)
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
    W_ = np.sum(abs(weights),axis=1)
    obj_means = np.sum((np.swapaxes(responses,2,3) - biases) * weights[0,:],axis=3)/W_[0]
    bkg_means = np.sum((np.swapaxes(responses,2,3) - biases) * weights[1, :],axis=3)/W_[1]
    x = np.swapaxes(responses, 2, 3) - biases
    x_ = np.sum(x, axis=(3)) / n_filters
    x_diff = np.moveaxis(x, 3, 0) - x_
    obj_uncertainties = (np.sqrt(n_filters/(n_filters-1))*np.linalg.norm(np.moveaxis(x_diff,0,3)**2 * weights[0,:] ** 2,axis=3))/W_[0]
    bkg_uncertainties = (np.sqrt(n_filters / (n_filters - 1)) * np.linalg.norm(np.moveaxis(x_diff,0,3)**2 * weights[1, :] ** 2, axis=3))/W_[1]

    output = np.stack((obj_means,bkg_means),axis=3)
    output_uncertainties = np.stack((obj_uncertainties,bkg_uncertainties),axis=3)
    cv2.imshow("obj_means", (255*output[:,:,0,0]).astype(np.uint8))
    cv2.imshow("bkg_means", (255*output[:,:,0,1]).astype(np.uint8))
    cv2.imshow("obj_uncertainties", (255*output_uncertainties[:,:,0,0]/np.max(output_uncertainties)).astype(np.uint8))
    cv2.imshow("bkg_uncertainties", (255*output_uncertainties[:,:,0,1]/np.max(output_uncertainties)).astype(np.uint8))
    cv2.waitKey(1)

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
    print(np.max(responses),np.max(output),np.min(output),np.max(uncertainties),np.min(L))

    dLdf = L*(1-L) #1xDDn #1x1
    dfdu = 2*np.equal(output,np.tile(np.expand_dims(np.max((output),axis=3),3),(1,1,1,2)),dtype=float)-1 #DDnx2 #1x2
    dfds = np.array([-1,-1]) # 1x2
    x = np.swapaxes(responses,2,3) - biases
    x_ = np.sum(x,axis=(3))/channels
    x_diff = np.moveaxis(np.tile(np.moveaxis(np.moveaxis(x,3,0) - x_,0,3),(2,1,1,1,1)),0,-1)
    dudw = (x-biases)/n #DDnxf #1xf
    dsdw = np.moveaxis(n/(n-1)*np.moveaxis(weights.T * (x_diff ** 2),3,0)/(uncertainties*W_**2),0,3) #DDnx2xf, 2xf
    dudb = -weights #2xf
    dsdb = -np.moveaxis(n/(n-1)*np.moveaxis(weights.T**2 * x_diff * (1-1/channels),3,0)/(uncertainties*W_**2),0,3) #DDnx2xf, 2xf

    dLdf = np.reshape(dLdf,(1,h*w*n))
    dfdu = np.reshape(dfdu,(h*w*n,2))
    dudw = np.reshape(dudw, (h * w * n, channels))
    dsdw = np.reshape(np.swapaxes(dsdw,3,4), (h * w * n, 2, channels))
    dsdb = np.reshape(np.swapaxes(dsdb,3,4), (h*w*n,2,channels))

    #print(np.max(dLdf),np.max(dfdu),np.max(dfds),np.max(dudw),np.max(dsdw),np.max(dudb),np.max(dsdb))

    grad_w = ((dLdf.T*dfdu).T @ dudw + np.sum((dLdf.T @ np.expand_dims(dfds,0)) * np.moveaxis(dsdw,2,0),axis=1).T)/np.size(dLdf)
    grad_b = (dLdf @ (dfdu @ dudb) - dLdf @ np.sum(dsdb,axis=1))/np.size(dLdf)
    return np.reshape(grad_w,(2,6)),np.reshape(grad_b,(6))

 ## not terrible weights:[[-0.02310255 -0.09239055 -0.04196529 -0.03325345 -0.12176202 -0.00874592]
 #[-0.06079465  0.00337789 -0.02507019 -0.0489201  -0.10368871 -0.08661218]] [ 0.13654664  0.11183966  0.17248183  0.22858321 -0.01380614  0.10985678]
def optimizeWeights(ims,depths,bkgs,bkgdepths, epochs=10,learning_rate = .2, starting_weights=None, starting_biases=None, bk_mean=None, bk_cov=None):
    in_size = 6
    out_size = 2
    if starting_weights is None:
        in_size = 6
        out_size = 2
        W = np.random.uniform(0, np.sqrt(6 / (in_size + out_size)), out_size * in_size)
        W = np.reshape(W, (out_size, in_size))
        W = (W.T / np.sum(W, axis=1)).T
        W[1, :] *= -1
    else:
        W = starting_weights
    if starting_biases is None:
        b = np.ones(in_size) * .1
    else:
        b = starting_biases
    print(W,b)
    responses = getResponses(ims, depths, bkgs, bkgdepths, bk_mean=bk_mean, bk_cov=bk_cov)
    for i in range(epochs):
        print(i)
        [means, uncertainties] = getOutput(responses, W, b)
        cv2.imshow("mask",(np.argmin(means[:,:,0,:],axis=2)).astype(np.float))
        cv2.waitKey(1)
        [dW, db] = getDerivs(W,b,responses,means,uncertainties)
        print(dW,db)
        W -= dW*learning_rate
        #W /= .5*np.sum(abs(W))
        b -= db*learning_rate
        print(W,b)
    return W,b

def removeBkg(ims,depths,bkgs,bkgdepths,optimize=True, W0=None, b0=None):
    b0 = np.array([ 0.27433597,  0.18717923 , 0.17323259,  0.97169975, -0.10707744,  0.24528569])
    W0 = np.array([[ 0.575848,   0.09548454 , 0.22955949 , 0.78463556,  0.71672132 , 0.56989626],
                    [-0.23533103, -0.19698118 ,-0.04199131 ,-0.75826699, -0.75890685, -0.33609552]])

    if os.path.isfile('bk_mean.pkl'):
        bk_mean = pickle.load(open('bk_mean.pkl',"rb"))
        bk_cov = pickle.load(open('bk_cov.pkl','rb'))
        print(bk_mean)
        print(bk_cov)
    else:
        bk_mean, bk_cov = analyzeBackdrop('backdrop.png')
    if optimize:
        weights, biases = optimizeWeights(ims,depths,bkgs,bkgdepths, starting_weights=W0,starting_biases=b0,bk_mean=bk_mean,bk_cov=bk_cov)
        return weights, biases
    else:
        weights = W0
        biases = b0
    responses = getResponses(ims, depths, bkgs, bkgdepths, bk_mean, bk_cov)
    [means, uncertainties] = getOutput(responses, weights, biases)
    print(np.shape(means))
    mask = (np.argmin(means[:,:,0,:],axis=2)).astype(bool)
    print(np.shape(mask))
    for i in range(1):
        cv2.imshow("mask", (mask * 255).astype(np.uint8))
        cv2.waitKey(1)
        obj = np.ma.array(ims[:,:,:,i],mask=np.repeat(np.expand_dims(mask,2),3,axis=2))
        depth_obj = np.ma.array(depths[:,:,i],mask=mask)
    return obj.filled(0), depth_obj.filled(0)
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
'''
def getLoss(certainties):
    u = np.sum(certainties,axis=2)
    stddev = np.linalg.norm()

def optimizeOffsets(img,dist,weights):
'''

def analyzeBackdrop(path):
    bkdrop = cv2.imread(path)
    pix_inds = np.nonzero(~np.equal(bkdrop,[0,0,0]).all(axis=2))
    bkdrop = cv2.cvtColor(bkdrop,cv2.COLOR_BGR2HSV)/255
    hsv_avg = np.sum(bkdrop[pix_inds], axis=0)/len(pix_inds[0])
    hsv_cov = np.cov(np.reshape(bkdrop[pix_inds],(len(pix_inds[0]),3)).T)
    pickle.dump(hsv_avg, open("bk_avg.pkl", "wb"))
    pickle.dump(hsv_cov, open("bk_cov.pkl", "wb"))
    print(hsv_avg,hsv_cov)
    return  hsv_avg, hsv_cov

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
        cv2.waitKey(1)
    if depth_output_path is not None:
        cv2.imwrite(depth_output_path + f[0], bkg_thresh_depth)
        im = Image.open(depth_output_path + f[0])
        if (not self.is_crop_error(crop_top, crop_left, crop_bottom, crop_right, im)):
            im2 = im.crop((crop_left, crop_top, crop_right, crop_bottom))
            im2.save(depth_output_path + f[0])
        else:
            print("crop dimension error! showing original image")
    print("done!")