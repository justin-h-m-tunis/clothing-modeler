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
        if dist=='step':
            return ((I1 > lb) & (I1 < ub)).astype(float)
        elif dist=='linear':
            return sigmoid(np.min(np.dstack((I1-lb,ub-I1)),axis=2))
        else:
            I1 = I1 / 255
            return 1-chi2.cdf(mahalaobisDist(I1, mean, cov), 3)
    else:
        if dist=='step':
            return ((I1 < lb) | (I1 > ub)).all(axis=2).astype(float)
        elif dist=='linear':
            return sigmoid(np.max(np.dstack((lb-I1,I1-ub)),axis=2))
        else:
            I1 = I1/255
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
def getResponses(ims,depths,bkgs,bkgdepths,depthrange=(700,1270),bk_colors=None, man_colors=None, show_images=True):
    for i in range(3):
        ims[:,:,i,:] = gaussian_filter(ims[:, :, i, :], (1,1,0))
        bkgs[:,:,i,:] = gaussian_filter(bkgs[:,:,i,:],(1,1,0))
    depths[:, :, :] = gaussian_filter(depths[:, :, :], (1,1,0))
    bkgdepths[:, :, :] = gaussian_filter(bkgdepths[:, :, :], (1,1,0))

    num_filters = 6
    #depth_scale = np.max([np.max(depths),np.max(bkgdepths)])
    responses = np.zeros((len(ims),len(ims[0]),num_filters,np.shape(ims)[3]))
    print("computing responses")
    for i in range(np.shape(ims)[3]):
        hsv_ims = cv2.cvtColor(ims[:, :, :, i], cv2.COLOR_BGR2HSV)
        hsvdist = np.tanh(2*abs(hsv_ims - cv2.cvtColor(bkgs[:,:,:,i],cv2.COLOR_BGR2HSV))/255.0)
        rgbdist = np.tanh(2*abs(ims[:,:,:,i] - bkgs[:,:,:,i])/255.0)
        '''ddist = np.tanh(2*abs(depths[:,:,i].astype(float) - bkgdepths[:,:,i].astype(float))/depth_scale)
        cv2.imshow("ddist",(255*ddist).astype(np.uint8))
        depth_mask = thresholdFn(depths[:,:,i],lb=depthrange[0],ub=depthrange[1])
        cv2.imshow("d mask",(255*depth_mask).astype(np.uint8))
        backdrop_mask = thresholdFn(hsv_ims,mean=bk_colors[0],cov=bk_colors[1], direction='exclusive', dist='normal')
        print(backdrop_mask)
        cv2.imshow("bkg mask",(255*backdrop_mask).astype(np.uint8))
        mannequin_mask = thresholdFn(hsv_ims, mean=man_colors[0], cov=man_colors[1], direction='exclusive', dist='normal')
        print(mannequin_mask)
        cv2.imshow("man mask", (255 * mannequin_mask).astype(np.uint8))'''
        if show_images:
            cv2.imshow("ims", ims[:, :, :, i])
            cv2.imshow("bks", bkgs[:, :, :, i])
            cv2.imshow("hsv", (255 * hsvdist).astype(np.uint8))
            cv2.imshow("rgb", (255 * rgbdist).astype(np.uint8))
            cv2.waitKey(1)

        responses[:,:,:,i] = np.dstack((hsvdist,
                                       rgbdist,
                                       #backdrop_mask,
                                       #mannequin_mask,
                                       #depth_mask
                                    ))
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
def getOutput(responses,weights,biases, show_images=True):
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
    if show_images:
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

def getDerivs(weights, biases, responses, output, uncertainties, ignore=None):
    print("Determining Gradients")
    n = np.shape(responses)[3]
    h = np.shape(responses)[0]
    w = np.shape(responses)[1]
    W_ = np.sum(weights,axis=1)
    channels = np.shape(responses)[2]
    L = calculateLoss(output,uncertainties)
    print(np.max(responses),np.max(output),np.min(output),np.max(uncertainties),np.min(L))

    if ignore is not None:
        pix_inds = np.nonzero(~np.equal(output,ignore).all(axis=3))
        dLdf = L[pix_inds]*(1-L[pix_inds]) #1xDDn #1x1
        dfdu = 2*np.equal(output[pix_inds],np.tile(np.expand_dims(np.max((output[pix_inds]),axis=3),3),(1,1,1,2)),dtype=float)-1 #DDnx2 #1x2
        dfds = np.array([-1,-1]) # 1x2
        x = np.swapaxes(responses[pix_inds[0],pix_inds[1],:,pix_inds[2]],2,3) - biases
        x_ = np.sum(x,axis=(3))/channels
        x_diff = np.moveaxis(np.tile(np.moveaxis(np.moveaxis(x,3,0) - x_,0,3),(2,1,1,1,1)),0,-1)
        dudw = (x-biases)/n #DDnxf #1xf
        dsdw = np.moveaxis(n/(n-1)*np.moveaxis(weights.T * (x_diff ** 2),3,0)/((uncertainties[pix_inds]+.0001)*W_**2),0,3) #DDnx2xf, 2xf
        dudb = -weights #2xf
        dsdb = -np.moveaxis(n/(n-1)*np.moveaxis(weights.T**2 * x_diff * (1-1/channels),3,0)/((uncertainties[pix_inds]+.0001)*W_**2),0,3) #DDnx2xf, 2xf
        print(np.min(dsdb),np.max(dsdb),np.min(abs(uncertainties)))
    else:
        dLdf = L*(1-L) #1xDDn #1x1
        dfdu = 2*np.equal(output,np.tile(np.expand_dims(np.max((output),axis=3),3),(1,1,1,2)),dtype=float)-1 #DDnx2 #1x2
        dfds = np.array([-1,-1]) # 1x2
        x = np.swapaxes(responses,2,3) - biases
        x_ = np.sum(x,axis=(3))/channels
        x_diff = np.moveaxis(np.tile(np.moveaxis(np.moveaxis(x,3,0) - x_,0,3),(2,1,1,1,1)),0,-1)
        dudw = (x-biases)/n #DDnxf #1xf
        dsdw = np.moveaxis(n/(n-1)*np.moveaxis(weights.T * (x_diff ** 2),3,0)/((uncertainties+.0001)*W_**2),0,3) #DDnx2xf, 2xf
        dudb = -weights #2xf
        dsdb = -np.moveaxis(n/(n-1)*np.moveaxis(weights.T**2 * x_diff * (1-1/channels),3,0)/((uncertainties+.0001)*W_**2),0,3) #DDnx2xf, 2xf
        print(np.min(dsdb),np.max(dsdb),np.min(abs(uncertainties)))

        dLdf = np.reshape(dLdf,(1,h*w*n))
        dfdu = np.reshape(dfdu,(h*w*n,2))
        dudw = np.reshape(dudw, (h * w * n, channels))
        dsdw = np.reshape(np.swapaxes(dsdw,3,4), (h * w * n, 2, channels))
        dsdb = np.reshape(np.swapaxes(dsdb,3,4), (h*w*n,2,channels))

    #print(np.max(dLdf),np.max(dfdu),np.max(dfds),np.max(dudw),np.max(dsdw),np.max(dudb),np.max(dsdb))

    grad_w = ((dLdf.T*dfdu).T @ dudw + np.sum((dLdf.T @ np.expand_dims(dfds,0)) * np.moveaxis(dsdw,2,0),axis=1).T)/np.size(dLdf)
    grad_b = (dLdf @ (dfdu @ dudb) - dLdf @ np.sum(dsdb,axis=1))/np.size(dLdf)
    return np.reshape(grad_w,(2,channels)),np.reshape(grad_b,(channels))

 ## not terrible weights:[-0.23539651 -0.20197062 -0.03738734 -0.9411517  -0.77748915 -0.65677627
 # -0.32563077]] [ 0.34155395  0.20720649  0.19224365  1.29058724 -0.0726096   0.01480159  0.31051498]
def optimizeWeights(ims,depths,bkgs,bkgdepths, epochs=1,learning_rate = .2, starting_weights=[], starting_biases=[]):
    in_size = 6
    out_size = 2
    if len(starting_weights) == 0:
        '''W = np.random.uniform(0, np.sqrt(6 / (in_size + out_size)), out_size * in_size)
        W = np.reshape(W, (out_size, in_size))
        W = (W.T / np.sum(W, axis=1)).T
        W[1, :] *= -1'''
        W = np.array([[0.,0.,1.,.7,.7,.7],[-0.,0.,1.,-.7,-.7,-.7]])
    else:
        W = starting_weights
    if len(starting_biases) == 0:
        b = np.ones(in_size) * .1
    else:
        b = starting_biases
    print(W,b)
    responses = getResponses(ims, depths, bkgs, bkgdepths, bk_colors=analyzeBackdrop('backdrop.png'), man_colors=analyzeBackdrop('mannequin.png'))
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
    print("done optimizing!")
    return W,b

def gaussianThinning(im, kernel_size, passes=1, thresh=.45):
    dim3 = int(np.shape(im)[0]*np.shape(im)[1]/np.size(im))
    for i in range(passes):
        im = cv2.GaussianBlur(im.astype(float),(kernel_size,kernel_size),0)
        im = (im > thresh).astype(bool)
    return im

def removeBkg(ims,depths,bkgs,bkgdepths,optimize=True, W0=None, b0=None):
    if os.path.isfile('bk_mean.pkl'):
        bk_mean = pickle.load(open('bk_mean.pkl',"rb"))
        bk_cov = pickle.load(open('bk_cov.pkl','rb'))
        print(bk_mean)
        print(bk_cov)
    else:
        bk_mean, bk_cov = analyzeBackdrop('backdrop.png')
    if optimize:
        weights, biases = optimizeWeights(ims,depths,bkgs,bkgdepths, starting_weights=W0,starting_biases=b0)
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

def applyMask(im, mask, dims):
    if dims==1:
        obj = np.ma.array(im, mask=mask)
        return obj.filled(0)
    else:
        obj = np.ma.array(im, mask=np.repeat(np.expand_dims(mask, 2), dims, axis=2))
        return obj.filled(0)

def getThresholdMask(im,depth,bk_path, man_path, depth_range, bk_confidence=.4, man_confidence=.3,apply_mask=False):
    print("Thresholding Image")
    hsv_im = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    depth_mask = thresholdFn(depth, lb=depth_range[0], ub=depth_range[1], dist='step')
    bk_colors = analyzeBackdrop(bk_path)
    backdrop_mask = thresholdFn(hsv_im, mean=bk_colors[0], cov=bk_colors[1], dist='normal')
    man_colors = analyzeBackdrop(man_path)
    mannequin_mask = thresholdFn(hsv_im, mean=man_colors[0], cov=man_colors[1], dist='normal')

    mask = ~(depth_mask.astype(bool) & (backdrop_mask < bk_confidence).astype(bool) & (
                mannequin_mask < man_confidence).astype(bool))
    return mask

def removeBackgroundThreshold(im,depth, bk, depth_bk, depth_range, bk_path, man_path, bk_confidence=.4, man_confidence=.3, bk_weights=[],bk_biases=[], blur_kernel=45, blur_passes=3, blur_thresh=.43,  show_images=True):
    #for i in range(3):
        #im = cv2.GaussianBlur(im,(5,5),0)
    #depth = cv2.GaussianBlur(depth, (5,5),0)

    mask = getThresholdMask(im,depth,bk_path, man_path,depth_range=depth_range, bk_confidence=bk_confidence, man_confidence=man_confidence)
    if len(bk_weights) > 0 and len(bk_biases) > 0:
        responses = getResponses(np.expand_dims(im, 3), np.expand_dims(depth, 2), np.expand_dims(bk, 3),
                                 np.expand_dims(depth_bk, 2), show_images=False)
        means, _ = getOutput(responses, bk_weights, bk_biases, show_images=False)
        means_mask = np.argmin(means[:, :, 0, :], axis=2).astype(bool)
        mask = ~(~mask & means_mask)
    thinned_mask = gaussianThinning(mask, blur_kernel, passes=blur_passes, thresh=blur_thresh)
    if show_images:
        #cv2.imshow("d mask", (255 * depth_mask).astype(np.uint8))
        #cv2.imshow("bkg mask", (255 * (backdrop_mask > bk_confidence)).astype(np.uint8))
        #cv2.imshow("mannequin mask", (255 * (mannequin_mask > man_confidence)).astype(np.uint8))
        #cv2.imshow("optimizer mask", (255 * means_mask).astype(np.uint8))
        cv2.imshow("thinned mask", thinned_mask.astype(float))
        cv2.waitKey(1)
    obj = np.ma.array(im, mask=np.repeat(np.expand_dims(thinned_mask, 2), 3, axis=2))
    depth_obj = np.ma.array(depth, mask=thinned_mask)
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