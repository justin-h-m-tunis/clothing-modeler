import cv2
import numpy as np
import scipy.optimize
import os

def getBkgDistances(img, im_depth, bkg, bkg_depth,gaussian_kernel=(5,5)):
    im_blur = cv2.GaussianBlur(img,gaussian_kernel,0)
    bkg_blur = cv2.GaussianBlur(bkg, gaussian_kernel, 0)
    rgb_dist = np.linalg.norm(im_blur - bkg_blur,axis=2)/255*np.sqrt(3)
    hue_dist = abs(cv2.cvtColor(im_blur,cv2.COLOR_RGB2HSV)[:,:,0] - cv2.cvtColor(bkg_blur,cv2.COLOR_RGB2HSV)[:,:,0])/255
    depth_dist = np.linalg.norm(im_depth - bkg_depth,axis=2)/(np.sqrt(3)*np.max([np.max(im_depth),np.max(im_depth)]))
    return np.stack((rgb_dist, hue_dist, depth_dist),axis=2)

#certainty bounds [bkg cutoff, bkg cutoff]
#certainty offset how the distance of the parameter skews certainty
#guess_channel_logic_table

def removeBkg(img,dist,weights=(1,1,.5),certainty_bounds=(0,0), certainty_offsets=(.95,.4,0), guessFn=lambda x: x[np.argmax(abs(x))] > 0,gaussian_kernel=(5,5)):
    certainties = np.multiply(dist-np.array(certainty_offsets),np.array(weights))
    cv2.imshow("positive certainties",cv2.cvtColor(((certainties).clip(min=0)*255).astype(np.uint8),cv2.COLOR_BGR2RGB))
    cv2.imshow("negative certainties",cv2.cvtColor(((certainties).clip(max=0)*-255).astype(np.uint8),cv2.COLOR_BGR2RGB))
    cv2.imwrite("certainties5.png",cv2.cvtColor(((certainties).clip(min=0)*255).astype(np.uint8),cv2.COLOR_BGR2RGB))
    print(np.sum(certainties, axis=2) < certainty_bounds[0])
    mask_bkg = (np.sum(certainties, axis=2) < certainty_bounds[0]).astype(bool)
    mask_obj = (np.sum(certainties, axis=2) > certainty_bounds[1]).astype(bool)
    uncertainties = ~ mask_obj ^ mask_bkg
    if np.sum(uncertainties.astype(int))==0:
        mask = ~(cv2.GaussianBlur((mask_obj).astype(np.uint8),gaussian_kernel,0).astype(bool))
        cv2.imshow("mask", mask.astype(np.uint8)*255)
        obj = np.ma.array(img,mask=np.repeat(mask,np.size(img,axis=2)))
        return obj.filled(0)
    else:
        pass

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