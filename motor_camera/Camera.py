from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime
import numpy as np
import cv2
import scipy.optimize

def skew(v):
    return np.array([[0, -v[2], v[1]],[v[2], 0, -v[0]],[-v[2],v[0],0]])

def interpolateFloatPixel(A,f_x,f_y, order=2): #xy
    N = np.array([[[np.floor(f_x),np.ceil(f_x)],[np.floor(f_x),np.ceil(f_x)]],[[np.floor(f_y), np.floor(f_y)],[np.ceil(f_y),np.ceil(f_y)]]]).astype(int)
    A_N = A[N[1,:,:],N[0,:,:]]
    x_dec = f_x % 1
    y_dec = f_y % 1
    return (A_N[1,1] - A_N[0,1] - A_N[1,0] + A_N[0,0])*x_dec*y_dec + A[0,0]

def determineMatch(x1,x2,P1,P2,A, bounds, interp=False): #xy
    if interp:
        if (np.array([x1,x2]) - bounds[0,:,:] < 0).any() or (bounds[1,:,:] - np.array([x1,x2]) < 0).any():
            return np.inf
        z = interpolateFloatPixel(A,x2[0],x2[1])
    else:
        z = A[x2[1],x2[0]]
    if z == 0:
        return np.inf
    p1 = P1[:,[0,1,3]]
    p2 = P2[:,[0,1,3]]
    x1_ = np.array([x1[0],x1[1],1])
    x2_ = np.array([x2[0],x2[1],1])
    dist = np.linalg.norm(p1 @ np.linalg.inv(p2) @ (x1_-P1[:,2]*z) - (x2_ - P2[:,2]*z))
    return dist

def getAtoBCorrespondence(ax,ay,P_a,P_b,z):
    p_a = P_a[:, [0, 1, 3]]
    p_inv = np.linalg.inv(p_a)
    A = np.array([ax, ay, 1]).T
    if not (p_a[2,0:2] == [0, 0]).all():
        X = np.linalg.inv(p_a - skew(np.cross(p_a[2,:],A)) - np.eye(3)*(p_a[2,:] @ A)) @ (P_a[2,2]*A - P_a[:,2])*z
    else:
        X = p_inv @ (A + (P_a[2,2]*A - P_a[:,2])*z)
    print(X)
    #print(P_a @ np.array([X[0]/X[2],X[1]/X[2],z,1]).T)
    B = P_b @ np.array([X[0],X[1],z,1])
    b = np.array([B[0]/B[2],B[1]/B[2]])
    return b

def getRGBtoDepthCorrespondence(i,j,F,P_rgb,P_d,depthFrame,res_rgb,res_d):
                l = F @ np.array([i, j, 1]).T
                p0 = -l[2] / l[1]
                v = -l[0]/l[1]
                y = lambda x:p0 + v*x
                if(y(0) < 0) :
                    xmin = -p0/v
                else:
                    xmin = 0
                if(y(res_d[1]) < 0):
                    xmax = -p0/v
                else:
                    xmax = res_d[1]-1
                c_min = np.inf
                x_best = -1
                for r in range(int(xmin),int(xmax)):
                    cost = determineMatch(np.array([i, j]),
                                           np.array([r, int(y(r))]),
                                           P_rgb,
                                           P_d,
                                           depthFrame,
                                           np.array([[[0,0],[0,0]],[[res_rgb[1], res_rgb[0]],[res_d[1],res_d[0]]]]))
                    if cost < c_min:
                        x_best = r
                if x_best == -1:
                    depth_image[i,j] = 0;
                    depth_corresp= [-1,-1]
                else:
                    depth_corresp = [int(round(y(x_best))),int(round(x_best))]
                return depth_corresp

def getCropScaleFromK(dim1,dim2,K1,K2,c_p):
    scale_x = K1[0,0]/K2[0,0]
    scale_y = K1[1,1]/K2[1,1]
    shift_1 = (c_p[0,:]-K1[0:2,2])
    shift_2 = np.multiply(c_p[1,:]-K2[0:2,2],[scale_x,scale_y])
    crop_x = int(min(dim2[1]*scale_x, dim1[1]) - max(abs(shift_1[0]),abs(shift_2[0]),abs(shift_1[0] - shift_2[0])))
    crop_y = int(min(dim2[0]*scale_y, dim1[0]) - max(abs(shift_1[1]),abs(shift_2[1]),abs(shift_1[1] - shift_2[1])))
    return [scale_x, scale_y, crop_x, crop_y,shift_1, shift_2]

def reshapeImage(I, scale_x, scale_y, crop_x, crop_y, shift):
    h = len(I)
    w = len(I[0])
    h_new = int(h*scale_y)
    w_new = int(w*scale_x)
    shift_x = int(shift[0])
    shift_y = int(shift[1])
    bound_x = np.array([(w_new-crop_x + shift_x)/2,(w_new+crop_x + shift_x)/2]).astype(int)
    bound_y = np.array([(h_new-crop_y + shift_y)/2,(h_new+crop_y + shift_y)/2]).astype(int)
    I = cv2.resize(I,(int(w_new),int(h_new)))
    I = I[bound_y[0]:bound_y[1],bound_x[0]:bound_x[1]]
    return I

class Camera:
    def __init__(self):
        self.kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Depth | PyKinectV2.FrameSourceTypes_Color)
        self.res_d = (424, 512)
        self.res_rgb = (1080, 1920)
        T = np.array([-.06015, .00221, .02714]).T * 1.000
        T_hat = np.array([[0, -T[2], T[1]], [T[2], 0, -T[0]], [-T[1], T[0], 0]])
        R = np.array([[.99997, .00715, -.00105], [-.00715, .99995, .00662], [.00110, -.00661, .99998]])
        K_rgb = np.array([[1144.361, 0, 966.359], [0, 1147.337, 548.038], [0, 0, 1]])
        K_d = np.array([[388.198, 0, 253.270], [0, 389.033, 213.934], [0, 0, 1]])

        F = np.linalg.inv(K_d).T @ T_hat @ R @ np.linalg.inv(K_rgb)
        P_rgb = K_rgb @ np.concatenate((np.eye(3), np.reshape(-1* T, (3, 1))), axis=1)
        P_d = K_d @ np.concatenate((np.eye(3), np.reshape(0*T, (3, 1))), axis=1)
        print(P_d)
        print(P_rgb)
        self.focus_depth = .889
        center_points = np.array([P_rgb @ np.array([0, 0, self.focus_depth, 1]), P_d @ np.array([0, 0, self.focus_depth, 1])])
        print(center_points)
        center_points = np.array(
            [center_points[0, :2] / center_points[0, 2], center_points[1, :2] / center_points[1, 2]])
        [self.scale_x, self.scale_y, self.crop_x, self.crop_y, self.shift_rgb, self.shift_d] = getCropScaleFromK(self.res_rgb, self.res_d, K_rgb, K_d,
                                                                                   center_points)

    def captureRGBD(self,num,show_image=True, path='data/'): #will return False when frames are not available
        if not(self.kinect.has_new_color_frame() and self.kinect.has_new_depth_frame()):
            return False
        depthFrame = self.kinect.get_last_depth_frame()
        depthFrame = np.reshape(depthFrame, (424, 512))
        colorFrame = self.kinect.get_last_color_frame()
        colorFrame = np.reshape(colorFrame, (1080, 1920, 4))
        # colorFrame = colorFrame[:,int(crop[0,0,0]):int(crop[0,1,0])]
        colorFrame = reshapeImage(colorFrame, 1, 1, self.crop_x, self.crop_y, self.shift_rgb)
        # depthFrame = depthFrame[int(crop[1,0,1]):int(crop[1,1,1]),:]
        # depth_image = cv2.resize(depthFrame,(len(colorFrame[0]),len(colorFrame)))
        depth_image = reshapeImage(depthFrame, self.scale_x, self.scale_y, self.crop_x, self.crop_y, self.shift_d)
        if not cv2.imwrite(path + 'color/color_' + str(num) + '.png', colorFrame[:, :, 0:3]):
            raise Exception("could not save picture")
        if not cv2.imwrite(path + 'depth/Depth_' + str(num) + '.png', depth_image.astype(np.uint16)):
            raise Exception("could not save picture")
        if show_image:
            cv2.imshow('color', colorFrame)
            cv2.imshow('depth', depth_image)
            I = cv2.cvtColor(((1 - depth_image / np.amax(depth_image)) * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
            colorFrame = np.multiply(colorFrame[:, :, 0:3], I / 255.).astype(np.uint8)
            cv2.imshow('color x depth', cv2.line(colorFrame, (int(len(colorFrame[0]) / 2), 0),
                                    (int(len(colorFrame[0]) / 2), len(colorFrame[0])), (0, 0, 255)))
            cv2.waitKey(1)
        print("Images captured!")
        return True
