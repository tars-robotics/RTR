'''
A common class of tool functions for gelsight publisher
Consist of parameter settings and marker detection 
'''

import cv2 
import numpy as np
from scipy.ndimage import gaussian_filter
from shapely.geometry import Polygon

class GelsightUtility:
    '''
    tool functions class for gelsight publisher
    '''
    def __init__(
        self,
        RESCALE=1,
        N_=7,
        M_=9,
        fps_=30,
        x0_=143,
        y0_=108,
        dx_=42,
        dy_=46,
        camera_type='Bnz',
        camera_dimension=2
    ):
        '''
        N_, M_: the row and column of the marker array
        x0_, y0_: the coordinate of upper-left marker (in original size)
        dx_, dy_: the horizontal and vertical interval between adjacent markers (in original size)
        fps_: the desired frame per second, the algorithm will find the optimal solution in 1/fps seconds
        '''
        self.RESCALE = RESCALE
        self.N = N_
        self.M = M_
        self.fps = fps_
        self.x0 = x0_ / RESCALE
        self.y0 = y0_ / RESCALE
        self.dx = dx_ / RESCALE
        self.dy = dy_ / RESCALE
        
        self.camera_type = camera_type
        self.camera_dimension = camera_dimension
        
    def img_initiation(self, img):
        '''
        Input:
        an original frame
        Output:
        a frame initiated by setting parameters
        '''
        if self.camera_type == 'Bnz':
            return cv2.resize(img, (0, 0), fx=1.0/self.RESCALE, fy=1.0/self.RESCALE)
        elif self.camera_type == 'Hsr':
            # Resize image for HSR camera and apply distortion correction and perspective warp
            return self.init_HSR(img)
        
    def init_HSR(self, img):
        '''
        Initialize image for HSR camera type:
        - Undistort the image using the fisheye model.
        - Apply a perspective transformation.
        '''
        DIM = (640, 480)
        img = cv2.resize(img, DIM)
        
        # Camera matrix (intrinsic parameters) and distortion coefficients
        K = np.array([[225.57469247811056, 0.0, 280.0069549918857],
                      [0.0, 221.40607131318117, 294.82435570493794],
                      [0.0, 0.0, 1.0]])
        D = np.array([[0.7302503082668154], [-0.18910060205317372], 
                      [-0.23997727800712282], [0.13938490908400802]])
        
        h, w = img.shape[:2]
        
        # Generate undistortion maps
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
        undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        
        return self.warp_perspective(undistorted_img)
        
    def find_marker(self, frame):
        '''
        Find markers in the given frame and return a binary mask where markers are 255, otherwise 0.
        '''
        markerThresh = -5
        img_gaussian = np.int16(cv2.GaussianBlur(frame, (int(63/self.RESCALE), int(63/self.RESCALE)), 0))
        I = frame.astype(np.double) - img_gaussian.astype(np.double)
        
        # This is the mask of the markers
        markerMask = ((np.max(I, 2)) < markerThresh).astype(np.uint8)
        
        # Smooth the marker regions to reduce noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(15/self.RESCALE), int(15/self.RESCALE)))
        markerMask = cv2.morphologyEx(markerMask, cv2.MORPH_CLOSE, kernel)
        
        M, N = markerMask.shape
        w_margin = int(N // 2 - N // 2.5)
        h_margin = int(M // 2 - M // 2.5)
        top = h_margin
        bottom = M - h_margin
        left = w_margin
        right = N - w_margin
        
        markerMask[:top, :] = 0
        markerMask[bottom:, :] = 0
        markerMask[:, :left] = 0
        markerMask[:, right:] = 0
        
        mask = markerMask * 255
        return mask
        
    def marker_center(self, mask): 
        '''
        Detect the center of the markers in the mask and return their coordinates.
        Filter out small or irregularly shaped markers.
        '''
        areaThresh1 = 50 / self.RESCALE ** 2
        areaThresh2 = 1920 / self.RESCALE ** 2
        MarkerCenter = []

        contours = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours[0]) < 25:  # If too few markers, return an empty list
            print("Too few markers detected:", len(contours))
            return MarkerCenter
        
        for contour in contours[0]:
            x, y, w, h = cv2.boundingRect(contour)
            AreaCount = cv2.contourArea(contour)
            
            # Filter markers based on area and aspect ratio
            if areaThresh1 < AreaCount < areaThresh2 and abs(np.max([w, h]) / np.min([w, h]) - 1) < 2:
                t = cv2.moments(contour)
                if self.camera_dimension == 2:
                    mc = [t['m10'] / t['m00'], t['m01'] / t['m00']]
                elif self.camera_dimension == 3: 
                    mc = [t['m10'] / t['m00'], t['m01'] / t['m00'], 0]
                MarkerCenter.append(mc) # type: ignore
        
        # Return the coordinates of detected marker centers
        return MarkerCenter
    
    def ComputesurroundingArea(self, Cx, Cy):
        '''
        Calculate the surrouding area of each marker based on positions of current markers
        Use square of diagnols to represent areas for sake of efficiency
        edge makers and corner markers are considered independently
        return the surrounding area of each marker
        '''
        Area = [[0.] * self.M for _ in range(self.N)] 

        # compute square distances of inner markers 
        for i in range(self.N - 1):
            for j in range(self.M - 1):
                points = [(Cx[i][j], Cy[i][j]), (Cx[i+1][j], Cy[i+1][j]), (Cx[i][j+1], Cy[i][j+1]), (Cx[i+1][j+1], Cy[i+1][j+1])]
                area = Polygon(points).area
                Area[i][j] += area
                Area[i][j + 1] += area
                Area[i + 1][j] += area
                Area[i + 1][j + 1] += area

        # deal with markers on the edges
        for i in range(1, self.M - 1):  # upper edge
            Area[0][i] *= 2
            Area[self.N - 1][i] *= 2  # bottom edge
        
        for i in range(1, self.N - 1):  # left edge
            Area[i][0] *= 2
            Area[i][self.M - 1] *= 2  # right edge

        # deal with markers on the corners
        Area[0][0] *= 4  # up-left corner
        Area[0][self.M - 1] *= 4  # up-rught corner
        Area[self.N - 1][0] *= 4  # bottom-left corner
        Area[self.N - 1][self.M - 1] *= 4  # bottom-right corner
        
        Area = gaussian_filter(Area, sigma=1)
        
        return Area
        
        
    def warp_perspective(self, img):
        '''
        Apply a fixed perspective transformation to the image.
        Defined by 4 points (TOPLEFT, TOPRIGHT, BOTTOMLEFT, BOTTOMRIGHT).
        '''
        TOPLEFT = (175, 230)
        TOPRIGHT = (380, 225)
        BOTTOMLEFT = (10, 410)
        BOTTOMRIGHT = (530, 400)
        
        WARP_W, WARP_H = 215, 215
        
        points1 = np.float32([TOPLEFT, TOPRIGHT, BOTTOMLEFT, BOTTOMRIGHT])
        points2 = np.float32([[0, 0], [WARP_W, 0], [0, WARP_H], [WARP_W, WARP_H]])
        
        matrix = cv2.getPerspectiveTransform(points1, points2)
        result = cv2.warpPerspective(img, matrix, (WARP_W, WARP_H))
        
        return result
