'''
A common class of tool functions for MC_Tac publisher
Consist of parameter settings and marker detection 
'''

import cv2 
import numpy as np
from shapely.geometry import Polygon
from scipy.ndimage import gaussian_filter

class MCTacUtility:
    '''
    tool functions class for gelsight publisher
    '''
    def __init__(
        self,
        RESCALE=1,
        N_=5,
        M_=5,
        fps_=30,
        x0_ = 149,
        y0_ = 67,
        dx_ = 85,
        dy_ = 83,
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
        return cv2.resize(img, (0, 0), fx=1.0/self.RESCALE, fy=1.0/self.RESCALE)
    
    def rotate_points(self, mc, angle):
        '''
        rotate the marker center array instead of the whole image
        Input:
        mc: M * N marker center array
        angle: rotation angle
        OUtput:
        rotated_mc: M * N marker center array after rotation 
        '''
        angle_radians = np.radians(angle)
        cos_theta = np.cos(angle_radians)
        sin_theta = np.sin(angle_radians)
    
        # Rotation matrix
        R = np.array([
            [cos_theta, sin_theta],
            [-sin_theta, cos_theta]
        ])

        # Apply rotation to each point
        rotated_mc = np.dot(mc, R.T)
        return rotated_mc
    
    def rotate_image(self, image, angle):
    
        # the mctac camera needs rotation to cater to tracking_CLASS
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h))
        return rotated
        
    def find_marker(self, frame):
        '''
        Find markers in the given frame and return a binary mask where markers are 255, otherwise 0.
        '''
        markerThresh = 100
        # img_gaussian = np.int16(cv2.GaussianBlur(frame, (int(63/self.RESCALE), int(63/self.RESCALE)), 0))
        # I = frame.astype(np.double) - img_gaussian.astype(np.double)

        I = frame.astype(np.double)
        
        # This is the mask of the markers
        markerMask = ((np.max(I, 2)) < markerThresh).astype(np.uint8)

        # cv2.imshow('markerMask', markerMask * 255)
        # cv2.waitKey(1)
        
        # Smooth the marker regions to reduce noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(15/self.RESCALE), int(15/self.RESCALE)))
        markerMask = cv2.morphologyEx(markerMask, cv2.MORPH_CLOSE, kernel)
        
        M, N = markerMask.shape
        w_margin = int(N // 2 - N // 2.4)
        h_margin = int(M // 2 - M // 2.4)
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
        
        if len(contours[0]) <20:  # If too few markers, return an empty list
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
                    # TODO: deal with 3d marker extractor 
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
        Area[self.N - 1][0] *= 4  # bottom-left cornercd
        Area[self.N - 1][self.M - 1] *= 4  # bottom-right corner
        
        # Use Gaussian smoothig to smooth vertical offset in neighbouring areas
        Area = gaussian_filter(Area, sigma=1)
        
        # # Apply edge weighting using Gaussian-like decay based on distance to edges
        # for i in range(self.N):
        #     for j in range(self.M):
        #         dist_to_edge = min(i, self.N - 1 - i, j, self.M - 1 - j)
        #         weight = np.exp(-dist_to_edge**2 / (2 * 1.0**2))  # 1.0 is the effective sigma for edge decay
        #         Area[i][j] *= weight
                
        return Area