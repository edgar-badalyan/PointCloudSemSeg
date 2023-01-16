import numpy as np
from detectron2.data import transforms as T
import cv2


class PerspectiveTransform(T.Transform):
    """
    Transform to apply homography (perspective transform)
    """
    def __init__(self, xs, ys, H, W):
        self.xs = xs
        self.ys = ys
        
        self.H = H
        self.W = W
        
        points_original = np.float32([[0, 0],
                                   [0, H],
                                   [W, H],
                                   [W, 0]])
                           
        points_warped = np.float32([[xs[0], ys[0]],
                                [xs[1], H-ys[1]],
                                [W-xs[2], H-ys[2]],
                                [W-xs[3], ys[3]]])
        
        self.M_affine = cv2.getPerspectiveTransform(points_original, points_warped)

    def apply_image(self, image):

        new_image = cv2.warpPerspective(image, self.M_affine, (self.H, self.W))
        
        return new_image
     
    def apply_coords(self, coords):
        new_coords = []

        for point in coords:
            new_point = np.array([point[0], point[1], 1])
            new_point = self.M_affine @ new_point
            
            new_coords.append([new_point[0]/new_point[2], new_point[1]/new_point[2]])
            
        return np.array(new_coords)
            
    def apply_segmentation(self, segmentation):
        return self.apply_image(segmentation)
        

class PerspectiveAugmentation(T.Augmentation):
    def __init__(self, f: float):
        self.f = f
        
    def get_transform(self, image, boxes, sem_seg):
        H, W = image.shape[:2]
        
        xs = np.random.randint(0, int(self.f*W), size=4)
        ys = np.random.randint(0, int(self.f*H), size=4)
        
        return PerspectiveTransform(xs, ys, H, W)
