from typing import Tuple

import numpy as np
import cv2
from scipy.ndimage import label
import rospy

rospy.set_param('left_gain', 1)
rospy.set_param('right_gain', 1)


def get_steer_matrix_left_lane_markings(shape: Tuple[int, int]) -> np.ndarray:
    """
    Args:
        shape:              The shape of the steer matrix.

    Return:
        steer_matrix_left:  The steering (angular rate) matrix for reactive control
                            using the masked left lane markings (numpy.ndarray)
    """
    H, W = shape
    steer_matrix_left = np.ones(shape)

    # # Create the reference lane marking masks with lines
    # left_lane_ref = np.zeros((H, W), dtype=np.uint8)

    # # Specify the reference angles for the lane
    # left_lane_ref_degree = 45 * np.pi / 180
    # thickness_line = 40
    # scaling_factor = 1 / 10000

    # # Draw the reference lines
    # cv2.line(left_lane_ref, (0, H), (int(H / np.tan(left_lane_ref_degree)), 0), 255, thickness_line)

    # # Find the regions on each side of the reference lines
    # line_mask = left_lane_ref == 255
    # labeled_array_left, num_features = label(~line_mask)  # Invert the line mask for regions

    # # do distance transform
    # dist_left_line = cv2.distanceTransform(~left_lane_ref, cv2.DIST_L2, 5)
    # dist_left_line[labeled_array_left == 2] *= -1

    # # ignore half of the image
    steer_matrix_left[:, W // 2:] = 0

    # # normalize by the max value
    # steer_matrix_left = dist_left_line / np.max(np.abs(dist_left_line)) * scaling_factor
    # max_value = np.percentile(np.abs(steer_matrix_left), q=95)
    # steer_matrix_left[steer_matrix_left > max_value] = max_value
    # steer_matrix_left[steer_matrix_left < -max_value] = -max_value

    return steer_matrix_left * -1 * rospy.get_param('left_gain') # -1 is right turn and +1 is left turn


def get_steer_matrix_right_lane_markings(shape: Tuple[int, int]) -> np.ndarray:
    """
    Args:
        shape:               The shape of the steer matrix.

    Return:
        steer_matrix_right:  The steering (angular rate) matrix for reactive control
                             using the masked right lane markings (numpy.ndarray)
    """

    H, W = shape

    steer_matrix_right = np.ones(shape)

    # # Create the reference lane marking masks with lines
    # right_lane_ref = np.zeros((H, W), dtype=np.uint8)

    # # Specify the reference angles for the lane
    # right_lane_ref_degree = 45 * np.pi / 180
    # thickness_line = 40
    # scaling_factor = 1 / 10000

    # # Draw the reference lines
    # cv2.line(right_lane_ref, (W, H), (int(W - H / np.tan(right_lane_ref_degree)), 0), 255, thickness_line)

    # # Find the regions on each side of the reference lines
    # line_mask = right_lane_ref == 255
    # labeled_array_right, num_features = label(~line_mask)  # Invert the line mask for regions

    # # do distance transform
    # dist_right_line = cv2.distanceTransform(~right_lane_ref, cv2.DIST_L2, 5)

    # dist_right_line[labeled_array_right == 2] *= -1

    # # ignore half of the image
    steer_matrix_right[:, :W // 2] = 0

    # # normalize by the mean value
    # steer_matrix_right = dist_right_line / np.max(np.abs(dist_right_line)) * scaling_factor
    # max_value = np.percentile(np.abs(steer_matrix_right), q=95)
    # steer_matrix_right[steer_matrix_right > max_value] = max_value
    # steer_matrix_right[steer_matrix_right < -max_value] = -max_value

    return steer_matrix_right * rospy.get_param('right_gain') * 0.7

def detect_lane_markings(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Args:
        image: An image from the robot's camera in the BGR color space (numpy.ndarray)
    Return:
        mask_left_edge:   Masked image for the dashed-yellow line (numpy.ndarray)
        mask_right_edge:  Masked image for the solid-white line (numpy.ndarray)
    """
    h, w, _ = image.shape

    ####### params
    threshold_left = 5  # gradient magnitude threshold
    threshold_right = 20
    sigma = 8       # gaussian blurring threshold
    # white_lower_hsv = np.array([35, 0, 0])
    # white_upper_hsv = np.array([179, 70, 100])
    
    white_lower_hsv = np.array([0, 0, 200]) # 0 0 150
    white_upper_hsv = np.array([179, 50, 255])  # 179 80 255

    yellow_lower_hsv = np.array([10, 80, 40])
    yellow_upper_hsv = np.array([70, 255, 255])
    # yellow_lower_hsv = np.array([0, 0, 0])
    # yellow_upper_hsv = np.array([179, 255, 255])

    # Convert the image to HSV for any color-based filtering
    imghsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Most of our operations will be performed on the grayscale version
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    ###### blurring
    # Smooth the image using a Gaussian kernel
    img_gaussian_filter = cv2.GaussianBlur(img,(0,0), sigma)

    ###### non-maximal supression
    # Convolve the image with the Sobel operator (filter) to compute the numerical derivatives in the x and y directions
    sobelx = cv2.Sobel(img_gaussian_filter,cv2.CV_64F,1,0)
    sobely = cv2.Sobel(img_gaussian_filter,cv2.CV_64F,0,1)

    # Compute the magnitude of the gradients
    Gmag = np.sqrt(sobelx*sobelx + sobely*sobely)

    # Compute the orientation of the gradients
    Gdir = cv2.phase(np.array(sobelx, np.float32), np.array(sobely, dtype=np.float32), angleInDegrees=True)
    mask_mag_left = (Gmag > threshold_left)
    mask_mag_right = (Gmag > threshold_right)

    ###### color masking
    mask_white = cv2.inRange(imghsv, white_lower_hsv, white_upper_hsv)
    mask_yellow = cv2.inRange(imghsv, yellow_lower_hsv, yellow_upper_hsv)

    #### edge-based masking
    width = img.shape[1]
    mask_left = np.ones(sobelx.shape)
    mask_left[:,int(np.floor(width/2)):width + 1] = 0
    mask_right = np.ones(sobelx.shape)
    mask_right[:,0:int(np.floor(width/2))] = 0

    mask_sobelx_pos = (sobelx > 0)
    mask_sobelx_neg = (sobelx < 0)
    mask_sobely_pos = (sobely > 0)
    mask_sobely_neg = (sobely < 0)

    mask_left_edge = mask_left * mask_mag_left * mask_sobelx_neg * mask_sobely_neg * mask_yellow
    mask_left_edge[:250] = 0
    mask_right_edge = mask_right * mask_mag_right * mask_sobelx_pos * mask_sobely_neg * mask_white
    mask_right_edge[:250] = 0

    return mask_left_edge, mask_right_edge
