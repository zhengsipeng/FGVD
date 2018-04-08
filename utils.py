'''
Created on Sun Apr 8 2018

@ zhengsipeng
'''
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

def orb_matches(frame1, frame2):
    #img1 = cv2.imread(img1_path, 0)
    orb = cv2.ORB_create(1000)

    #find the keypoints and descriptors
    kp1, des1 = orb.detectAndCompute(frame1, None)
    kp2, des2 = orb.detectAndCompute(frame2, None)

    #abstract matching keypoints
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    #img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=2)
    #plt.imshow(img3)
    #plt.show()
    return len(matches)/float(len(kp1))

def keyframe_or_not(curr_frame, key_frame):
    sim = orb_matches(curr_frame, key_frame)
    if sim > 0.6:
        return True
    else:
        return False

if __name__ == '__main__':
    dirpath = os.path.join(os.getcwd(), 'test_img')
    img1_path = os.path.join(dirpath, 'orb_1.png')
    img2_path = os.path.join(dirpath, 'orb_2.png')

    img1 = cv2.imread(img1_path, 0)
    img2 = cv2.imread(img2_path, 0)
    orb_matches(img1, img2)