import cv2
import numpy as np
import random

rect_size = 50
radius = 50

def draw_rectangle(image, center, color):
    pt1 = (center[0] - int(rect_size/2), center[1] - int(rect_size/2))
    pt2 = (center[0] + int(rect_size/2), center[1] + int(rect_size/2))

    cv2.rectangle(image, pt1=pt1, pt2=pt2, color=color, thickness=-1)

def draw_circle(image, center, color):
    cv2.circle(image, center=center, radius=radius, color=color, thickness=-1)

colors  = [
    (0, 0, 255),
    (0, 255, 0),
    (255, 0, 0)
]
if __name__ == '__main__':
    print('data generator')

    for idx in range(10):
        print('idx:', idx)
        image = np.zeros((256, 256, 3), dtype=np.uint8)
        gt_image = np.zeros((256, 256, 1), dtype=np.uint8)
        
        x = random.randint(0, 255)
        y = random.randint(0, 255)
        c_idx = random.randint(0, 2)
        draw_circle(image, (x, y), color=colors[c_idx])
        draw_circle(gt_image, (x, y), color=200)

        x = random.randint(0, 255)
        y = random.randint(0, 255)
        c_idx = random.randint(0, 2)
        draw_rectangle(image, (x, y), color=colors[c_idx])
        draw_rectangle(gt_image, (x, y), color=100)

        #cv2.imshow('test', image)
        #cv2.imshow('gt_test', gt_image)
        cv2.imwrite('./test_samples/img{:08}.png'.format(idx), image)
        cv2.imwrite('./test_samples/seg_gt/gt{:08}.png'.format(idx), gt_image)
