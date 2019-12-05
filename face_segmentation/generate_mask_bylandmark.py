import os.path as osp
import os
import torch
import PIL.Image as Image
from torchvision.transforms import transforms
import numpy as np
import cv2
from face_alignment import FaceAlignment, LandmarksType


def get_image_hull_mask(image_shape, image_landmarks, ie_polys=None):
    # get the mask of the image
    if image_landmarks.shape[0] != 68:
        raise Exception(
            'get_image_hull_mask works only with 68 landmarks')
    int_lmrks = np.array(image_landmarks, dtype=np.int)

    hull_mask = np.zeros(image_shape[0:2]+(1,), dtype=np.float32)

    cv2.fillConvexPoly(hull_mask, cv2.convexHull(
        np.concatenate((int_lmrks[0:9],
                        int_lmrks[17:18]))), (1,))

    cv2.fillConvexPoly(hull_mask, cv2.convexHull(
        np.concatenate((int_lmrks[8:17],
                        int_lmrks[26:27]))), (1,))

    cv2.fillConvexPoly(hull_mask, cv2.convexHull(
        np.concatenate((int_lmrks[17:20],
                        int_lmrks[8:9]))), (1,))

    cv2.fillConvexPoly(hull_mask, cv2.convexHull(
        np.concatenate((int_lmrks[24:27],
                        int_lmrks[8:9]))), (1,))

    cv2.fillConvexPoly(hull_mask, cv2.convexHull(
        np.concatenate((int_lmrks[19:25],
                        int_lmrks[8:9],
                        ))), (1,))

    cv2.fillConvexPoly(hull_mask, cv2.convexHull(
        np.concatenate((int_lmrks[17:22],
                        int_lmrks[27:28],
                        int_lmrks[31:36],
                        int_lmrks[8:9]
                        ))), (1,))

    cv2.fillConvexPoly(hull_mask, cv2.convexHull(
        np.concatenate((int_lmrks[22:27],
                        int_lmrks[27:28],
                        int_lmrks[31:36],
                        int_lmrks[8:9]
                        ))), (1,))

    # nose
    cv2.fillConvexPoly(
        hull_mask, cv2.convexHull(int_lmrks[27:36]), (1,))

    if ie_polys is not None:
        ie_polys.overlay_mask(hull_mask)

    return hull_mask


def merge(img_1, mask):
    # merge rgb and mask into a rgba image
    r_channel, g_channel, b_channel = cv2.split(img_1)
    if mask is not None:
        alpha_channel = np.ones(mask.shape, dtype=img_1.dtype)
        alpha_channel *= mask*255
    else:
        alpha_channel = np.zeros(img_1.shape[:2], dtype=img_1.dtype)
    img_BGRA = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))
    return img_BGRA


def evaluate(respth='./results/data_src', dspth='../data'):
    respth = osp.join(os.path.abspath(os.path.dirname(__file__)), respth)
    if not os.path.exists(respth):
        os.makedirs(respth)

    face_model = FaceAlignment(LandmarksType._2D, device="cuda")
    data_path = osp.join(os.path.abspath(os.path.dirname(__file__)), dspth)
    for image_path in os.listdir(data_path):
        image = cv2.imread(osp.join(data_path, image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        landmark = face_model.get_landmarks(image)[-1]
        # print(landmark)
        mask = get_image_hull_mask(np.shape(image), landmark).astype(np.uint8)
        # cv2.imshow("mask", (mask*255).astype(np.uint8))

        image_bgra = merge(image, mask)
        # cv2.imshow("image_bgra", image_bgra)
        # cv2.waitKey(1)
        save_path = osp.join(respth, image_path)
        cv2.imwrite(save_path[:-4] + '.png', image_bgra)


if __name__ == "__main__":
    evaluate(dspth='../data/data_obj', respth='./results/data_obj')
