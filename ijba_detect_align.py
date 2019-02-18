# coding: utf-8
import mxnet as mx
from mtcnn_detector import MtcnnDetector
import cv2
import csv
import os
import numpy as np
from tqdm import tqdm
import logging
from skimage import transform
import matplotlib.pyplot as plt

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('dir', help='ijba dataset that contains the folder images and protocol')
parser.add_argument('out_dir', help='output dataset')
parser.add_argument('--sub_dir', default='mtcnn', type=str, help='sub directory')
parser.add_argument('--enlarge-factor', type=float, default=0.3, help='enlarge factor')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--size', type=int, default=224, help='gpu')
parser.add_argument('--align', action='store_true', help='store_true')
parser.add_argument('--show', action='store_true', help='store_true')
parser.add_argument('--no-crop', action='store_true', help='no crop, just save bounding box and landmarks')
args = parser.parse_args()

ctx = mx.gpu(args.gpu) if args.gpu >= 0 else mx.cpu()

detector = MtcnnDetector(model_folder='model', ctx=ctx, num_worker=4, accurate_landmark=False)

dst = np.array([
      [38.2946, 51.6963],
      [73.5318, 51.5014],
      [56.0252, 71.7366],
      [41.5493, 92.3655],
      [70.7299, 92.2041]], dtype=np.float32) / 112. * args.size

trans = transform.SimilarityTransform()


def read_csv(csv_path):
    items = []
    with open(csv_path) as f:
        csv_reader = csv.reader(f)
        for i, item in enumerate(csv_reader):
            if i == 0:
                label = item
            else:
                t = {}
                for i, l in enumerate(label):
                    if l in ['TEMPLATE_ID', 'SUBJECT_ID']:
                        t[l] = int(item[i])
                    elif l in ['FACE_X', 'FACE_Y', 'FACE_WIDTH', 'FACE_HEIGHT']:
                        t[l] = int(float(item[i]))
                    else:
                        t[l] = item[i]

                items.append(t)
    return items


def crop_align_images(items):
    p_bar = tqdm(total=len(items))

    for item in items:
        template_id, subject_id = item['TEMPLATE_ID'], item['SUBJECT_ID']
        base_d, base_f = os.path.dirname(item['FILE']), os.path.basename(item['FILE'])
        ext = os.path.splitext(base_f)[1]
        out_im = os.path.join(args.out_dir, args.sub_dir, base_d, '%d_%s' % (subject_id, base_f))
        out_box = os.path.join(args.out_dir, 'mtcnn', base_d, '%d_%s' % (subject_id, base_f)).replace(ext, '.box')
        out_lm = os.path.join(args.out_dir, 'mtcnn', base_d, '%d_%s' % (subject_id, base_f)).replace(ext, '.lm')

        if args.no_crop:
            if not os.path.exists(out_box) and not os.path.exists(out_lm):
                try:
                    im = cv2.imread(os.path.join(args.dir, 'images', item['FILE']))

                    if im is None:
                        continue

                    x, y, w, h = item['FACE_X'], item['FACE_Y'], item['FACE_WIDTH'], item['FACE_HEIGHT']

                    if item['RIGHT_EYE_X'] and item['RIGHT_EYE_Y'] and item['LEFT_EYE_X'] and item['LEFT_EYE_Y'] \
                            and item['NOSE_BASE_X'] and item['NOSE_BASE_Y']:
                        lms = np.array([[float(item['RIGHT_EYE_X']), float(item['RIGHT_EYE_Y'])],
                                        [float(item['LEFT_EYE_X']), float(item['LEFT_EYE_Y'])],
                                        [float(item['NOSE_BASE_X']), float(item['NOSE_BASE_Y'])]])
                        lm_max = np.max(lms, axis=0)
                        lm_min = np.min(lms, axis=0)
                        delta = lm_max - lm_min
                        # delta[1] = delta[1] * 2
                        edge = max(delta) * 2
                        cx, cy = np.mean(lms[:, 0]), np.mean(lms[:, 1])

                    else:
                        cx, cy = int(x + 0.5 * w), int(y + 0.6 * h)
                        edge = int((w + h) // 4)
                    # cx, cy = int(x + 0.5 * w), int(y + 0.6 * h)
                    # edge = int((w + h) // 4)

                    x1, y1, x2, y2 = int(cx - edge), int(cy - edge), int(cx + edge), int(cy + edge)
                    x1 = max(0, min(x1, im.shape[1]))
                    x2 = max(0, min(x2, im.shape[1]))
                    y1 = min(im.shape[0], max(y1, 0))
                    y2 = min(im.shape[0], max(y2, 0))

                    crop_im = im[y1: y2, x1: x2]

                    if args.show:
                        plt.imshow(cv2.cvtColor(crop_im, cv2.COLOR_BGR2RGB))
                        plt.show()

                    results = detector.detect_face(crop_im)

                    if results is not None:
                        box = results[0][0] + np.array([x1, y1, x1, y1, 0])
                        x, y = results[1][0][:5].reshape((5, 1)), results[1][0][5:].reshape((5, 1))
                        lm = np.concatenate((x, y), axis=1) + np.array([x1, y1])

                        if not os.path.exists(os.path.dirname(out_box)):
                            os.makedirs(os.path.dirname(out_box))

                        if not os.path.exists(os.path.dirname(out_lm)):
                            os.makedirs(os.path.dirname(out_lm))

                        np.savetxt(out_box, box)
                        np.savetxt(out_lm, lm)
                except Exception as e:
                    # logging.warning(e)
                    continue
        else:
            if not os.path.exists(out_im):
                try:
                    im = cv2.imread(os.path.join(args.dir, 'images', item['FILE']))

                    if im is None:
                        continue

                    x, y, w, h = item['FACE_X'], item['FACE_Y'], item['FACE_WIDTH'], item['FACE_HEIGHT']

                    if item['RIGHT_EYE_X'] and item['RIGHT_EYE_Y'] and item['LEFT_EYE_X'] and item['LEFT_EYE_Y'] \
                            and item['NOSE_BASE_X'] and item['NOSE_BASE_Y']:
                        lms = np.array([[float(item['RIGHT_EYE_X']), float(item['RIGHT_EYE_Y'])],
                                        [float(item['LEFT_EYE_X']), float(item['LEFT_EYE_Y'])],
                                        [float(item['NOSE_BASE_X']), float(item['NOSE_BASE_Y'])]])
                        lm_max = np.max(lms, axis=0)
                        lm_min = np.min(lms, axis=0)
                        delta = lm_max - lm_min
                        # delta[1] = delta[1] * 2
                        edge = max(delta) * 2
                        cx, cy = np.mean(lms[:, 0]), np.mean(lms[:, 1])

                    else:
                        cx, cy = int(x + 0.5 * w), int(y + 0.6 * h)
                        edge = int((w + h) // 4)
                    # cx, cy = int(x + 0.5 * w), int(y + 0.6 * h)
                    # edge = int((w + h) // 4)

                    x1, y1, x2, y2 = int(cx - edge), int(cy - edge), int(cx + edge), int(cy + edge)
                    x1 = max(0, min(x1, im.shape[1]))
                    x2 = max(0, min(x2, im.shape[1]))
                    y1 = min(im.shape[0], max(y1, 0))
                    y2 = min(im.shape[0], max(y2, 0))

                    crop_im = im[y1: y2, x1: x2]

                    if args.show:
                        plt.imshow(cv2.cvtColor(crop_im, cv2.COLOR_BGR2RGB))
                        plt.show()

                    results = detector.detect_face(crop_im)

                    if results is not None:
                        x, y = results[1][0][:5].reshape((5, 1)), results[1][0][5:].reshape((5, 1))
                        lm = np.concatenate((x, y), axis=1) + np.array([x1, y1])

                        if args.align:

                            if args.show:
                                plt.imshow(cv2.cvtColor(crop_im, cv2.COLOR_BGR2RGB))
                                plt.scatter(lm[:, 0] - x1, lm[:, 1] - y1)
                                plt.show()

                            trans.estimate(lm, dst)
                            M = trans.params[0:2, :]

                            crop_im = cv2.warpAffine(im, M, (args.size, args.size), borderValue=0)

                        else:
                            lm_mean = np.mean(lm, axis=0)
                            lm_min = np.min(lm, axis=0)
                            lm_max = np.max(lm, axis=0)
                            edge = np.mean(lm_max - lm_min) * (1 + args.enlarge_factor)
                            x1, y1, x2, y2 = int(lm_mean[0] - edge), int(lm_mean[1] - edge), \
                                             int(lm_mean[0] + edge), int(lm_mean[1] + edge)

                            x1 = max(0, min(x1, im.shape[1]))
                            x2 = max(0, min(x2, im.shape[1]))
                            y1 = min(im.shape[0], max(y1, 0))
                            y2 = min(im.shape[0], max(y2, 0))
                            crop_im = im[y1:y2, x1:x2]

                        if not os.path.exists(os.path.dirname(out_im)):
                            os.makedirs(os.path.dirname(out_im))

                        if not os.path.exists(os.path.dirname(out_lm)):
                            os.makedirs(os.path.dirname(out_lm))

                        if args.show:
                            plt.imshow(cv2.cvtColor(crop_im, cv2.COLOR_BGR2RGB))
                            plt.show()
                            plt.pause(5)

                        cv2.imwrite(out_im, crop_im)
                        np.savetxt(out_lm, lm)
                except Exception as e:
                    # logging.warning(e)
                    continue

        p_bar.update()

    p_bar.close()


proto_dir = os.path.join(args.dir, 'IJB-A_1N_sets')

for split in range(1, 11):
    gallery_items = read_csv(os.path.join(proto_dir, 'split%d' % split, 'search_gallery_%d.csv' % split))
    probe_items = read_csv(os.path.join(proto_dir, 'split%d' % split, 'search_probe_%d.csv' % split))
    train_items = read_csv(os.path.join(proto_dir, 'split%d' % split, 'train_%d.csv' % split))

    crop_align_images(gallery_items)
    crop_align_images(probe_items)
    crop_align_images(train_items)

proto_dir = os.path.join(args.dir, 'IJB-A_11_sets')

for split in range(1, 11):
    verify_items = read_csv(os.path.join(proto_dir, 'split%d' % split, 'verify_metadata_%d.csv' % split))
    train_items = read_csv(os.path.join(proto_dir, 'split%d' % split, 'train_%d.csv' % split))

    crop_align_images(verify_items)
    crop_align_images(train_items)
