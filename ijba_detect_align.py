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
from matplotlib.patches import Rectangle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('dir', help='ijba dataset that contains the folder images and protocol')
parser.add_argument('out_dir', help='output dataset')
parser.add_argument('--crop_dir', default='loose_crop', type=str, help='sub directory')
parser.add_argument('--align_dir', default='align', type=str, help='sub directory')
parser.add_argument('--enlarge-factor', type=float, default=0.3, help='enlarge factor')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--size', type=int, default=224, help='gpu')
parser.add_argument('--avoid-miss-detect', action='store_true', help='store_true')
parser.add_argument('--show', action='store_true', help='store_true')
args = parser.parse_args()

ctx = mx.gpu(args.gpu) if args.gpu >= 0 else mx.cpu()

global detector

dst = np.array([
      [38.2946, 51.6963],
      [73.5318, 51.5014],
      [56.0252, 71.7366],
      [41.5493, 92.3655],
      [70.7299, 92.2041]], dtype=np.float32) / 112. * args.size

trans = transform.SimilarityTransform()


def read_csv(csv_path):
    items = []
    with open(csv_path) as fin:
        csv_reader = csv.DictReader(fin)
        for item in csv_reader:
            items.append(item)

    return items


def face_detection(item):
    im = cv2.imread(os.path.join(args.dir, 'images', item['FILE']))

    if im is None:
        logging.warning('Cannot read image: ' + os.path.join(args.dir, 'images', item['FILE']))
        return None

    x, y, w, h = float(item['FACE_X']), float(item['FACE_Y']), float(item['FACE_WIDTH']), float(item['FACE_HEIGHT'])

    cx, cy = int(x + 0.5 * w), int(y + 0.6 * h)
    edge = int((w + h) // 4) * (1 + args.enlarge_factor)

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

        return {'box': box, 'lm': lm}
    else:
        logging.warning('Cannot detect face from {}'.format(item['FILE']))
        return {'box': np.array([x1, y1, x2, y2, 0])}


def crop_images(protocol_file):
    items = read_csv(protocol_file)

    for item in tqdm(items, desc='processing {}'.format(protocol_file)):
        template_id, subject_id = item['TEMPLATE_ID'], item['SUBJECT_ID']
        base_d, base_f = os.path.dirname(item['FILE']), os.path.basename(item['FILE'])
        ext = os.path.splitext(base_f)[1]

        postfix = os.path.join(base_d, '{}_{}'.format(subject_id, base_f))

        out_im = os.path.join(args.out_dir, args.crop_dir, postfix) \
            if args.crop_dir else os.path.join(args.out_dir, postfix)

        out_box = os.path.join(args.out_dir, 'mtcnn', postfix).replace(ext, '.box')
        out_lm = os.path.join(args.out_dir, 'mtcnn', postfix).replace(ext, '.lm')

        if os.path.exists(out_box) and os.path.exists(out_lm):
            # res = face_detection(item)
            box = np.loadtxt(out_box, dtype=float)
            lm = np.loadtxt(out_lm, dtype=float)
            res = {'box': box, 'lm': lm}
        else:
            res = face_detection(item)

        if res is None:
            missed_images[postfix] = item
        else:
            if 'lm' not in res:
                box, lm = res['box'], None
                missed_images[postfix] = item
            else:
                box, lm = res['box'], res['lm']

            if not os.path.exists(out_im):
                im = cv2.imread(os.path.join(args.dir, 'images', item['FILE']))
                if args.show:
                    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
                    plt.gca().add_patch(Rectangle(box[:2], box[2] - box[0], box[3] - box[1], fill=False))
                    plt.scatter(lm[:, 0], lm[:, 1])
                    plt.show()

                if lm is not None:
                    lm_mean = np.mean(lm, axis=0)
                    lm_min = np.min(lm, axis=0)
                    lm_max = np.max(lm, axis=0)
                    edge = np.mean(lm_max - lm_min) * (1 + args.enlarge_factor * 2)

                    x1, y1, x2, y2 = int(lm_mean[0] - edge), int(lm_mean[1] - edge), \
                                     int(lm_mean[0] + edge), int(lm_mean[1] + edge)

                    x1 = max(0, min(x1, im.shape[1]))
                    x2 = max(0, min(x2, im.shape[1]))
                    y1 = min(im.shape[0], max(y1, 0))
                    y2 = min(im.shape[0], max(y2, 0))
                    crop_im = im[y1:y2, x1:x2]

                    # transform box and lms
                    box -= np.array([x1, y1, x1, y1, 0])
                    lm -= np.array([x1, y1])
                else:
                    x1, y1, x2, y2 = box[:-1]
                    crop_im = im[y1:y2, x1:x2]

                if args.show:
                    plt.imshow(cv2.cvtColor(crop_im, cv2.COLOR_BGR2RGB))
                    plt.gca().add_patch(Rectangle(box[:2], box[2] - box[0], box[3] - box[1], fill=False))
                    plt.scatter(lm[:, 0], lm[:, 1])
                    plt.show()

                if not os.path.exists(os.path.dirname(out_lm)):
                    os.makedirs(os.path.dirname(out_lm))

                if not os.path.exists(os.path.dirname(out_box)):
                    os.makedirs(os.path.dirname(out_box))

                if not os.path.exists(os.path.dirname(out_im)):
                    os.makedirs(os.path.dirname(out_im))

                if lm is not None:
                    np.savetxt(out_box, box)
                    np.savetxt(out_lm, lm)

                cv2.imwrite(out_im, crop_im)

            item['detection_score'] = box[-1]
            if lm is not None:
                item['lms'] = lm.flatten().tolist()

            if postfix not in enrolled_images:
                # assert enrolled_images[postfix] == item, '{} {}'.format(enrolled_images[postfix], item)
                enrolled_images[postfix] = item


def align_image(line):
    i, tp_id, sb_id, md_id = line[:4]
    lms = line[4: 14]
    fs = float(line[14])
    file = line[15]

    base_d, base_f = os.path.dirname(file), os.path.basename(file)
    postfix = os.path.join(base_d, '{}_{}'.format(sb_id, base_f))

    out_im = os.path.join(args.out_dir, args.align_dir, postfix)

    if not os.path.exists(out_im):
        im = cv2.imread(os.path.join(args.out_dir, args.crop_dir, postfix))

        if not os.path.exists(os.path.dirname(out_im)):
            os.makedirs(os.path.dirname(out_im))

        if fs == 0:
            # no landmarks
            if args.avoid_miss_detect:
                return None
            else:
                h, w = im.shape[:2]
                cx, cy = w // 2, h // 2
                r = int((h + w) / 4 / (1 + args.enlarge_factor))
                crop_im = im[cy - r: cy + r, cx - r: cx + r]

                if args.show:
                    plt.imshow(cv2.cvtColor(crop_im, cv2.COLOR_BGR2RGB))
                    plt.show()

                cv2.imwrite(out_im, crop_im)
                return [int(sb_id), int(tp_id), int(md_id), float(fs), postfix]
        else:
            lmk = np.array(list(map(float, lms))).reshape((5, 2))

            trans.estimate(lmk, dst)
            M = trans.params[0:2, :]

            crop_im = cv2.warpAffine(im, M, (args.size, args.size), borderValue=0)

            if args.show:
                plt.imshow(cv2.cvtColor(crop_im, cv2.COLOR_BGR2RGB))
                plt.show()

            cv2.imwrite(out_im, crop_im)

            return [int(sb_id), int(tp_id), int(md_id), float(fs), postfix]


# detect and crop
if not os.path.exists(os.path.join(args.out_dir, args.crop_dir + '.lst')):
    detector = MtcnnDetector(model_folder='model', minsize=5, ctx=ctx, num_worker=1, accurate_landmark=True)
    enrolled_images = {}
    missed_images = {}

    proto_dir = os.path.join(args.dir, 'IJB-A_1N_sets')
    for split in range(1, 11):
        crop_images(os.path.join(proto_dir, 'split%d' % split, 'search_gallery_%d.csv' % split))
        crop_images(os.path.join(proto_dir, 'split%d' % split, 'search_probe_%d.csv' % split))
        crop_images(os.path.join(proto_dir, 'split%d' % split, 'train_%d.csv' % split))

    proto_dir = os.path.join(args.dir, 'IJB-A_11_sets')

    for split in range(1, 11):
        crop_images(os.path.join(proto_dir, 'split%d' % split, 'verify_metadata_%d.csv' % split))
        crop_images(os.path.join(proto_dir, 'split%d' % split, 'train_%d.csv' % split))

    image_list = []
    for i, (key, item) in enumerate(enrolled_images.items()):
        t = [i, item['TEMPLATE_ID'], item['SUBJECT_ID'], item['MEDIA_ID']]
        if 'lms' in item:
            t.extend(item['lms'])
        else:
            t.extend([-1] * 10)
        t.append(item['detection_score'])
        t.append(item['FILE'])
        image_list.append(t)

    with open(os.path.join(args.out_dir, args.crop_dir + '.lst'), 'w') as fout:
        csv_writer = csv.writer(fout, delimiter='\t')
        csv_writer.writerows(image_list)

    with open(os.path.join(args.out_dir, 'ijba_miss_detect.lst'), 'w') as fout:
        for k in missed_images.keys():
            fout.write(k + '\n')

# align
if not os.path.exists(os.path.join(args.out_dir, args.align_dir, 'align.lst')):
    image_list = []
    with open(os.path.join(args.out_dir, args.crop_dir + '.lst')) as fin:
        csv_reader = csv.reader(fin, delimiter='\t')
        count = 0
        for line in tqdm(csv_reader):
            item = align_image(line)
            if item is not None:
                item.insert(0, count)
                image_list.append(item)
                count += 1

    with open(os.path.join(args.out_dir, args.align_dir, 'align.lst'), 'w') as fout:
        csv_writer = csv.writer(fout, delimiter='\t')
        csv_writer.writerows(image_list)
