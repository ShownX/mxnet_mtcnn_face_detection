import os
import cv2
import numpy as np
import mxnet as mx
import argparse
from tqdm import tqdm
from mtcnn_detector import MtcnnDetector

parser = argparse.ArgumentParser()
parser.add_argument('root', help="input directory")
parser.add_argument('out_root', help='output file')
parser.add_argument('--gpu', type=int, default=0, help='gpu')

args = parser.parse_args()

args.root = args.root if args.root.endswith(os.sep) else args.root + os.sep

img_files = [os.path.join(path, name) for path, subdirs, files in os.walk(args.root) for name in files
            if os.path.splitext(name)[1].lower() in ['.jpeg', '.jpg', '.png', '.bmp']]

ctx = mx.gpu(args.gpu) if args.gpu >= 0 else mx.cpu()

detector = MtcnnDetector(model_folder='model', ctx=ctx, num_worker=4, accurate_landmark=True)

with tqdm(total=len(img_files)) as pbar:
    for item in img_files:
        ext = os.path.splitext(item)[1]
        out_im = item.replace(args.root, args.out_root)
        out_box = out_im.replace(ext, '.box')
        out_lm = out_im.replace(ext, '.lm')

        try:
            im = cv2.imread(os.path.join(args.root, item))

            if im is None:
                continue

            results = detector.detect_face(im)

            if results is not None:
                box = results[0][0]
                x, y = results[1][0][:5].reshape((5, 1)), results[1][0][5:].reshape((5, 1))
                lm = np.concatenate((x, y), axis=1)

                if not os.path.exists(os.path.dirname(out_box)):
                    os.makedirs(os.path.dirname(out_box))

                if not os.path.exists(os.path.dirname(out_lm)):
                    os.makedirs(os.path.dirname(out_lm))

                np.savetxt(out_box, box)
                np.savetxt(out_lm, lm)

        except Exception as e:
            # logging.warning(e)
            continue

        pbar.update()
