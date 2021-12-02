import os
import os.path
import sys
from multiprocessing import Pool
import numpy as np
import cv2
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from codes.utils.progress_bar import ProgressBar # 進度條

def main():
    """A multi-thread tool to crop sub imags."""
    input_folder = '/home/a2902793/DIV2K_train_LR_mild'
    save_folder = '/home/a2902793/DIV2K_train_LR_sub/'
    # input_folder = '/home/e517herb/Desktop/GDSR/image_set/my_photo'
    # save_folder = '/home/e517herb/Desktop/image_sub1'
    n_thread = 20
    crop_sz = 120 #480
    step = 60 #240
    thres_sz = 12 #48
    compression_level = 3  # 3 is the default value in cv2
    # CV_IMWRITE_PNG_COMPRESSION from 0 to 9. A higher value means a smaller size and longer
    # compression time. If read raw images during training, use 0 for faster IO speed.

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        print('mkdir [{:s}] ...'.format(save_folder))
    else:
        print('Folder [{:s}] already exists. Exit...'.format(save_folder))
        sys.exit(1)

    img_list = []
    for root, _, file_list in sorted(os.walk(input_folder)):
        path = [os.path.join(root, x) for x in file_list]  # assume only images in the input_folder
        img_list.extend(path)

    def update(arg):
        pbar.update(arg) # 更新進度條

    pbar = ProgressBar(len(img_list)) # 計算我有幾個工作

    pool = Pool(n_thread) # 執行序(多人一起執行分工)
    for path in img_list:
        pool.apply_async(worker,
                         args=(path, save_folder, crop_sz, step, thres_sz, compression_level),
                         callback=update)
    pool.close()
    pool.join()
    print('All subprocesses done.')


def worker(path, save_folder, crop_sz, step, thres_sz, compression_level):
    img_name = os.path.basename(path)
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    n_channels = len(img.shape)
    if n_channels == 2:
        h, w = img.shape
    elif n_channels == 3:
        h, w, c = img.shape
    else:
        raise ValueError('Wrong image shape - {}'.format(n_channels))

    h_space = np.arange(0, h - crop_sz + 1, step)
    if h - (h_space[-1] + crop_sz) > thres_sz:
        h_space = np.append(h_space, h - crop_sz)
    w_space = np.arange(0, w - crop_sz + 1, step)
    if w - (w_space[-1] + crop_sz) > thres_sz:
        w_space = np.append(w_space, w - crop_sz)

    index = 0
    for x in h_space:
        for y in w_space:
            index += 1
            if n_channels == 2:
                crop_img = img[x:x + crop_sz, y:y + crop_sz]
            else:
                crop_img = img[x:x + crop_sz, y:y + crop_sz, :]
            crop_img = np.ascontiguousarray(crop_img) #ascontiguousarray函數將一個內存不連續存儲的數組
                                                      #轉換為內存連續存儲的數組，使得運行速度更快。
            
            cv2.imwrite(
                os.path.join(save_folder, img_name.replace('.png', '_s{:03d}.png'.format(index))),
                crop_img, [cv2.IMWRITE_PNG_COMPRESSION, compression_level])
    return 'Processing {:s} ...'.format(img_name)


if __name__ == '__main__':
    main()
