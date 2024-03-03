import torch
import cv2
import os
import numpy as np
from torch.utils.data import Dataset

class SegSet(Dataset):
    def __init__(self, data_path):
        super(SegSet, self).__init__()
        files = os.scandir(data_path)
        self.image_list = []

        for f in files:
            filename = f.name
            if f.name.startswith('img') and f.name.endswith('.png'):
                self.image_list.append(f.path)

    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        image = cv2.imread(self.image_list[idx])
        
        image = image.astype(np.float32)
        image /= 255.0
        image =  image.transpose((2, 0, 1))
        image = torch.from_numpy(image)

        file_dir, file_name = os.path.split(self.image_list[idx])
        gt_filepath = os.path.join(file_dir, 'seg_gt', 'gt'+file_name[3:])
        gt_image = cv2.imread(gt_filepath, cv2.IMREAD_GRAYSCALE)

        # make gt mask
        gt_seg = (gt_image / 100).astype(dtype=np.int64)    # 0: bkg, 1: rect, 2: circle
        gt_seg = torch.from_numpy(gt_seg)

        return image, gt_seg

if __name__ == '__main__':
    data_set = SegSet('../train_samples')
    dataloader = torch.utils.data.DataLoader(data_set, batch_size=1)

    print(len(dataloader))
    for batch, (x, y_seg) in enumerate(dataloader):
        gt = y_seg.cpu().numpy()
        gt *= 100
        gt = gt.astype(np.uint8)
        gt = np.transpose(gt, (1, 2, 0))

        print(gt.shape)
        cv2.namedWindow('test')
        cv2.imshow('test', gt)
        cv2.waitKey(0)

    
