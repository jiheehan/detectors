import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets.seg_set import SegSet
from models.tiny_model import TinyModel

seg_class = [
    ['Bkgnd', 0],
    ['Rectangle', '1'],
    ['Circle', '2']
]

if '__main__' == __name__:
    print('Test Seg')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device))

    validation_data = SegSet('test_samples')
    valid_dataloader = DataLoader(validation_data, batch_size = 1)

    model = TinyModel().to(device)
    model.load_state_dict(torch.load('weights/weights_min_test_loss.pth'))
    model.eval()

    def test_loop(dataloader, model):
        with torch.no_grad():
            for batch, (x, y_seg) in enumerate(dataloader):
                x = x.to(device)

                pred_seg = model(x)

                image = x.cpu().numpy()
                image *= 255
                image = np.transpose(image, (0, 2, 3, 1))
                image = np.ascontiguousarray(image, dtype=np.uint8)

                seg_results = torch.argmax(pred_seg, dim=1)
                vis_image = seg_results.cpu().numpy()
                vis_image *= 100
                vis_image = np.transpose(vis_image, (1, 2, 0))
                vis_image = np.ascontiguousarray(vis_image, dtype=np.uint8)

                gt_seg = y_seg * 100
                gt_seg = gt_seg.cpu().numpy()
                gt_seg = np.transpose(gt_seg, (1, 2, 0))
                gt_seg = np.ascontiguousarray(gt_seg, dtype=np.uint8)

                cv2.namedWindow('results')
                cv2.imshow('results', vis_image)
                cv2.namedWindow('GT')
                cv2.imshow('GT', gt_seg)
                cv2.waitKey(0)

    test_loop(valid_dataloader, model)

