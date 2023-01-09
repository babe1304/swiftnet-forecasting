import contextlib

import numpy as np
import torch
from tqdm import tqdm
from time import perf_counter
import time


__all__ = ['compute_errors', 'get_pred', 'evaluate_semseg']


class ConfusionMatrix():
    def __init__(self, num_classes, ignore_label=255, class_info=None):
        self.num_classes = num_classes
        self.ignore_label = ignore_label
        self.class_info = class_info
        self.conf_matrix = np.zeros((num_classes, num_classes), dtype=np.long)

    def update(self, preds, labels):
        mask = labels != self.ignore_label
        preds = preds[mask]
        labels = labels[mask]

        arr = preds * self.num_classes + labels
        curr_conf_matrix = np.bincount(arr, minlength=self.num_classes**2)
        curr_conf_matrix = curr_conf_matrix.reshape((self.num_classes, self.num_classes)).astype(np.long)
        self.conf_matrix += curr_conf_matrix
        return curr_conf_matrix

    def get_iou(self, class_id):
        TP = self.conf_matrix[class_id, class_id]
        FP = np.sum(self.conf_matrix[class_id]) - TP
        FN = np.sum(self.conf_matrix[:, class_id]) - TP
        iou = TP / (TP + FP + FN)
        return iou
        
    def get_subset_miou(self, class_ids):
        index = 0
        iou_lst = []
        for id in class_ids:
            iou_lst.append(self.get_iou(id))
            index += 1
        return sum(iou_lst) / index

    def get_matrix(self):
        return self.conf_matrix

    def get_metrics(self, verbose=True):
        class_iou = np.zeros(self.num_classes)

        for id in range(self.num_classes):
            TP = self.conf_matrix[id, id]
            FP = np.sum(self.conf_matrix[id]) - TP
            FN = np.sum(self.conf_matrix[:, id]) - TP
            iou = TP / (TP + FP + FN)
            class_iou[id] = iou
            if verbose:
                print('\t{} IoU accuracy = {:.2f} %'.format(self.class_info[id] if self.class_info is not None else id, iou * 100))

        miou = np.mean(class_iou)
        if verbose:
            print('\tmIoU accuracy = {:.2f} %'.format(miou * 100))

        return miou, class_iou

    def reset(self):
        self.conf_matrix = np.zeros_like(self.conf_matrix)


def compute_errors(conf_mat, class_info, verbose=True):
    num_correct = conf_mat.trace()
    num_classes = conf_mat.shape[0]
    total_size = conf_mat.sum()
    avg_pixel_acc = num_correct / total_size * 100.0
    TPFP = conf_mat.sum(1)
    TPFN = conf_mat.sum(0)
    FN = TPFN - conf_mat.diagonal()
    FP = TPFP - conf_mat.diagonal()
    class_iou = np.zeros(num_classes)
    class_recall = np.zeros(num_classes)
    class_precision = np.zeros(num_classes)
    per_class_iou = []
    if verbose:
        print('Errors:')
    for i in range(num_classes):
        TP = conf_mat[i, i]
        class_iou[i] = (TP / (TP + FP[i] + FN[i])) * 100.0
        if TPFN[i] > 0:
            class_recall[i] = (TP / TPFN[i]) * 100.0
        else:
            class_recall[i] = 0
        if TPFP[i] > 0:
            class_precision[i] = (TP / TPFP[i]) * 100.0
        else:
            class_precision[i] = 0

        class_name = class_info[i]
        per_class_iou += [(class_name, class_iou[i])]
        if verbose:
            print('\t%s IoU accuracy = %.2f %%' % (class_name, class_iou[i]))
    avg_class_iou = class_iou.mean()
    avg_class_recall = class_recall.mean()
    avg_class_precision = class_precision.mean()
    if verbose:
        print('IoU mean class accuracy -> TP / (TP+FN+FP) = %.2f %%' % avg_class_iou)
        print('mean class recall -> TP / (TP+FN) = %.2f %%' % avg_class_recall)
        print('mean class precision -> TP / (TP+FP) = %.2f %%' % avg_class_precision)
        print('pixel accuracy = %.2f %%' % avg_pixel_acc)
    return avg_pixel_acc, avg_class_iou, avg_class_recall, avg_class_precision, total_size, per_class_iou


def get_pred(logits, labels, conf_mat):
    _, pred = torch.max(logits.data, dim=1)
    print(labels.shape, logits.shape, pred.shape)
    pred = pred.byte().cpu()
    pred = pred.numpy().astype(np.int32)
    true = labels.numpy().astype(np.int32)
    # cylib.collect_confusion_matrix(pred.reshape(-1), true.reshape(-1), conf_mat)


def mt(sync=False):
    if sync:
        torch.cuda.synchronize()
    return 1000 * perf_counter()


def evaluate_semseg(model, data_loader, class_info, observers=()):
    model.eval()
    managers = [torch.no_grad()] + list(observers)
    conf_matrix = ConfusionMatrix(num_classes=19, ignore_label=255, class_info=class_info)
    with contextlib.ExitStack() as stack:
        for ctx_mgr in managers:
            stack.enter_context(ctx_mgr)
        for step, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
            batch['original_labels'] = batch['original_labels'].numpy().astype(np.uint32)
            logits, additional = model.do_forward(batch, batch['original_labels'].shape[1:3])
            pred = torch.argmax(logits.data, dim=1).byte().cpu().numpy().astype(np.uint32)
            for o in observers:
                o(pred, batch, additional)

            #t1 = time.time()
            conf_matrix.update(pred.flatten(), batch['original_labels'].flatten())
            #t2 = time.time()
            #print('Confusion matrix compute time: {:.1f} ms'.format((t2-t1)*1000))
            #time.sleep(2)
        print('')
        pixel_acc, iou_acc, recall, precision, _, per_class_iou = compute_errors(conf_matrix.get_matrix(), class_info, verbose=True)
        #print('Mine:')
        #conf_matrix.get_metrics()
    model.train()
    return iou_acc, per_class_iou
