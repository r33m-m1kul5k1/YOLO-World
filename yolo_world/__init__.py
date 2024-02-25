import os.path as osp
from typing import List

import matplotlib.pyplot as plt
import cv2
import numpy as np
import supervision as sv
import torch
from mmengine.config import Config
from mmengine.dataset import Compose
from mmengine.runner import Runner
from mmengine.runner.amp import autocast
from mmyolo.registry import RUNNERS


def load_runner(checkpoint_file: str) -> Runner:
    """Loads a mmengine runner object using the given checkpoint file"""
    config_file = 'yolo_world_x_dual_vlpan_l2norm_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py'
    config_object = Config.fromfile(config_file)
    config_object.work_dir = osp.join('../work_dirs', osp.splitext(osp.basename(config_file))[0])

    config_object.load_from = checkpoint_file
    runner = Runner.from_cfg(config_object) if 'runner_type' not in config_object else RUNNERS.build(config_object)

    runner.call_hook('before_run')
    runner.load_or_resume()
    pipeline = config_object.test_dataloader.dataset.pipeline[1:]  # [1:] removes load image from file
    runner.pipeline = Compose(pipeline)
    runner.model.eval()
    return runner


def inference_detector(runner: Runner,
                       image: np.ndarray,
                       texts: List[List[str]],
                       topk,
                       score_thr,
                       use_amp=False,
                       show_results=False):
    data_info = dict(img_id=0, img=image, ori_shape=image.shape, texts=texts)
    data_info = runner.pipeline(data_info)
    data_batch = dict(inputs=data_info['inputs'].unsqueeze(0),
                      data_samples=[data_info['data_samples']])

    with autocast(enabled=use_amp), torch.no_grad():
        output = runner.model.test_step(data_batch)[0]
        pred_instances = output.pred_instances
        pred_instances = pred_instances[pred_instances.scores.float() > score_thr]
    if len(pred_instances.scores) > topk:
        indices = pred_instances.scores.float().topk(topk)[1]
        pred_instances = pred_instances[indices]

    pred_instances = pred_instances.cpu().numpy()
    detections = sv.Detections(xyxy=pred_instances['bboxes'],
                               class_id=pred_instances['labels'],
                               confidence=pred_instances['scores'])

    labels = [
        f"{texts[class_id][0]} {confidence:0.2f}" for class_id, confidence in
        zip(detections.class_id, detections.confidence)
    ]

    if show_results:
        # label images
        overlayed_image = sv.BoundingBoxAnnotator().annotate(image, detections)
        overlayed_image = sv.LabelAnnotator().annotate(overlayed_image, detections, labels=labels)
        plt.imshow(overlayed_image[..., ::-1])
        plt.show()

    number_of_predictions = len(detections.class_id)
    predictions = np.empty((number_of_predictions, 6))
    for i, bbox, class_id, confidence in zip(
            range(number_of_predictions), detections.xyxy, detections.class_id, detections.confidence):
        x0, y0, x1, y1 = bbox
        predictions[i] = np.array([x0, y0, x1, y1, class_id, confidence])
    return predictions


if __name__ == '__main__':
    runner = load_runner(
        '../../../checkpoints/yolo_world_x_clip_base_dual_vlpan_2e-3adamw_32xb16_100e_o365_goldg_cc3mlite_train_pretrained-8cf6b025.pth')

    inference_detector(runner=runner, image=cv2.imread('../input/(-200, -120).png'),
                       texts=[['car'], ['building'], ['tree'], [' ']],
                       topk=100, score_thr=0.1)
