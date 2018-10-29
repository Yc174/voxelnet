import sys
import os
import numpy as np
import datetime
import subprocess
from distutils import dir_util

from lib.functions.anchor_projector import project_to_image_space

def save_predictions_in_kitti_format(dataset, score_threshold=0.1, global_step=10000):
    """ Converts a set of network predictions into text files required for
    KITTI evaluation.
    """
    # Get available prediction folders
    root_dir = os.path.join(os.path.dirname(__file__), '..')
    predictions_root_dir = os.path.join(root_dir, 'experiments', 'predictions')
    final_predictions_root_dir = predictions_root_dir + \
        '/final_predictions_and_scores/' + dataset.split
    final_predictions_dir = final_predictions_root_dir + \
        '/' + str(global_step)

    # 3D prediction directories
    kitti_predictions_3d_dir = predictions_root_dir + \
        '/kitti_native_eval/' + \
        str(score_threshold) + '/' + \
        str(global_step) + '/data'

    if not os.path.exists(kitti_predictions_3d_dir):
        os.makedirs(kitti_predictions_3d_dir)

    # Do conversion
    num_samples = dataset.num
    num_valid_samples = 0

    print('\nGlobal step:', global_step)
    print('Converting detections from:', final_predictions_dir)

    print('3D Detections being saved to:', kitti_predictions_3d_dir)

    for sample_idx in range(num_samples):

        # Print progress
        sys.stdout.write('\rConverting {} / {}'.format(
            sample_idx + 1, num_samples))
        sys.stdout.flush()

        sample_name = dataset.img_ids[sample_idx]

        prediction_file = sample_name + '.txt'

        kitti_predictions_3d_file_path = kitti_predictions_3d_dir + \
            '/' + prediction_file

        predictions_file_path = final_predictions_dir + \
            '/' + prediction_file

        # If no predictions, skip to next file
        if not os.path.exists(predictions_file_path):
            np.savetxt(kitti_predictions_3d_file_path, [])
            continue

        # n*(x,y,z,l,w,h,ry,scores)
        all_predictions = np.loadtxt(predictions_file_path)

        # # Swap l, w for predictions where w > l
        # swapped_indices = all_predictions[:, 4] > all_predictions[:, 3]
        # fixed_predictions = np.copy(all_predictions)
        # fixed_predictions[swapped_indices, 3] = all_predictions[
        #     swapped_indices, 4]
        # fixed_predictions[swapped_indices, 4] = all_predictions[
        #     swapped_indices, 3]

        score_filter = all_predictions[:, 8] >= score_threshold
        all_predictions = all_predictions[score_filter]

        # If no predictions, skip to next file
        if len(all_predictions) == 0:
            np.savetxt(kitti_predictions_3d_file_path, [])
            continue

        # Project to image space
        img_id = int(sample_name)

        # Load image for truncation
        image = dataset.kitti.get_image(img_id)

        calib = dataset.kitti.get_calibration(img_id)

        boxes = []
        image_filter = []
        for i in range(len(all_predictions)):
            box_3d = all_predictions[i, 0:7]
            img_box, _ = project_to_image_space(box_3d, calib.P, image.size)

            # Skip invalid boxes (outside image space)
            if img_box is None:
                image_filter.append(False)
                continue

            image_filter.append(True)
            boxes.append(img_box)

        boxes = np.asarray(boxes)
        all_predictions = all_predictions[image_filter]

        # If no predictions, skip to next file
        if len(boxes) == 0:
            np.savetxt(kitti_predictions_3d_file_path, [])
            continue

        num_valid_samples += 1

        # To keep each value in its appropriate position, an array of zeros
        # (N, 16) is allocated but only values [4:16] are used
        kitti_predictions = np.zeros([len(boxes), 16])

        # Get object types
        all_pred_classes = all_predictions[:, 8].astype(np.int32)
        obj_types = [dataset.classes[class_idx]
                     for class_idx in all_pred_classes]

        # Truncation and Occlusion are always empty (see below)

        # Alpha (Not computed)
        kitti_predictions[:, 3] = -10 * np.ones((len(kitti_predictions)),
                                                dtype=np.int32)

        # 2D predictions
        kitti_predictions[:, 4:8] = boxes[:, 0:4]

        # 3D predictions
        # (l, w, h)
        kitti_predictions[:, 8] = all_predictions[:, 5]
        kitti_predictions[:, 9] = all_predictions[:, 4]
        kitti_predictions[:, 10] = all_predictions[:, 3]
        # (x, y, z)
        kitti_predictions[:, 11:14] = all_predictions[:, 0:3]
        # (ry, score)
        kitti_predictions[:, 14:16] = all_predictions[:, 6:8]

        # Round detections to 3 decimal places
        kitti_predictions = np.round(kitti_predictions, 3)

        # Empty Truncation, Occlusion
        kitti_empty_1 = -1 * np.ones((len(kitti_predictions), 2),
                                     dtype=np.int32)

        # Stack 3D predictions text
        kitti_text_3d = np.column_stack([obj_types,
                                         kitti_empty_1,
                                         kitti_predictions[:, 3:16]])

        # Save to text files
        np.savetxt(kitti_predictions_3d_file_path, kitti_text_3d,
                   newline='\r\n', fmt='%s')

    print('\nNum valid:', num_valid_samples)
    print('Num samples:', num_samples)


def copy_kitti_native_code(checkpoint_name):
    pass

def run_kitti_native_script(checkpoint_name, score_threshold, global_step):
    pass