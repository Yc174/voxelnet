import os
from lib.dataset.kitti_object import kitti_object

total_labels = {'Car': 1, 'Pedestrian': 2, 'Cyclist': 3}

def generate_idx_file(root_dir, labels=[], split='train'):
    if not isinstance(labels, list):
        raise AssertionError('labels should be list!')
    assert len(labels) < 3 and len(labels) > 0
    for l in labels:
        if l not in total_labels:
            raise AssertionError('labels not in total_labels!!')
    idx_filename = ''
    if split == 'train':
        idx_filename = 'train_backup.txt'
        split = 'training'
    elif split == 'val':
        idx_filename = 'val_backup.txt'
        split = 'training'  # rename
    elif split == 'test':
        idx_filename = 'test.txt'
        split = 'testing'
    idx_filename = os.path.join(os.path.dirname(__file__), '..', 'lib/dataset/idx_files', idx_filename)
    generated_file = '_'.join(labels)
    generated_file = os.path.join(os.path.dirname(idx_filename), 'generated_%s_%s.txt'%(generated_file, split))
    img_ids = [int(line.rstrip()) for line in open(idx_filename)]

    kitti = kitti_object(root_dir, split)
    new_ids=[]
    f = open(generated_file, 'w')
    for idx in img_ids:
        gts = kitti.get_label_objects(idx)
        tmp = 0
        for gt in gts:
            if gt.type in labels:
                tmp += 1
        if tmp != 0:
            idx = '%06d'%idx
            new_ids.append(idx)
            f.writelines(str(idx)+'\n')
    f.close()
    print('labels: {}, split: {}, idx: {}'.format(labels, split, new_ids))
    print('len of files: %d'%(len(new_ids)))

if __name__ == '__main__':
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'datasets/KITTI/object')
    # {'Car': 1, 'Pedestrian': 2, 'Cyclist': 3}
    generate_idx_file(data_dir, labels=['Cyclist'], split='train')