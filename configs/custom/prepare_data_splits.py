import os
import json
from pycocotools.coco import COCO

def prepare_splits(
    coco_json_path="/kaggle/input/datasets/nta212/ip102-for-object-detection/train.json",
    output_dir="data/IP102/VOC2007/ImageSets/Main",
    num_exemplars_per_class=20
):
    """
    Chuẩn bị dữ liệu ImageSets cho IP102 dataset (25 lớp, chia làm 4 task):
    - Task 1: 7 lớp (0..6)
    - Task 2: 6 lớp (7..12)
    - Task 3: 6 lớp (13..18)
    - Task 4: 6 lớp (19..24)
    """
    print(f"Đang đọc file annotations: {coco_json_path}")
    coco = COCO(coco_json_path)

    # Chia 25 lớp vào 4 task: Task 1 có 7 lớp, các task sau có 6 lớp
    task_splits = {
        't1': list(range(0, 7)),
        't2': list(range(7, 13)),
        't3': list(range(13, 19)),
        't4': list(range(19, 25))
    }

    # 1. Tạo tập train cho từng task
    train_images = {}
    for task_name, classes in task_splits.items():
        task_img_ids = set()
        for ann in coco.anns.values():
            if ann['category_id'] in classes:
                task_img_ids.add(ann['image_id'])
        train_images[task_name] = sorted(list(task_img_ids))

    # Hàm lấy mẫu đại diện (exemplars) từ các lớp cũ để finetuning
    def get_exemplars(class_ids, num_images_per_class=num_exemplars_per_class):
        exemplar_ids = set()
        for cid in class_ids:
            img_ids = [ann['image_id'] for ann in coco.anns.values() if ann['category_id'] == cid]
            selected = img_ids[:num_images_per_class]
            exemplar_ids.update(selected)
        return exemplar_ids

    # 2. Tạo tập Finetuning (ft) giữ lại tri thức các task trước
    ft_images = {
        't2': sorted(list(set(train_images['t2']).union(get_exemplars(task_splits['t1'])))),
        't3': sorted(list(set(train_images['t3']).union(get_exemplars(task_splits['t1'] + task_splits['t2'])))),
        't4': sorted(list(set(train_images['t4']).union(get_exemplars(task_splits['t1'] + task_splits['t2'] + task_splits['t3']))))
    }

    os.makedirs(output_dir, exist_ok=True)

    splits = {
        't1_train': train_images['t1'],
        't2_train': train_images['t2'],
        't3_train': train_images['t3'],
        't4_train': train_images['t4'],
        't2_ft': ft_images['t2'],
        't3_ft': ft_images['t3'],
        't4_ft': ft_images['t4']
    }

    for name, img_ids in splits.items():
        path = os.path.join(output_dir, f"{name}.txt")
        with open(path, "w") as f:
            for img_id in img_ids:
                f.write(f"{img_id:06d}\n" if isinstance(img_id, int) else f"{img_id}\n")
        print(f"Đã tạo {name}.txt với {len(img_ids)} ảnh.")

if __name__ == "__main__":
    prepare_splits()
