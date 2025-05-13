import numpy as np
from ultralytics.data.utils import visualize_image_annotations
from ultralytics.utils.ops import segments2boxes


def convert_seg_to_bbox(file_path: str):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        items = list(map(float, line.strip().split()))
        cls_id = int(items[0])
        points = np.array([items[1:]])

        bbox = segments2boxes([s.reshape(-1, 2) for s in points])[0]

        new_line = f"{cls_id} {' '.join(map(str, bbox))}\n"
        new_lines.append(new_line)

    with open(file_path, 'w') as f:
        f.writelines(new_lines)


def visualize_image(label_map: dict[int, str], image_path: str, label_path: str):
    return visualize_image_annotations(image_path, label_path, label_map)
