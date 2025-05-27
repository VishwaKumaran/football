import os
import shutil

if __name__ == '__main__':
    SRC_DIR = "/Users/vishwa/Documents/Projects/football/model/dataset-keypoints"
    DST_DIR = "/Users/vishwa/Documents/Projects/football/model/dataset-test-keypoints"
    os.makedirs(DST_DIR, exist_ok=True)

    FLIP_IDX = [
        24,
        25,
        26,
        27,
        28,
        29,
        22,
        23,
        21,
        17,
        18,
        19,
        20,
        13,
        14,
        15,
        16,
        9,
        10,
        11,
        12,
        8,
        6,
        7,
        0,
        1,
        2,
        3,
        4,
        5,
        31,
        30,
    ]

    MAPPING = {
        0: 0,
        1: 1,
        2: 9,
        3: 12,
        4: 4,
        5: 5,
        6: 13,
        7: 14,
        8: 30,
        9: 15,
        10: 31,
        11: 16,
        12: 29,
        13: 28,
        14: 20,
        15: 17,
        16: 25,
        17: 24,
    }


    # === CONFIG ===
    def convert_annotation(src_file: str, dst_file: str):
        with open(src_file, 'r') as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            class_id = parts[0]
            x = parts[1]
            y = parts[2]
            width = parts[3]
            height = parts[4]
            kpts = list(map(float, parts[5:]))

            keypoints_18 = [kpts[i:i + 3] for i in range(0, len(kpts), 3)]
            keypoints_32 = [[0.0, 0.0, 0.0] for _ in range(32)]

            for i_18, i_32 in MAPPING.items():
                keypoints_32[i_32] = keypoints_18[i_18]

            flat_kpts = [str(v) for kp in keypoints_32 for v in kp]
            new_lines.append(f"{class_id} {x} {y} {width} {height} " + " ".join(flat_kpts))

        with open(dst_file, 'w') as f:
            f.write("\n".join(new_lines))


    for split in ["train", "valid", "test"]:
        src_labels = os.path.join(SRC_DIR, split, "labels")
        src_images = os.path.join(SRC_DIR, split, "images")
        dst_labels = os.path.join(DST_DIR, split, "labels")
        dst_images = os.path.join(DST_DIR, split, "images")

        os.makedirs(dst_labels, exist_ok=True)
        os.makedirs(dst_images, exist_ok=True)

        for fname in os.listdir(src_labels):
            if fname.endswith(".txt"):
                convert_annotation(
                    os.path.join(src_labels, fname),
                    os.path.join(dst_labels, fname)
                )

        for fname in os.listdir(src_images):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                shutil.copy(
                    os.path.join(src_images, fname),
                    os.path.join(dst_images, fname)
                )

    print(f"Converted all .txt files from 18 to 32 keypoints in: {DST_DIR}")

    # === WRITE YAML ===
    yaml_content = """
train: ../train/images
val: ../valid/images
test: ../test/images

kpt_shape: [32, 3]
flip_idx: [24, 25, 26, 27, 28, 29, 22, 23, 21, 17, 18, 19, 20, 13, 14, 
           15, 16, 9, 10, 11, 12, 8, 6, 7, 0, 1, 2, 3, 4, 5, 31, 30]

nc: 1
names: ['pitch']
    """

    with open(f"{DST_DIR}/data.yaml", "w") as f:
        f.write(yaml_content.strip())

    print("Created dataset.yaml with 32 keypoints config.")
