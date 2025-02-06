import os
import cv2
import argparse
import numpy as np

def create_mask(image_shape, polygon):
    """Creates a binary mask from polygon coordinates."""
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)
    return mask

def isolate_objects(image_folder, annotations_folder, output_folder):
    """Processes images and extracts objects into 512x512 canvases."""
    
    class_names = ['HoPla_Cirsium', 'HoPla_Convlvulus', 'HoPla_Echinochloa', 'HoPla_Fallopia', 'HoPla_Sugarbeet']

    for class_name in class_names:
        os.makedirs(os.path.join(output_folder, class_name.lower()), exist_ok=True)

    for filename in os.listdir(image_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(image_folder, filename)
            annotations_path = os.path.join(annotations_folder, os.path.splitext(filename)[0] + '.txt')

            image = cv2.imread(image_path)
            if image is None:
                print(f"Could not read image {image_path}. Skipping.")
                continue

            if not os.path.exists(annotations_path):
                print(f"No annotation file for {filename}. Skipping.")
                continue

            with open(annotations_path, 'r') as file:
                lines = file.readlines()
                for i, line in enumerate(lines):
                    parts = line.strip().split()
                    class_id = int(parts[0])
                    coordinates = list(map(float, parts[1:]))

                    height, width, _ = image.shape
                    polygon = np.array([
                        (int(coordinates[j] * width), int(coordinates[j+1] * height))
                        for j in range(0, len(coordinates), 2)
                    ], dtype=np.int32)

                    mask = create_mask(image.shape[:2], polygon)
                    isolated_object = cv2.bitwise_and(image, image, mask=mask)

                    x_min, x_max = polygon[:, 0].min(), polygon[:, 0].max()
                    y_min, y_max = polygon[:, 1].min(), polygon[:, 1].max()
                    object_width = x_max - x_min
                    object_height = y_max - y_min

                    canvas_size = 512
                    padded_object = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
                    start_x = max((canvas_size - object_width) // 2, 0)
                    start_y = max((canvas_size - object_height) // 2, 0)

                    obj_x_end = min(object_width, canvas_size - start_x)
                    obj_y_end = min(object_height, canvas_size - start_y)

                    padded_object[start_y:start_y+obj_y_end, start_x:start_x+obj_x_end] = isolated_object[y_min:y_min+obj_y_end, x_min:x_min+obj_x_end]

                    class_name = class_names[class_id]
                    object_output_path = os.path.join(output_folder, class_name.lower(), f"{os.path.splitext(filename)[0]}_object_{i}.jpg")
                    cv2.imwrite(object_output_path, padded_object)
                    print(f"Object {i} in {filename} saved as {object_output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract objects from images using polygon masks.")
    
    parser.add_argument("--image_folder", type=str, required=True, help="Path to the input image folder.")
    parser.add_argument("--annotations_folder", type=str, required=True, help="Path to YOLO segmentation labels.")
    parser.add_argument("--output_folder", type=str, required=True, help="Folder to save extracted objects.")

    args = parser.parse_args()

    isolate_objects(args.image_folder, args.annotations_folder, args.output_folder)
