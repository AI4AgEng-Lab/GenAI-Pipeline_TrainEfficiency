#!/bin/bash

# Define an array of dataset names
datasets=("sug_real" "syn10_real90" "syn20_real80" "syn30_real70" "syn40_real60" "syn50_real50" \
          "syn60_real40" "syn70_real30" "syn80_real20" "syn90_real10" )
#datasets=("sug_real")

# Iterate over each dataset for YOLOv8n training
for dataset_name in "${datasets[@]}"
do
    echo "Training YOLOv8n on dataset: $dataset_name"
    
    # Run the training command for YOLOv8n
    yolo task=detect mode=train model=yolov8n.pt \
    data=/home/sourav/stab-diff/stab-diff/YOLO/DATASET_SUG_V2/cross_val/set_1/$dataset_name/dataset.yaml \
    epochs=300 patience=40 imgsz=640 batch=16 lr0=0.01 device=0 cos_lr=True workers=2 \
    hsv_h=0 hsv_s=0 hsv_v=0 degrees=0 translate=0 scale=0 shear=0 flipud=0 fliplr=0 perspective=0 \
    mosaic=0 mixup=0 erasing=0 auto_augment=None bgr=0 crop_fraction=0 copy_paste=0 \
    project=yolov8n_$dataset_name plots=True

    echo "Finished training YOLOv8n on dataset: $dataset_name"
    echo "============================================="

done

# Iterate over each dataset for YOLOv9t training
for dataset_name in "${datasets[@]}"
do
    echo "Training YOLOv9t on dataset: $dataset_name"
    
    # Run the training command for YOLOv9t
    yolo task=detect mode=train model=yolov9t.pt \
    data=/home/sourav/stab-diff/stab-diff/YOLO/DATASET_SUG_V2/cross_val/set_1/$dataset_name/dataset.yaml \
    epochs=300 patience=40 imgsz=640 batch=16 lr0=0.01 device=0 cos_lr=True workers=2 \
    hsv_h=0 hsv_s=0 hsv_v=0 degrees=0 translate=0 scale=0 shear=0 flipud=0 fliplr=0 perspective=0 \
    mosaic=0 mixup=0 erasing=0 auto_augment=None bgr=0 crop_fraction=0 copy_paste=0 \
    project=yolov9t_$dataset_name plots=True

    echo "Finished training YOLOv9t on dataset: $dataset_name"
    echo "============================================="

done

# Iterate over each dataset for YOLOv10n training
for dataset_name in "${datasets[@]}"
do
    echo "Training YOLOv10n on dataset: $dataset_name"
    
    # Run the training command for YOLOv10n
    yolo task=detect mode=train model=yolov10n.pt \
    data=/home/sourav/stab-diff/stab-diff/YOLO/DATASET_SUG_V2/cross_val/set_1/$dataset_name/dataset.yaml \
    epochs=300 patience=40 imgsz=640 batch=16 lr0=0.01 device=0 cos_lr=True workers=2 \
    hsv_h=0 hsv_s=0 hsv_v=0 degrees=0 translate=0 scale=0 shear=0 flipud=0 fliplr=0 perspective=0 \
    mosaic=0 mixup=0 erasing=0 auto_augment=None bgr=0 crop_fraction=0 copy_paste=0 \
    project=yolov10n_$dataset_name plots=True

    echo "Finished training YOLOv10n on dataset: $dataset_name"
    echo "============================================="

done

# Iterate over each dataset for YOLOv8s training
for dataset_name in "${datasets[@]}"
do
    echo "Training YOLOv8s on dataset: $dataset_name"
    
    # Run the training command for YOLOv8s
    yolo task=detect mode=train model=yolov8s.pt \
    data=/home/sourav/stab-diff/stab-diff/YOLO/DATASET_SUG_V2/cross_val/set_1/$dataset_name/dataset.yaml \
    epochs=300 patience=40 imgsz=640 batch=16 lr0=0.01 device=0 cos_lr=True workers=2 \
    hsv_h=0 hsv_s=0 hsv_v=0 degrees=0 translate=0 scale=0 shear=0 flipud=0 fliplr=0 perspective=0 \
    mosaic=0 mixup=0 erasing=0 auto_augment=None bgr=0 crop_fraction=0 copy_paste=0 \
    project=yolov8s_$dataset_name plots=True

    echo "Finished training YOLOv8s on dataset: $dataset_name"
    echo "============================================="

done

# Iterate over each dataset for YOLOv9s training
for dataset_name in "${datasets[@]}"
do
    echo "Training YOLOv9s on dataset: $dataset_name"
    
    # Run the training command for YOLOv9s
    yolo task=detect mode=train model=yolov9s.pt \
    data=/home/sourav/stab-diff/stab-diff/YOLO/DATASET_SUG_V2/cross_val/set_1/$dataset_name/dataset.yaml \
    epochs=300 patience=40 imgsz=640 batch=16 lr0=0.01 device=0 cos_lr=True workers=2 \
    hsv_h=0 hsv_s=0 hsv_v=0 degrees=0 translate=0 scale=0 shear=0 flipud=0 fliplr=0 perspective=0 \
    mosaic=0 mixup=0 erasing=0 auto_augment=None bgr=0 crop_fraction=0 copy_paste=0 \
    project=yolov9s_$dataset_name plots=True

    echo "Finished training YOLOv9s on dataset: $dataset_name"
    echo "============================================="

done

# Iterate over each dataset for YOLOv10s training
for dataset_name in "${datasets[@]}"
do
    echo "Training YOLOv10s on dataset: $dataset_name"
    
    # Run the training command for YOLOv10s
    yolo task=detect mode=train model=yolov10s.pt \
    data=/home/sourav/stab-diff/stab-diff/YOLO/DATASET_SUG_V2/cross_val/set_1/$dataset_name/dataset.yaml \
    epochs=300 patience=40 imgsz=640 batch=16 lr0=0.01 device=0 cos_lr=True workers=2 \
    hsv_h=0 hsv_s=0 hsv_v=0 degrees=0 translate=0 scale=0 shear=0 flipud=0 fliplr=0 perspective=0 \
    mosaic=0 mixup=0 erasing=0 auto_augment=None bgr=0 crop_fraction=0 copy_paste=0 \
    project=yolov10s_$dataset_name plots=True

    echo "Finished training YOLOv10s on dataset: $dataset_name"
    echo "============================================="

done

echo "All datasets training completed for YOLOv8n, YOLOv9t, YOLOv10n, YOLOv8s, YOLOv9s, and YOLOv10s."
