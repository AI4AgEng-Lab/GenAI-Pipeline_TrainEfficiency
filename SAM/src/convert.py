import argparse
import os
import torch
from SAM.SAMImageProcessor.SAM_image_processor import SAMImageProcessor
from segment_anything import sam_model_registry, SamPredictor

class SAMProcessor:
    def __init__(self, checkpoint_path, device, model_type, image_folder, label_folder, label_out_folder, yaml_file, batch_size):
        """
        Initializes the SAM model and image processor.
        """
        self.checkpoint_path = checkpoint_path
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model_type = model_type
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.label_out_folder = label_out_folder
        self.yaml_file = yaml_file
        self.batch_size = batch_size

        # Load SAM model
        print(f"Loading SAM model from {self.checkpoint_path}...")
        if not os.path.isfile(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file {self.checkpoint_path} not found!")

        self.sam = sam_model_registry[self.model_type](checkpoint=self.checkpoint_path).to(self.device)
        self.predictor = SamPredictor(self.sam)

        # Initialize Image Processor
        self.sam_image_processor = SAMImageProcessor(
            sam_predictor=self.predictor,
            image_folder=self.image_folder,
            label_folder=self.label_folder,
            label_out_folder=self.label_out_folder,
            yaml_file=self.yaml_file,
            batch_size=self.batch_size
        )

    def apply_sam(self):
        """Applies the SAM model to process all image batches."""
        print("Applying SAM to all image batches...")
        self.sam_image_processor.apply_sam_to_all_batches()
        print("Processing complete.")

def main():
    """Parses command-line arguments and runs the SAMProcessor."""
    parser = argparse.ArgumentParser(description="Apply SAM model to images for segmentation.")
    
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the SAM model checkpoint file.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run the model on (e.g., 'cuda:0' or 'cpu').")
    parser.add_argument("--model_type", type=str, default="vit_h", help="Model type (e.g., 'vit_h').")
    parser.add_argument("--image_folder", type=str, required=True, help="Path to the folder containing images.")
    parser.add_argument("--label_folder", type=str, required=True, help="Path to the folder containing labels.")
    parser.add_argument("--label_out_folder", type=str, required=True, help="Path to save the segmented labels.")
    parser.add_argument("--yaml_file", type=str, required=True, help="Path to the dataset YAML configuration file.")
    parser.add_argument("--batch_size", type=int, default=10, help="Number of images to process per batch.")

    args = parser.parse_args()

    # Initialize and run the SAM processor
    sam_processor = SAMProcessor(
        checkpoint_path=args.checkpoint_path,
        device=args.device,
        model_type=args.model_type,
        image_folder=args.image_folder,
        label_folder=args.label_folder,
        label_out_folder=args.label_out_folder,
        yaml_file=args.yaml_file,
        batch_size=args.batch_size
    )

    # Run segmentation
    sam_processor.apply_sam()

if __name__ == "__main__":
    main()

"""python convert.py --checkpoint_path "D:/JSA_Rep/sam_vit_h_4b8939.pth" \
                  --device "cuda:0" \
                  --model_type "vit_h" \
                  --image_folder "D:/JSA_Rep/Wheat_pallet_YOLO/images" \
                  --label_folder "D:/JSA_Rep/Wheat_pallet_YOLO/labels" \
                  --label_out_folder "D:/JSA_Rep/Wheat_pallet_YOLO/seg_lbl" \
                  --yaml_file "D:/JSA_Rep/Wheat_pallet_YOLO/dataset.yaml" \
                  --batch_size 10
    for windows cmd
    python convert.py --checkpoint_path "D:\JSA_Rep\sam_vit_h_4b8939.pth" `
                  --device "cuda:0" `
                  --model_type "vit_h" `
                  --image_folder "D:\JSA_Rep\Wheat_pallet_YOLO\images" `
                  --label_folder "D:\JSA_Rep\Wheat_pallet_YOLO\labels" `
                  --label_out_folder "D:\JSA_Rep\Wheat_pallet_YOLO\seg_lbl" `
                  --yaml_file "D:\JSA_Rep\Wheat_pallet_YOLO\dataset.yaml" `
                  --batch_size 10
"""