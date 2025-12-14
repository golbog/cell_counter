import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from model.cell_counter import PytorchCellCounter


def process_image(model, image_path, output_path, threshold=0.85):
    try:
        img = Image.open(image_path).convert('L')
        img_np = (np.array(img) / 255.0).astype(np.float32)
        
        img_np = img_np[..., np.newaxis]
        
        # transpose to (C, H, W) -> (1, H, W)
        img_np = np.transpose(img_np, (2, 0, 1))
        
        mask = model.segment(img_np)
        
        filtered_mask = model.filter(mask, threshold=threshold)
        
        # filtered_mask is 0/1 (if shed=True) or 0-255 (if shed=False)
        if filtered_mask.max() <= 1:
            mask_uint8 = (filtered_mask * 255).astype(np.uint8)
        else:
            mask_uint8 = filtered_mask.astype(np.uint8)
        
        # if output is (1, H, W), squeeze it
        if mask_uint8.ndim == 3:
            mask_uint8 = np.squeeze(mask_uint8)
            
        Image.fromarray(mask_uint8).save(output_path)
        print(f"Processed: {image_path} -> {output_path}")
        
    except Exception as e:
        print(f"Error processing {image_path}: {e}")


def main():
    parser = argparse.ArgumentParser(description='Generate masks using trained model')
    parser.add_argument('--model', type=str, required=True, help='Path to the trained model file')
    parser.add_argument('--input', type=str, required=True, help='Path to image file or folder')
    parser.add_argument('--output', type=str, required=True, help='Path to output file or folder')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for filtering (0.0 - 1.0)')
    
    args = parser.parse_args()
    
    print(f"Loading model from {args.model}...")
    model = PytorchCellCounter.load(args.model)
    if model is None:
        print("Failed to load model.")
        return

    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if input_path.is_file():
        if output_path.suffix == '': # output is a folder
             output_path.mkdir(parents=True, exist_ok=True)
             output_file = output_path / input_path.name
        else:
             output_file = output_path
             output_file.parent.mkdir(parents=True, exist_ok=True)
        
        process_image(model, input_path, output_file, threshold=args.threshold)
        
    elif input_path.is_dir():
        if output_path.suffix != '': # output is a file, but input is dir
            print("Error: Input is a directory, but output is a file.")
            return
            
        output_path.mkdir(parents=True, exist_ok=True)
        
        extensions = ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff', '*.bmp']
        files = []
        for ext in extensions:
            files.extend(input_path.glob(ext))
            
        for file in files:
            output_file = output_path / file.name
            process_image(model, file, output_file, threshold=args.threshold)
            
    else:
        print(f"Error: Input path {input_path} does not exist.")

if __name__ == '__main__':
    main()
