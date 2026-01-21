import os
import cv2
import numpy as np
from tqdm import tqdm
from config import Config

def preprocess_tif_segments():
    """
    Core preprocessing script. 
    It reads the .tif ROI segments from your specific folders, 
    resizes them to 224x224 and saves them in the processed folder
    """
    # Initialize directory structure and random seeds
    Config.setup_project()
    
    # Use your exact folder names
    folders = ['0_Control', '1_ELA']
    muscle_subfolders = ['01_BBr', '02_FxM', '03_Cdr', '04_TbA']

    print(f"[INFO] Starting TIF segment processing on {Config.DEVICE}...")

    for folder in folders:
        # Path to the source segments: data/0_Control/segments/ or data/1_ELA/segments/
        segments_base_path = os.path.join(Config.BASE_DIR, "data/classified_data", folder, "segments")
        
        # Output path mirrors your names: data/processed/0_Control/ or data/processed/1_ELA/
        output_class_path = os.path.join(Config.PROCESSED_DATA_PATH, folder)
        os.makedirs(output_class_path, exist_ok=True)

        for muscle in muscle_subfolders:
            muscle_path = os.path.join(segments_base_path, muscle)
            
            if not os.path.exists(muscle_path):
                print(f"[WARNING] Path not found: {muscle_path}")
                continue

            print(f"  Processing {folder} - {muscle}...")
            
            # Filter only .tif files
            tif_files = [f for f in os.listdir(muscle_path) if f.lower().endswith('.tif')]

            for filename in tqdm(tif_files):
                file_path = os.path.join(muscle_path, filename)
                
                # 1. Load the TIF segment
                img = cv2.imread(file_path)
                
                if img is None:
                    continue

                # 2. Resize to 224x224 (Standard for AI models)
                resized_img = cv2.resize(img, Config.IMAGE_SIZE, interpolation=cv2.INTER_AREA)

                # 3. Save to processed folder
                # We keep the muscle name in the filename to distinguish between them
                clean_name = f"{filename.replace('.tif', '_clean.jpg')}"
                save_path = os.path.join(output_class_path, clean_name)
                
                # Save as high-quality JPG for the training phase
                cv2.imwrite(save_path, resized_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    print(f"\n[SUCCESS] Preprocessing complete.")
    print(f"[INFO] Processed images are saved in: {Config.PROCESSED_DATA_PATH}")

if __name__ == "__main__":
    preprocess_tif_segments()