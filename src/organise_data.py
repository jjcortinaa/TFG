import os
import shutil

# Root path where your data is currently located
base_path = "data/processed"

# Mapping between filename keywords and your new folder names
# Based on Martinez-Paya (2017) nomenclature
muscle_mapping = {
    "BBr": "Bicep",
    "Cdr": "Quadriceps",
    "FxM": "Antebrazo",
    "TbA": "Tibial",
}

# The two root labels to process
labels = ["0_Control", "1_ELA"]


def organize_images():
    for label in labels:
        source_folder = os.path.join(base_path, label)

        # Check if the root label folder exists
        if not os.path.exists(source_folder):
            print(f"Skipping {source_folder}, folder not found.")
            continue

        # List all files in the current root label folder
        files = [
            f
            for f in os.listdir(source_folder)
            if os.path.isfile(os.path.join(source_folder, f))
        ]

        for file_name in files:
            moved = False
            for key, target_folder in muscle_mapping.items():
                if key in file_name:
                    dest_dir = os.path.join(base_path, target_folder, label)

                    os.makedirs(dest_dir, exist_ok=True)

                    shutil.move(
                        os.path.join(source_folder, file_name),
                        os.path.join(dest_dir, file_name),
                    )
                    print(f"Moved: {file_name} -> {target_folder}/{label}/")
                    moved = True
                    break

            if not moved:
                print(f"Warning: No muscle keyword found in {file_name}")


if __name__ == "__main__":
    print("Starting reorganization...")
    organize_images()
    print("Done!")
