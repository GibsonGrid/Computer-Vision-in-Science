
import os
import shutil
import zipfile



# Base directory where the dataset was extracted
data_dir = r"Z:\Road detection_mohammed_exps\Data"


zip_path = os.path.join(data_dir, "MMU_orginal.zip")









# ---------------------------------------------------------
# Extract ZIP contents
# ---------------------------------------------------------
print("Extracting ZIP file...")
with zipfile.ZipFile(zip_path, "r") as zip_ref:
    zip_ref.extractall(data_dir)

print("Extraction completed.")
print(f"Files extracted to: {data_dir}")





# # Old and new main folder names
# old_main = os.path.join(data_dir, "MMU_orginal")
# new_main = os.path.join(data_dir, "eye_ds")

# # Rename main dataset folder
# if os.path.exists(old_main):
#     os.rename(old_main, new_main)
#     print(f"Renamed folder:\n  {old_main}  -->  {new_main}")
# else:
#     raise FileNotFoundError(f"{old_main} not found.")

# # Paths for subfolders
# old_annotations = os.path.join(new_main, "annotation")
# new_annotations = os.path.join(new_main, "labels")

# old_frames = os.path.join(new_main, "fullFrames")
# new_frames = os.path.join(new_main, "images")

# # Rename annotation → labels
# if os.path.exists(old_annotations):
#     os.rename(old_annotations, new_annotations)
#     print(f"Renamed folder:\n  {old_annotations}  -->  {new_annotations}")
# else:
#     print("annotation folder not found.")

# # Rename fullFrames → images
# if os.path.exists(old_frames):
#     os.rename(old_frames, new_frames)
#     print(f"Renamed folder:\n  {old_frames}  -->  {new_frames}")
# else:
#     print("fullFrames folder not found.")

# # -------------------------------------------------------
# # Move all PNG label files from labels/png/ → labels/
# # -------------------------------------------------------
# png_dir = os.path.join(new_annotations, "png")

# if os.path.exists(png_dir):
#     for fname in os.listdir(png_dir):
#         src = os.path.join(png_dir, fname)
#         dst = os.path.join(new_annotations, fname)
#         shutil.move(src, dst)
#         print(f"Moved: {src}  -->  {dst}")

#     # Remove the now-empty png directory
#     os.rmdir(png_dir)
#     print(f"Removed folder: {png_dir}")
# else:
#     print("No png/ folder found inside labels.")
