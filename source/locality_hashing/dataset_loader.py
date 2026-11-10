from collections import defaultdict
import os
import tarfile
import urllib.request

out_dir = "eval_images"
os.makedirs(out_dir, exist_ok=True)

# Clear out the previous single-class images
for f in os.listdir(out_dir):
  if f.endswith(".jpg"):
    os.remove(os.path.join(out_dir, f))

url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"
tar_path = "temp_eval.tgz"

if not os.path.exists(tar_path):
  print("Downloading archive (~10 MB)...")
  urllib.request.urlretrieve(url, tar_path)

print("Extracting 10 balanced images per class (100 total)...")
with tarfile.open(tar_path, "r:gz") as tar:
  # Group validation files by their class folder name
  class_files = defaultdict(list)
  for member in tar.getmembers():
    if member.name.endswith(".JPEG") and "/val/" in member.name:
      class_id = member.name.split("/val/")[1].split("/")[0]
      class_files[class_id].append(member)

  total_saved = 0
  for class_id, members in sorted(class_files.items()):
    for idx, member in enumerate(members[:10]):  # 10 images per class
      f = tar.extractfile(member)
      if f is not None:
        out_path = os.path.join(
            out_dir, f"sample_c{class_id}_{total_saved:03d}.jpg"
        )
        with open(out_path, "wb") as img_file:
          img_file.write(f.read())
        total_saved += 1

if os.path.exists(tar_path):
  os.remove(tar_path)

print(f"Done! {total_saved} diverse images saved across 10 distinct classes.")