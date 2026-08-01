import zipfile
import json
import shutil
import tempfile
import os

src = "cnn_v9_model.keras"
dst = "cnn_v9_model_patched.keras"

tmp = tempfile.mkdtemp()

with zipfile.ZipFile(src, "r") as z:
    z.extractall(tmp)

cfg_path = os.path.join(tmp, "config.json")

with open(cfg_path) as f:
    cfg = json.load(f)

removed = 0

def clean(obj):
    global removed
    if isinstance(obj, dict):
        if "quantization_config" in obj:
            del obj["quantization_config"]
            removed += 1
        for v in obj.values():
            clean(v)
    elif isinstance(obj, list):
        for v in obj:
            clean(v)

clean(cfg)

with open(cfg_path, "w") as f:
    json.dump(cfg, f)

with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as z:
    for root, _, files in os.walk(tmp):
        for file in files:
            full = os.path.join(root, file)
            arc = os.path.relpath(full, tmp)
            z.write(full, arc)

shutil.rmtree(tmp)

print(f"Removed {removed} quantization_config entries")
print(f"Saved {dst}")