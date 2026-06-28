# MISDD-MM Extended — Windows Setup Guide (via WSL2)

This guide walks through setting up this project from scratch on a fresh Windows
machine. It is written from real experience setting this project up after a full
disk failure — every step here exists because something actually broke without it.
Follow it in order and you should avoid the issues documented below.

---

## 1. Install WSL2 with Ubuntu 22.04

Open **PowerShell as Administrator** and run:

```powershell
wsl --install -d Ubuntu-22.04
```

Restart your PC when prompted. After restart, open **Ubuntu 22.04** from the Start
menu. The first launch takes a little while — it will ask you to create a Linux
username and password. Pick something simple; you'll type it often.

**If `wsl --install` fails with "component store corrupted":** this is a Windows
servicing issue, not a WSL issue. Run, as Administrator:

```powershell
sfc /scannow
DISM /Online /Cleanup-Image /RestoreHealth
```

Then retry the install.

**If `wsl -d Ubuntu-22.04` fails with `fopen(/etc/default/locale) failed 5` or
similar I/O errors immediately after a working install:** check whether the
**"Windows Subsystem for Linux"** optional Windows feature is actually enabled:

```powershell
Get-WindowsOptionalFeature -Online -FeatureName Microsoft-Windows-Subsystem-Linux
```

If it shows `Disabled`, enable it and restart:

```powershell
Enable-WindowsOptionalFeature -Online -FeatureName Microsoft-Windows-Subsystem-Linux -All
```

---

## 2. Verify GPU passthrough works before doing anything else

Inside the Ubuntu terminal:

```bash
nvidia-smi
```

You should see your GPU listed with driver/CUDA info. **You do not need to install
an NVIDIA driver inside WSL** — WSL2 uses the driver already installed on the
Windows side. If `nvidia-smi` isn't found at this path later in a fresh terminal,
try the full path directly:

```bash
/usr/lib/wsl/lib/nvidia-smi
```

---

## 3. Install Miniconda

```bash
cd ~
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

Accept the license, accept the default install path, and **say `yes`** when asked
to initialize conda in your shell. Then **close the terminal completely and reopen
it** — conda will not work until you do this.

```bash
conda --version
```

---

## 4. Install CUDA Toolkit 11.8 and build tools

```bash
cd ~
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run --silent --toolkit

echo 'export PATH=/usr/local/cuda-11.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

sudo apt update
sudo apt install -y build-essential gcc-11 g++-11 git unzip tmux
```

Verify: `nvcc --version` should report release 11.8.

---

## 5. Clone the repo and set up the conda environment

```bash
git clone https://github.com/KsKarim7/Reliability-Aware-Multimodal-Surface-Defect-Detection-Under-Uncertain-Sensor-Availability.git MISDD-MM
cd MISDD-MM
```

```bash
conda create -n misdd_mm python=3.11 -y
conda activate misdd_mm
pip install torch==2.2.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Immediately pin NumPy below version 2** — this project's compiled extensions and
several dependencies (OpenCV, scikit-image) are not compatible with NumPy 2.x, and
several packages installed later will silently pull NumPy 2.x back in as a
dependency. Re-run this command any time NumPy drifts:

```bash
pip install "numpy<2" --force-reinstall
```

Install the remaining dependencies:

```bash
pip install pandas "opencv-python<4.10" matplotlib seaborn scipy loguru \
    open_clip_torch timm einops wandb pyyaml scikit-image scikit-learn \
    tifffile imageio ninja
pip install "numpy<2" --force-reinstall
```

Run this check after every batch install — it's the fastest way to catch a silent
NumPy/OpenCV version conflict before it wastes hours of training time:

```bash
python -c "
import numpy; print('NumPy:', numpy.__version__)
import cv2; print('OpenCV:', cv2.__version__)
import torch; print('PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available())
"
```

`NumPy` must read `1.26.x`. If it ever shows `2.x`, run
`pip install "numpy<2" --force-reinstall` again — most likely whatever you just
installed (commonly `opencv-python` or `pandas`) pulled in NumPy 2 as a dependency.

---

## 6. Build the Pointnet2 CUDA extension

```bash
cd ~/MISDD-MM/Pointnet2_PyTorch/pointnet2_ops_lib
export TORCH_CUDA_ARCH_LIST="8.9"   # RTX 40-series. Use "8.6" for A6000/30-series.
pip install -e . --no-build-isolation
```

The `--no-build-isolation` flag is required — without it, pip builds this package
in a sandboxed environment that can't see your already-installed PyTorch, and the
build fails with `ModuleNotFoundError: No module named 'torch'` even though PyTorch
is clearly installed.

Make the architecture flag permanent:

```bash
echo 'export TORCH_CUDA_ARCH_LIST="8.9"' >> ~/.bashrc
```

Verify:

```bash
cd ~/MISDD-MM
python -c "from pointnet2_ops import pointnet2_utils; print('OK')"
```

---

## 7. A known dependency that no longer exists — `knn_cuda`

The original code imports a package called `knn_cuda` from
`github.com/unlimblue/KNN_CUDA`. **That GitHub repository has been deleted** and
no longer exists. Do not waste time trying to `pip install` or `git clone` it —
it will fail every time with a 404.

This repo already includes the fix: `knn_cuda_replacement.py` at the project root
is a from-scratch, dependency-free replacement using only `torch.cdist` and
`torch.topk`, producing mathematically equivalent output. `point_transformer.py`
already imports from it. You don't need to do anything here — just don't be
confused if you ever see the old import name mentioned in old logs or comments.

If you ever clone this project onto a machine with a **different username**, the
hardcoded path in `point_transformer.py`'s import line needs updating to match:

```python
import sys; sys.path.insert(0, "/home/<your-username>/MISDD-MM"); from knn_cuda_replacement import KNN
```

---

## 8. Fix the username-dependent dataset paths

`datasets/mvtec3d.py` and `datasets/eyescandies.py` both hardcode an absolute path
containing a specific Linux username. Update both to match your actual username
(check with `whoami`):

```bash
whoami
sed -i 's|/home/[^/]*/mvtec3d|/home/YOUR_USERNAME/mvtec3d|' ~/MISDD-MM/datasets/mvtec3d.py
sed -i 's|/home/[^/]*/eyescandies/Eyecandies|/home/YOUR_USERNAME/eyescandies/Eyecandies|' ~/MISDD-MM/datasets/eyescandies.py
```

---

## 9. Download the datasets

Datasets are not stored in git — download them fresh each time.

**MVTec 3D-AD** (current download link rotates; get it fresh from
`https://www.mvtec.com/company/research/datasets/mvtec-3d-ad/downloads` if this
one has expired):

```bash
mkdir -p ~/mvtec3d && cd ~/mvtec3d
wget "<current MVTec 3D-AD download URL>"
tar -xf mvtec_3d_anomaly_detection.tar.xz
```

**Eyecandies** (hosted on Google Drive — get current file IDs from
`https://eyecan-ai.github.io/eyecandies/download`):

```bash
pip install gdown
mkdir -p ~/eyescandies/Eyecandies && cd ~/eyescandies/Eyecandies
gdown "https://drive.google.com/uc?id=<FILE_ID>" -O CandyCane.zip
# repeat for all 10 categories
```

**Important — these files are actually `.tar` archives despite the `.zip`
extension.** Use `tar`, not `unzip`:

```bash
for f in CandyCane ChocolateCookie ChocolatePraline Confetto GummyBear \
         HazelnutTruffle LicoriceSandwich Lollipop Marshmallow PeppermintCandy; do
  tar -xf "${f}.zip"
done
```

Each archive extracts into a nested `Eyecandies/<CategoryName>/` subfolder rather
than directly into the category folder. Flatten it:

```bash
mv Eyecandies/* .
rmdir Eyecandies
rm *.zip
```

You should end up with `~/eyescandies/Eyecandies/<CategoryName>/{train,val,test_public,test_private}/`
for all 10 categories, with no leftover `.zip` files or nested `Eyecandies/`
directories.

If a single category's Google Drive download fails with "too many users have
viewed or downloaded this file recently," download it manually from a browser
instead and copy it in from Windows:

```bash
cp /mnt/c/Users/<your-windows-username>/Downloads/<Category>.tar ~/eyescandies/Eyecandies/
```

---

## 10. Always pass `--gpu-id 0` explicitly

`train_cls.py` defaults to `--gpu-id 1`. On a single-GPU machine, your GPU is
device index **0**, and this default will produce a confusing
`RuntimeError: No CUDA GPUs are available` error that looks exactly like a driver
or WSL problem — even though CUDA itself is working fine. This cost a significant
amount of debugging time to trace back to one default argument value.

**Always include `--gpu-id 0` on any single-GPU machine:**

```bash
python train_cls.py --dataset mvtec3d --class_name bagel \
    --missing_type both --missing_rate 0.7 --seed 111 --gpu-id 0
```

Every script in this repo already includes this flag where needed. If you write a
new script, don't forget it.

---

## 11. Running long training jobs without losing them

A training sweep across all classes and seeds takes hours. A few real failure
modes to know about, all observed firsthand on this exact setup:

- **`tail -f some_log.txt` followed by Ctrl+C does not stop training** — but get
  in the habit of using `tail -n 30 file.txt` (no `-f`) instead, so there is never
  anything in the foreground to accidentally interrupt.
- **Locking the Windows screen can interrupt the WSL session** in a way that
  kills background training, even with sleep disabled. Either leave the screen
  unlocked, or just let the display turn off naturally — that does *not* kill
  anything.
- **GPU power capping under sustained load can cause
  `CUDA error: CUBLAS_STATUS_EXECUTION_FAILED`.** If this happens, run
  `wsl --shutdown` from PowerShell, wait 15 seconds, reopen Ubuntu, and check power
  cap status before relaunching:
```bash
  /usr/lib/wsl/lib/nvidia-smi -q -d TEMPERATURE,PERFORMANCE | grep -A3 "Power Cap"
```
  In Windows, set NVIDIA Control Panel → Manage 3D Settings → Power management
  mode to **"Prefer Maximum Performance."**

**Always launch long jobs like this**, so they survive a closed terminal:

```bash
nohup bash your_script.sh > your_log.txt 2>&1 &
disown
```

**Always verify a run actually completed and didn't silently skip a category.**
A script reporting "SAVED" and "COMPLETE" does not guarantee every row is real —
check for zero-value rows before trusting any result file:

```bash
awk -F',' 'NR>1 && ($2==0 || $3==0 || $4==0) {print "ZERO FOUND:", $0}' path/to/results.csv
```

If a category silently produced a `0.00` row with no visible error, the
individual class probably still works fine in isolation — run it standalone,
let it finish completely, and manually patch the correct value back into the CSV.

---

## 12. Known result: seed-to-seed variance is much higher on Eyecandies than MVTec

This is expected, not a bug. MVTec 3D-AD shows under 1 percentage point of
standard deviation across seeds 111/222/333. Eyecandies shows roughly 5 points of
standard deviation across the same three seeds, with seed 333 in particular
landing notably higher than 111/222 across every ablation configuration. This was
investigated thoroughly — the new machine reproduces historical results within
0.2 points on a direct seed/category comparison, ruling out a hardware or
environment cause. The most likely explanation is Eyecandies' smaller per-category
test sets making individual seeds more impactful on the aggregate metric. Report
this variance honestly rather than treating any single seed as definitive.

---

## 13. Quick reference — daily startup

Once everything above is done once, your normal day-to-day startup is:

```powershell
wsl -d Ubuntu-22.04
```

Add this to your `~/.bashrc` once so every new terminal lands you in the right
place automatically:

```bash
echo "cd ~/MISDD-MM" >> ~/.bashrc
echo "conda activate misdd_mm" >> ~/.bashrc
```

After that, opening WSL alone gets you straight to a ready-to-work terminal.
