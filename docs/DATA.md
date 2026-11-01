# Data setup

The code expects this layout (on Google Drive when using Colab). Datasets are **not** stored in this repository.

```
pose-assistant/
├── code/                         # copy of src/*.py
├── coco_keypoints/
│   ├── val2017/                  # COCO val2017 images
│   └── person_keypoints_val2017.json
├── places365/
│   └── torchvision_val/          # Places365 validation set (small, 256x256) in torchvision layout
├── mpii_dataset/
│   ├── mpii_human_pose_v1_u12_2/
│   │   └── mpii_human_pose_v1_u12_1.mat
│   └── images/                   # MPII images (~25k .jpg)
├── checkpoints/                  # scene_analyzer.pt, SceneAnalyzer.mlpackage (created by the code)
└── templates/                    # pose_templates_v1.json, template_selection.json, mpii_work/ (created by the code)
```

Base path is `BASE_DRIVE = /content/drive/MyDrive/pose-assistant` in each script; change it there if yours differs.

## Where to get the data

| Data | Used for | Source |
|---|---|---|
| COCO 2017 keypoints (val) | Benchmarking pose detection | cocodataset.org → 2017 Val images, 2017 Train/Val annotations (use `annotations/person_keypoints_val2017.json`) |
| Places365 (val, small) | Training the indoor/outdoor head | Via torchvision: `torchvision.datasets.Places365(root=..., split="val", small=True, download=True)` |
| `IO_places365.txt` | Indoor/outdoor label per Places365 category | Downloaded automatically by `02_three_models.py` from the Places365 GitHub repo |
| MPII Human Pose | Selecting pose templates | human-pose.mpi-inf.mpg.de → Download: images archive and the annotations zip (`mpii_human_pose_v1_u12_2.zip`) |

Check the download pages for current links and sizes; MPII images are a large download.

## Check your setup

After mounting Drive, the second code cell of `notebooks/Pose_Assistant_v2.ipynb` prints which of these exist
(MPII `.mat`, image count, checkpoint, template file) and warns if the template file is still the mock one.

## Licences

COCO, Places365 and MPII each have their own terms (annotation licence vs image sources differ). Read the
licence files that come with each download. This project treats them as research use only; templates keep
joint coordinates only (no images), but replace MPII-derived templates with your own photos before any
commercial use.
