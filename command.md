## transfer

```bash
python generate_gt_keypoints.py \
  --input ~/Downloads/BlenderporcGenaratedDataset/box_condition/bop_data/hb/train_pbr \
  --output ../dataset/dataset_box_pose/gt_keypoints \
  --models-dir ~/Downloads/DTTD_obj/Apple_pick \
  --ply-unit m \
  --translation-unit mm \
  --obj-ids 1 4 
```

## visualize

```bash
python visualize.py \
    --dataset /home/yeo/Downloads/pose_refine/dataset/dataset_box_pose/gt_keypoints \
    --label-type gt \
    --num-samples 30
```

## merge

```bash
python merge_keypoint_dataset.py \
    --inputs \
        /home/yeo/Downloads/pose_refine/dataset/dataset_box_pose/gt_keypoints \
        /home/yeo/Downloads/pose_refine/dataset/dataset_random/gt_keypoints \
    --output \
        /home/yeo/Downloads/pose_refine/dataset/dataset_merged/gt_keypoints
```

## train

```bash
python train.py --config config.yaml
```

## test

```bash
python test.py \
    --config config.yaml \
    --checkpoint runs/keypoint_refinement/best.pth \
    --split validation \
    --samples 0 \
    --device cuda
```

## analysis
```bash
python analyze_history.py --history runs/keypoint_refinement/history.json
# 可选: --smooth 5 --output runs/keypoint_refinement/analysis
```




```bash
# 1. 退出当前虚拟环境并删除旧的 .venv
deactivate
rm -rf .venv

# 2. 使用 Python 3.10 创建新的虚拟环境
uv venv --python 3.10 .venv

# 3. 激活虚拟环境
source .venv/bin/activate

# 4. 安装兼容且极其稳定的依赖组合
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
uv pip install ultralytics "numpy<2.0" opencv-python-headless==4.10.0.84 pyyaml
```