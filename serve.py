import argparse
import os
import glob
import torch
from torch.utils.data import DataLoader
from copy import deepcopy
import tempfile
from functools import partial
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import Optional

CACHE_EXAMPLES = os.environ.get("CACHE_EXAMPLES", "0") == "1"
DEFAULT_CAM_DIST = 1.9

from image_preprocess.utils import image_preprocess, resize_image, sam_out_nosave, pred_bbox, sam_init
from gradio_splatting.backend.gradio_model3dgs import Model3DGS
from tgs.data import CustomImageOrbitDataset
from tgs.utils.misc import todevice
from tgs.utils.config import ExperimentConfig, load_config
from infer import TGS

from huggingface_hub import hf_hub_download
MODEL_CKPT_PATH = hf_hub_download(repo_id="VAST-AI/TriplaneGaussian", local_dir="./checkpoints", filename="model_lvis_rel.ckpt", repo_type="model")
SAM_CKPT_PATH = "checkpoints/sam_vit_h_4b8939.pth"
CONFIG = "config.yaml"
EXP_ROOT_DIR = "./outputs"

os.makedirs(EXP_ROOT_DIR, exist_ok=True)

gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
device = "cuda:{}".format(gpu) if torch.cuda.is_available() else "cpu"

print("device: ", device)

# init model
base_cfg: ExperimentConfig
base_cfg = load_config(CONFIG, cli_args=[], n_gpus=1)
base_cfg.system.weights = MODEL_CKPT_PATH
model = TGS(cfg=base_cfg.system).to(device)
print("load model ckpt done.")

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    global sam_predictor
    sam_predictor = sam_init(SAM_CKPT_PATH, gpu)
    print("load sam ckpt done.")

def assert_input_image(input_image):
    if input_image is None:
        raise HTTPException(status_code=400, detail="No image selected or uploaded!")

def preprocess(input_raw, sam_predictor=None):
    save_path = model.get_save_path("seg_rgba.png")
    input_raw = resize_image(input_raw, 512)
    image_sam = sam_out_nosave(
        sam_predictor, input_raw.convert("RGB"), pred_bbox(input_raw)
    )
    image_preprocess(image_sam, save_path, lower_contrast=False, rescale=True)
    return save_path

def init_trial_dir():
    trial_dir = tempfile.TemporaryDirectory(dir=EXP_ROOT_DIR).name
    model.set_save_dir(trial_dir)
    return trial_dir

@torch.no_grad()
def infer(image_path: str,
          cam_dist: float,):
    data_cfg = deepcopy(base_cfg.data)
    data_cfg.only_3dgs = True
    data_cfg.cond_camera_distance = cam_dist
    data_cfg.eval_camera_distance = cam_dist
    data_cfg.image_list = [image_path]
    dataset = CustomImageOrbitDataset(data_cfg)
    dataloader = DataLoader(dataset,
                batch_size=data_cfg.eval_batch_size, 
                num_workers=data_cfg.num_workers,
                shuffle=False,
                collate_fn=dataset.collate
            )

    for batch in dataloader:
        batch = todevice(batch, device)
        model(batch)

@app.post("/run")
async def run(image: UploadFile = File(...)):
    assert_input_image(image)
    save_path = init_trial_dir()
    input_image = await image.read()
    seg_image_path = preprocess(input_image, sam_predictor)
    infer(seg_image_path, DEFAULT_CAM_DIST, only_3dgs=True)
    gs = glob.glob(os.path.join(save_path, "3dgs", "*.ply"))[0]
    return FileResponse(gs)

if __name__ == "__main__":
    import uvicorn
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8888)
    args = parser.parse_args()
    uvicorn.run(app, host="0.0.0.0", port=args.port)