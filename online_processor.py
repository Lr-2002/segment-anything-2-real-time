import os
from tqdm import tqdm
import torch
import numpy as np
import cv2
import imageio

torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

from sam2.build_sam import build_sam2_camera_predictor
import time

class OnlineProcessor:
    def __init__(self, model_cfg, sam2_checkpoint, bboxes=None, obj_ids=None, init_frame=None):
        self.model_cfg = model_cfg
        self.sam2_checkpoint = sam2_checkpoint
        self.predictor = self.build_model()
        self.if_init = False
        self.reset(init_frame)
        self.ann_frame_idx = 0
        self.obj_ids = obj_ids if obj_ids else []
        self.init_state(bboxes)

    def build_model(self):
        return build_sam2_camera_predictor(self.model_cfg, self.sam2_checkpoint)

    def reset(self, frame):
        try:
            self.predictor.reset_state()
        except: 
            pass
        if isinstance(frame, str):
            frame = cv2.imread(frame)
            if frame is None:
                raise ValueError("Failed to load image from the given path.")
        if isinstance(frame, np.ndarray):
            self.predictor.load_first_frame(frame)
            self.if_init = True
        else:
            raise ValueError("Frame must be either a path to an image or a numpy image array.")
    def init_state(self, bboxes):
        if bboxes is None:
            raise ValueError("bboxes cannot be None.")
        self.obj_ids = self.obj_ids if self.obj_ids else list(range(len(bboxes)))
        for bbox, obj_id in zip(bboxes, self.obj_ids):
            self.add_new_prompt(bbox=bbox, obj_id=obj_id)
    def add_new_prompt(self, points=None, labels=None, bbox=None, obj_id=None):
        if not self.if_init:
            raise ValueError("Model is not initialized. Please initialize with a frame first.")
        _, out_obj_ids, out_mask_logits = self.predictor.add_new_prompt(
            frame_idx=self.ann_frame_idx, obj_id=obj_id, points=points, labels=labels, bbox=bbox
        )
        return out_obj_ids, out_mask_logits



    def batch_add_bbox(self, bbox_list=None, obj_id_list=None):
        if not self.if_init:
            raise ValueError("Model is not initialized. Please initialize with a frame first.")
        for bbox, obj_id in zip(bbox_list , obj_id_list ):
            self.add_new_prompt(
                 obj_id=obj_id, bbox=bbox
            )



    def add_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        width, height = frame.shape[:2][::-1]
        if not self.if_init:
            self.predictor.load_first_frame(frame)
            self.if_init = True
        else:
            out_obj_ids, out_mask_logits = self.predictor.track(frame)
            all_mask = np.zeros((height, width, 1), dtype=np.uint8)
            masks = [] 
            for i in range(len(out_obj_ids)):
                out_mask = (out_mask_logits[i] > 0.0).permute(1,2,0).cpu().numpy().astype(np.uint8) * 255
                masks.append(out_mask)
        masks = np.stack(masks, axis=0)
        return masks

    def process_video(self, video_path, output_folder):
        cap = cv2.VideoCapture(video_path)
        cnt = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            processed_frame = self.add_frame(frame)
            cv2.imwrite(os.path.join(output_folder, str(cnt) + '.png'), processed_frame)
            cnt += 1
        cap.release()

    def process_image_dirs(self, image_dir, output_folder='./tmp'):
        image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        image_files.sort()
        for cnt, image_path in tqdm(enumerate(image_files)):
            frame = cv2.imread(image_path)
            if frame is None:
                continue
            masks = self.add_frame(frame)
            for idx, mask in enumerate(masks):
                cv2.imwrite(os.path.join(output_folder, str(cnt) + '_' + str(idx) + '.png'), mask)
if __name__=='__main__':

    # Example usage:
    processor = OnlineProcessor(model_cfg="sam2_hiera_s.yaml", sam2_checkpoint="checkpoints/sam2_hiera_small.pt", bboxes=[np.array([[600, 214], [765, 286]], dtype=np.float32)], obj_ids=[2], init_frame='/ssd/lt/processed_dataset/lt_sim_seged/val/video_EiQKGXdvcmtlcl8wMDFfZXBfMTRfMDZfMDZfMjIQIxguIAMw6AkqIGY1MjU4NTI4N2ExNTc2Yjg1ZGZiYzI5OWI0OTgxMWZj/images/00000.jpg')
    bbox_dict = np.load('/ssd/lt/processed_dataset/lt_sim_seged/val/video_EiQKGXdvcmtlcl8wMDFfZXBfMTRfMDZfMDZfMjIQIxguIAMw6AkqIGY1MjU4NTI4N2ExNTc2Yjg1ZGZiYzI5OWI0OTgxMWZj/bbox/00000.npy', allow_pickle=True).item()
    del bbox_dict['image_size']
    bboxes = list(bbox_dict.values())
    ids = list(bbox_dict.keys())
    processor.batch_add_bbox(bbox_list=bboxes, obj_id_list=ids)
    processor.process_image_dirs('/ssd/lt/processed_dataset/lt_sim_seged/val/video_EiQKGXdvcmtlcl8wMDFfZXBfMTRfMDZfMDZfMjIQIxguIAMw6AkqIGY1MjU4NTI4N2ExNTc2Yjg1ZGZiYzI5OWI0OTgxMWZj/images/')
    # processor.process_video("../notebooks/videos/aquarium/aquarium.mp4", "./output_frames")
