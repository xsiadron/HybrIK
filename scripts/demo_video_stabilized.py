"""Image demo script with temporal stabilization for smooth animations."""
import argparse
import os
import pickle as pk
from collections import deque

import cv2
import numpy as np
import torch
from easydict import EasyDict as edict
from torchvision import transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d

from hybrik.models import builder
from hybrik.utils.config import update_config
from hybrik.utils.presets import SimpleTransform3DSMPLCam
from hybrik.utils.vis import get_max_iou_box, get_one_box, vis_2d

det_transform = T.Compose([T.ToTensor()])


class TemporalStabilizer:
    def __init__(self, window_size=7, alpha=0.15):
        """
        Enhanced temporal stabilizer for ultra-smooth pose estimation
        
        Args:
            window_size (int): Size of the temporal window for smoothing
            alpha (float): Exponential moving average factor (0.0 - 1.0) - lower = smoother
        """
        self.window_size = window_size
        self.alpha = alpha
        self.bbox_history = deque(maxlen=window_size)
        self.pose_history = deque(maxlen=window_size)
        self.translation_history = deque(maxlen=window_size)
        self.prev_bbox = None
        self.prev_pose = None
        self.prev_translation = None
        self.frame_count = 0
        
    def smooth_bbox(self, current_bbox):
        """Apply enhanced temporal smoothing to bounding box"""
        if self.prev_bbox is None:
            self.prev_bbox = current_bbox
            return current_bbox
            
        # Multi-stage smoothing for ultra-stability
        # Stage 1: Exponential moving average
        stage1_bbox = []
        for i in range(4):
            smoothed_val = self.alpha * current_bbox[i] + (1 - self.alpha) * self.prev_bbox[i]
            stage1_bbox.append(smoothed_val)
        
        # Stage 2: Add to history and apply median filter for outlier removal
        self.bbox_history.append(stage1_bbox)
        if len(self.bbox_history) >= 3:
            # Apply median filter to remove outliers
            history_array = np.array(list(self.bbox_history))
            median_bbox = np.median(history_array, axis=0).tolist()
            # Blend median with current
            final_bbox = []
            for i in range(4):
                final_val = 0.7 * stage1_bbox[i] + 0.3 * median_bbox[i]
                final_bbox.append(final_val)
        else:
            final_bbox = stage1_bbox
            
        self.prev_bbox = final_bbox
        return final_bbox
        
    def smooth_pose(self, current_pose):
        """Apply enhanced temporal smoothing to 3D pose"""
        if self.prev_pose is None:
            self.prev_pose = current_pose.copy()
            return current_pose
            
        # Multi-stage pose smoothing
        # Stage 1: Strong exponential moving average
        stage1_pose = self.alpha * current_pose + (1 - self.alpha) * self.prev_pose
        
        # Stage 2: Add to history and apply temporal consistency
        self.pose_history.append(stage1_pose.copy())
        if len(self.pose_history) >= 3:
            # Apply median filter per joint to remove pose outliers
            history_array = np.array(list(self.pose_history))
            median_pose = np.median(history_array, axis=0)
            
            # Blend for ultra-smooth result
            final_pose = 0.6 * stage1_pose + 0.4 * median_pose
        else:
            final_pose = stage1_pose
            
        self.prev_pose = final_pose.copy()
        return final_pose
        
    def smooth_translation(self, current_translation):
        """Apply enhanced temporal smoothing to translation"""
        if self.prev_translation is None:
            self.prev_translation = current_translation.copy()
            return current_translation
            
        # Very strong smoothing for translation stability
        stage1_translation = self.alpha * current_translation + (1 - self.alpha) * self.prev_translation
        
        # Add history-based smoothing
        self.translation_history.append(stage1_translation.copy())
        if len(self.translation_history) >= 3:
            history_array = np.array(list(self.translation_history))
            median_translation = np.median(history_array, axis=0)
            final_translation = 0.7 * stage1_translation + 0.3 * median_translation
        else:
            final_translation = stage1_translation
            
        self.prev_translation = final_translation.copy()
        return final_translation


def apply_gaussian_smoothing(data_sequence, sigma=1.0):
    """Apply Gaussian smoothing to a sequence of data"""
    if len(data_sequence) < 3:
        return data_sequence
        
    # Convert to numpy array for processing
    data_array = np.array(data_sequence)
    
    # Apply Gaussian filter along the temporal axis
    if data_array.ndim == 3:  # For pose data (frames, joints, coords)
        smoothed = np.zeros_like(data_array)
        for joint_idx in range(data_array.shape[1]):
            for coord_idx in range(data_array.shape[2]):
                smoothed[:, joint_idx, coord_idx] = gaussian_filter1d(
                    data_array[:, joint_idx, coord_idx], sigma=sigma
                )
    elif data_array.ndim == 2:  # For bbox or translation data (frames, coords)
        smoothed = np.zeros_like(data_array)
        for coord_idx in range(data_array.shape[1]):
            smoothed[:, coord_idx] = gaussian_filter1d(
                data_array[:, coord_idx], sigma=sigma
            )
    else:
        smoothed = gaussian_filter1d(data_array, sigma=sigma)
        
    return smoothed.tolist()


def stabilized_bbox_tracking(det_output, prev_box, stabilizer, confidence_threshold=0.5):
    """Enhanced bbox tracking with temporal stabilization"""
    if prev_box is None:
        bbox = get_one_box(det_output)
        if bbox is None:
            return None
        return stabilizer.smooth_bbox(bbox)
    
    # Get bbox with IoU tracking
    bbox = get_max_iou_box(det_output, prev_box)
    
    # Check if detection confidence is too low
    max_score = 0
    for i in range(det_output['boxes'].shape[0]):
        score = det_output['scores'][i]
        if float(score) > max_score:
            max_score = float(score)
    
    # If confidence is low, use previous bbox with slight adaptation
    if max_score < confidence_threshold:
        bbox = prev_box
    
    # Apply temporal smoothing
    return stabilizer.smooth_bbox(bbox)


def xyxy2xywh(bbox):
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return [cx, cy, w, h]


def get_video_info(in_file):
    stream = cv2.VideoCapture(in_file)
    assert stream.isOpened(), 'Cannot capture source'
    datalen = int(stream.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = int(stream.get(cv2.CAP_PROP_FOURCC))
    fps = stream.get(cv2.CAP_PROP_FPS)
    frameSize = (int(stream.get(cv2.CAP_PROP_FRAME_WIDTH)),
                 int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    videoinfo = {'fourcc': fourcc, 'fps': fps, 'frameSize': frameSize}
    stream.release()
    return stream, videoinfo, datalen


def recognize_video_ext(ext=''):
    if ext == 'mp4':
        return cv2.VideoWriter_fourcc(*'mp4v'), '.' + ext
    elif ext == 'avi':
        return cv2.VideoWriter_fourcc(*'XVID'), '.' + ext
    elif ext == 'mov':
        return cv2.VideoWriter_fourcc(*'XVID'), '.' + ext
    else:
        print(f"Unknown video format {ext}, will use .mp4 instead of .{ext}")
        return cv2.VideoWriter_fourcc(*'mp4v'), '.mp4'


parser = argparse.ArgumentParser(description='HybrIK Demo with Temporal Stabilization')

parser.add_argument('--gpu', help='gpu', default=0, type=int)
parser.add_argument('--video-name', help='video name', default='', type=str)
parser.add_argument('--out-dir', help='output folder', default='', type=str)
parser.add_argument('--save-pk', default=False, dest='save_pk',
                    help='save prediction', action='store_true')
parser.add_argument('--save-img', default=False, dest='save_img',
                    help='save prediction', action='store_true')
parser.add_argument('--smoothing-alpha', default=0.1, type=float,
                    help='Exponential smoothing factor (0.0-1.0, lower = much smoother)')
parser.add_argument('--gaussian-sigma', default=2.5, type=float,
                    help='Gaussian smoothing sigma for post-processing')
parser.add_argument('--confidence-threshold', default=0.3, type=float,
                    help='Detection confidence threshold')

opt = parser.parse_args()

# Configuration
cfg_file = 'configs/256x192_adam_lr1e-3-hrw48_cam_2x_w_pw3d_3dhp.yaml'
CKPT = './pretrained_models/hybrik_hrnet.pth'
cfg = update_config(cfg_file)

bbox_3d_shape = getattr(cfg.MODEL, 'BBOX_3D_SHAPE', (2000, 2000, 2000))
bbox_3d_shape = [item * 1e-3 for item in bbox_3d_shape]
dummpy_set = edict({
    'joint_pairs_17': None,
    'joint_pairs_24': None,
    'joint_pairs_29': None,
    'bbox_3d_shape': bbox_3d_shape
})

res_keys = [
    'pred_uvd', 'pred_xyz_17', 'pred_xyz_29', 'pred_xyz_24_struct',
    'pred_scores', 'pred_camera', 'pred_betas', 'pred_thetas', 'pred_phi',
    'pred_cam_root', 'transl', 'transl_camsys', 'bbox', 'height', 'width', 'img_path'
]
res_db = {k: [] for k in res_keys}

# Initialize enhanced stabilizer for ultra-smooth results
stabilizer = TemporalStabilizer(window_size=7, alpha=opt.smoothing_alpha)

transformation = SimpleTransform3DSMPLCam(
    dummpy_set, scale_factor=cfg.DATASET.SCALE_FACTOR,
    color_factor=cfg.DATASET.COLOR_FACTOR,
    occlusion=cfg.DATASET.OCCLUSION,
    input_size=cfg.MODEL.IMAGE_SIZE,
    output_size=cfg.MODEL.HEATMAP_SIZE,
    depth_dim=cfg.MODEL.EXTRA.DEPTH_DIM,
    bbox_3d_shape=bbox_3d_shape,
    rot=cfg.DATASET.ROT_FACTOR, sigma=cfg.MODEL.EXTRA.SIGMA,
    train=False, add_dpg=False,
    loss_type=cfg.LOSS['TYPE'])

# Load models
det_model = fasterrcnn_resnet50_fpn(pretrained=True)
hybrik_model = builder.build_sppe(cfg.MODEL)

print(f'Loading model from {CKPT}...')
save_dict = torch.load(CKPT, map_location='cpu')
if type(save_dict) == dict:
    model_dict = save_dict['model']
    hybrik_model.load_state_dict(model_dict)
else:
    hybrik_model.load_state_dict(save_dict)

det_model.cuda(opt.gpu)
hybrik_model.cuda(opt.gpu)
det_model.eval()
hybrik_model.eval()

print('### Extract Image...')
video_basename = os.path.basename(opt.video_name).split('.')[0]

# Setup directories
if not os.path.exists(opt.out_dir):
    os.makedirs(opt.out_dir)
if not os.path.exists(os.path.join(opt.out_dir, 'raw_images')):
    os.makedirs(os.path.join(opt.out_dir, 'raw_images'))
if not os.path.exists(os.path.join(opt.out_dir, 'res_2d_images')) and opt.save_img:
    os.makedirs(os.path.join(opt.out_dir, 'res_2d_images'))

_, info, _ = get_video_info(opt.video_name)
video_basename = os.path.basename(opt.video_name).split('.')[0]

savepath2d = f'./{opt.out_dir}/res_2d_{video_basename}.mp4'
info['savepath2d'] = savepath2d

write2d_stream = cv2.VideoWriter(
    *[info[k] for k in ['savepath2d', 'fourcc', 'fps', 'frameSize']])
if not write2d_stream.isOpened():
    print("Try to use other video encoders...")
    ext = info['savepath2d'].split('.')[-1]
    fourcc, _ext = recognize_video_ext(ext)
    info['fourcc'] = fourcc
    info['savepath2d'] = info['savepath2d'][:-4] + _ext
    write2d_stream = cv2.VideoWriter(
        *[info[k] for k in ['savepath2d', 'fourcc', 'fps', 'frameSize']])

assert write2d_stream.isOpened(), 'Cannot open video for writing'

# Extract frames
os.system(f'ffmpeg -i {opt.video_name} {opt.out_dir}/raw_images/{video_basename}-%06d.png')

files = os.listdir(f'{opt.out_dir}/raw_images')
files.sort()

img_path_list = []
for file in tqdm(files):
    if not os.path.isdir(file) and file[-4:] in ['.jpg', '.png']:
        img_path = os.path.join(opt.out_dir, 'raw_images', file)
        img_path_list.append(img_path)

prev_box = None

print('### Run Model with Temporal Stabilization...')
print(f'Smoothing parameters: alpha={opt.smoothing_alpha}, confidence_threshold={opt.confidence_threshold}')

# Store raw results for post-processing
raw_poses = []
raw_translations = []
raw_bboxes = []

idx = 0
for img_path in tqdm(img_path_list):
    dirname = os.path.dirname(img_path)
    basename = os.path.basename(img_path)

    with torch.no_grad():
        # Run Detection
        input_image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        det_input = det_transform(input_image).to(opt.gpu)
        det_output = det_model([det_input])[0]

        # Stabilized bbox tracking
        tight_bbox = stabilized_bbox_tracking(
            det_output, prev_box, stabilizer, opt.confidence_threshold)
        
        if tight_bbox is None:
            continue
            
        prev_box = tight_bbox

        # Run HybrIK
        pose_input, bbox, img_center = transformation.test_transform(
            input_image, tight_bbox)
        pose_input = pose_input.to(opt.gpu)[None, :, :, :]
        pose_output = hybrik_model(
            pose_input, flip_test=True,
            bboxes=torch.from_numpy(np.array(bbox)).to(
                pose_input.device).unsqueeze(0).float(),
            img_center=torch.from_numpy(img_center).to(
                pose_input.device).unsqueeze(0).float()
        )
        
        uv_29 = pose_output.pred_uvd_jts.reshape(29, 3)[:, :2]
        transl = pose_output.transl.detach()

        # Apply temporal smoothing to pose and translation
        pose_3d = pose_output.pred_xyz_jts_29.reshape(-1, 3).cpu().data.numpy()
        transl_np = transl[0].cpu().data.numpy()
        
        smoothed_pose = stabilizer.smooth_pose(pose_3d)
        smoothed_transl = stabilizer.smooth_translation(transl_np)
        
        # Store for post-processing
        raw_poses.append(pose_3d)
        raw_translations.append(transl_np)
        raw_bboxes.append(bbox)

        # Visualization - only 2D
        focal = 1000.0
        bbox_xywh = xyxy2xywh(bbox)
        transl_camsys = torch.from_numpy(smoothed_transl).unsqueeze(0)
        transl_camsys = transl_camsys * 256 / bbox_xywh[2]

        focal = focal / 256 * bbox_xywh[2]

        # vis 2d with smoothed data
        pts = uv_29 * bbox_xywh[2]
        pts[:, 0] = pts[:, 0] + bbox_xywh[0]
        pts[:, 1] = pts[:, 1] + bbox_xywh[1]
        image = input_image.copy()
        bbox_img = vis_2d(image, tight_bbox, pts)
        bbox_img = cv2.cvtColor(bbox_img, cv2.COLOR_RGB2BGR)
        write2d_stream.write(bbox_img)

        if opt.save_img:
            idx += 1
            res_path = os.path.join(
                opt.out_dir, 'res_2d_images', f'image-{idx:06d}.jpg')
            cv2.imwrite(res_path, bbox_img)

        if opt.save_pk:
            assert pose_input.shape[0] == 1, 'Only support single batch inference for now'

            pred_xyz_jts_17 = pose_output.pred_xyz_jts_17.reshape(
                17, 3).cpu().data.numpy()
            pred_uvd_jts = pose_output.pred_uvd_jts.reshape(
                -1, 3).cpu().data.numpy()
            pred_xyz_jts_29 = smoothed_pose  # Use smoothed pose
            pred_xyz_jts_24_struct = pose_output.pred_xyz_jts_24_struct.reshape(
                24, 3).cpu().data.numpy()
            pred_scores = pose_output.maxvals.cpu().data[:, :29].reshape(29).numpy()
            pred_camera = pose_output.pred_camera.squeeze(dim=0).cpu().data.numpy()
            pred_betas = pose_output.pred_shape.squeeze(dim=0).cpu().data.numpy()
            pred_theta = pose_output.pred_theta_mats.squeeze(dim=0).cpu().data.numpy()
            pred_phi = pose_output.pred_phi.squeeze(dim=0).cpu().data.numpy()
            pred_cam_root = pose_output.cam_root.squeeze(dim=0).cpu().numpy()
            img_size = np.array((input_image.shape[0], input_image.shape[1]))

            res_db['pred_xyz_17'].append(pred_xyz_jts_17)
            res_db['pred_uvd'].append(pred_uvd_jts)
            res_db['pred_xyz_29'].append(pred_xyz_jts_29)
            res_db['pred_xyz_24_struct'].append(pred_xyz_jts_24_struct)
            res_db['pred_scores'].append(pred_scores)
            res_db['pred_camera'].append(pred_camera)
            res_db['pred_betas'].append(pred_betas)
            res_db['pred_thetas'].append(pred_theta)
            res_db['pred_phi'].append(pred_phi)
            res_db['pred_cam_root'].append(pred_cam_root)
            res_db['transl'].append(smoothed_transl)  # Use smoothed translation
            res_db['transl_camsys'].append(transl_camsys[0].cpu().data.numpy())
            res_db['bbox'].append(np.array(bbox))
            res_db['height'].append(img_size[0])
            res_db['width'].append(img_size[1])
            res_db['img_path'].append(img_path)

# Apply post-processing Gaussian smoothing if requested
if opt.save_pk and opt.gaussian_sigma > 0:
    print(f'### Applying post-processing Gaussian smoothing (sigma={opt.gaussian_sigma})...')
    
    # Apply Gaussian smoothing to key pose data
    if len(res_db['pred_xyz_29']) > 2:
        smoothed_poses = apply_gaussian_smoothing(res_db['pred_xyz_29'], opt.gaussian_sigma)
        res_db['pred_xyz_29'] = smoothed_poses
        
    if len(res_db['transl']) > 2:
        smoothed_transl = apply_gaussian_smoothing(res_db['transl'], opt.gaussian_sigma)
        res_db['transl'] = smoothed_transl

if opt.save_pk:
    n_frames = len(img_path_list)
    for k in res_db.keys():
        print(k)
        res_db[k] = np.stack(res_db[k])
        assert res_db[k].shape[0] == n_frames

    with open(os.path.join(opt.out_dir, 'res.pk'), 'wb') as fid:
        pk.dump(res_db, fid)

write2d_stream.release()

print(f"Processing complete! Results saved to {opt.out_dir}")
print(f"2D visualization video: {info['savepath2d']}")
if opt.save_pk:
    print(f"3D pose data saved to: {os.path.join(opt.out_dir, 'res.pk')}")
print(f"Stabilization applied with alpha={opt.smoothing_alpha}, gaussian_sigma={opt.gaussian_sigma}")
