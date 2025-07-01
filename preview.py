import cv2
import sys
import torch
import numpy as np
from torchvision import transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from hybrik.models import builder
from hybrik.utils.config import update_config
from hybrik.utils.presets import SimpleTransform3DSMPLCam
from hybrik.utils.vis import get_max_iou_box, get_one_box, vis_2d


def list_cameras(max_cameras=10):
    available = []
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(i)
            cap.release()
    return available


def main():
    cv2.destroyAllWindows()
    print("Searching for available cameras...")
    cameras = list_cameras()
    if not cameras:
        print("No cameras found.")
        sys.exit(1)
    print("Available cameras:")
    for idx, cam_id in enumerate(cameras):
        print(f"{idx}: Camera #{cam_id}")
    while True:
        try:
            choice = int(input(f"Choose camera (0-{len(cameras)-1}): "))
            if 0 <= choice < len(cameras):
                cam_id = cameras[choice]
                break
            else:
                print("Invalid choice.")
        except ValueError:
            print("Enter camera number.")

    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        print("Cannot open selected camera.")
        sys.exit(1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg_file = 'configs/256x192_adam_lr1e-3-hrw48_cam_2x_w_pw3d_3dhp.yaml'
    CKPT = './pretrained_models/hybrik_hrnet.pth'
    cfg = update_config(cfg_file)
    bbox_3d_shape = getattr(cfg.MODEL, 'BBOX_3D_SHAPE', (2000, 2000, 2000))
    bbox_3d_shape = [item * 1e-3 for item in bbox_3d_shape]
    from easydict import EasyDict as edict
    dummpy_set = edict({
        'joint_pairs_17': None,
        'joint_pairs_24': None,
        'joint_pairs_29': None,
        'bbox_3d_shape': bbox_3d_shape
    })
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
    det_transform = T.Compose([T.ToTensor()])
    det_model = fasterrcnn_resnet50_fpn(pretrained=True)
    hybrik_model = builder.build_sppe(cfg.MODEL)
    print(f'Loading model from {CKPT}...')
    save_dict = torch.load(CKPT, map_location='cpu')
    if type(save_dict) == dict:
        model_dict = save_dict['model']
        hybrik_model.load_state_dict(model_dict)
    else:
        hybrik_model.load_state_dict(save_dict)
    det_model = det_model.to(device)
    hybrik_model = hybrik_model.to(device)
    det_model.eval()
    hybrik_model.eval()

    print("Press 'q' to quit.")
    import time
    import pickle
    import os
    prev_box = None
    frame_count = 0
    DETECT_EVERY = 5
    TARGET_FPS = 30
    frame_time = 1.0 / TARGET_FPS
    last_time = time.time()
    recording = False

    res_keys = [
        'pred_uvd', 'pred_xyz_17', 'pred_xyz_29', 'pred_xyz_24_struct',
        'pred_scores', 'pred_camera', 'pred_betas', 'pred_thetas', 'pred_phi',
        'pred_cam_root', 'transl', 'transl_camsys', 'bbox', 'height', 'width', 'img_path'
    ]
    res_db = {k: [] for k in res_keys}

    record_start_time = None
    record_end_time = None
    output_dir = os.path.join('output', 'recording')
    os.makedirs(output_dir, exist_ok=True)
    video_writer = None
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_path = os.path.join(output_dir, 'res.mp4')

    with torch.no_grad():
        while True:
            start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                print("Frame read error.")
                break
            input_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if frame_count % DETECT_EVERY == 0 or prev_box is None:
                det_input = det_transform(input_image).to(device)
                det_output = det_model([det_input])[0]
                tight_bbox = get_one_box(
                    det_output) if prev_box is None else get_max_iou_box(det_output, prev_box)
                if tight_bbox is None:
                    prev_box = None
                else:
                    prev_box = tight_bbox
            else:
                tight_bbox = prev_box

            data_predicted = False
            if tight_bbox is not None:
                pose_input, bbox, img_center = transformation.test_transform(
                    input_image, tight_bbox)
                pose_input = pose_input.to(device)[None, :, :, :]
                pose_output = hybrik_model(
                    pose_input, flip_test=True,
                    bboxes=torch.from_numpy(np.array(bbox)).to(
                        device).unsqueeze(0).float(),
                    img_center=torch.from_numpy(img_center).to(
                        device).unsqueeze(0).float()
                )
                uv_29 = pose_output.pred_uvd_jts.reshape(
                    29, 3)[:, :2].cpu().numpy()
                bbox_xywh = [
                    (bbox[0] + bbox[2]) / 2,
                    (bbox[1] + bbox[3]) / 2,
                    bbox[2] - bbox[0],
                    bbox[3] - bbox[1]
                ]
                pts = uv_29 * bbox_xywh[2]
                pts[:, 0] = pts[:, 0] + bbox_xywh[0]
                pts[:, 1] = pts[:, 1] + bbox_xywh[1]
                data_predicted = True
                image = input_image.copy()
                bbox_img = vis_2d(image, tight_bbox, pts)
                bbox_img = cv2.cvtColor(bbox_img, cv2.COLOR_RGB2BGR)
            else:
                bbox_img = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
                pts = None

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                if not recording:
                    recording = True
                    record_start_time = time.time()
                    res_db = {k: [] for k in res_keys}
                    print('Recording started!')
                    h, w = bbox_img.shape[:2]
                    video_writer = cv2.VideoWriter(
                        video_path, fourcc, TARGET_FPS, (w, h))
            elif key == ord('e'):
                if recording:
                    record_end_time = time.time()
                    print('Recording finished! Saving results...')
                    recording = False
                    if len(res_db['pred_xyz_29']) > 0:
                        try:
                            n_frames = len(res_db['pred_xyz_29'])
                            for k in res_db.keys():
                                if len(res_db[k]) > 0:
                                    res_db[k] = np.stack(res_db[k])
                                    assert res_db[k].shape[0] == n_frames, f"Key {k} has wrong shape"
                                else:
                                    if k == 'pred_uvd':
                                        res_db[k] = np.zeros((n_frames, 29, 3))
                                    elif k == 'pred_xyz_17':
                                        res_db[k] = np.zeros((n_frames, 17, 3))
                                    elif k == 'pred_xyz_29':
                                        res_db[k] = np.zeros((n_frames, 29, 3))
                                    elif k == 'pred_xyz_24_struct':
                                        res_db[k] = np.zeros((n_frames, 24, 3))
                                    elif k == 'pred_scores':
                                        res_db[k] = np.zeros((n_frames, 29))
                                    elif k == 'pred_camera':
                                        res_db[k] = np.zeros((n_frames, 3))
                                    elif k == 'pred_betas':
                                        res_db[k] = np.zeros((n_frames, 10))
                                    elif k == 'pred_thetas':
                                        res_db[k] = np.zeros(
                                            (n_frames, 24, 3, 3))
                                    elif k == 'pred_phi':
                                        res_db[k] = np.zeros((n_frames, 23, 2))
                                    elif k == 'pred_cam_root':
                                        res_db[k] = np.zeros((n_frames, 3))
                                    elif k == 'transl':
                                        res_db[k] = np.zeros((n_frames, 3))
                                    elif k == 'transl_camsys':
                                        res_db[k] = np.zeros((n_frames, 3))
                                    elif k == 'bbox':
                                        res_db[k] = np.zeros((n_frames, 4))
                                    elif k == 'height':
                                        res_db[k] = np.zeros((n_frames,))
                                    elif k == 'width':
                                        res_db[k] = np.zeros((n_frames,))
                                    elif k == 'img_path':
                                        res_db[k] = np.array(
                                            [f'frame_{i:06d}' for i in range(n_frames)])

                            with open(os.path.join(output_dir, 'res.pk'), 'wb') as f:
                                pickle.dump(res_db, f)
                            print(
                                f'Saved {n_frames} frames to res.pk (format like demo_video_stabilized.py)')
                        except Exception as e:
                            print(f'Error saving res.pk: {e}')
                    else:
                        print('No data to save! (No people detected?)')
                    if video_writer is not None:
                        video_writer.release()
                        video_writer = None
                    if record_start_time and record_end_time and len(res_db['pred_xyz_29']) > 1:
                        duration = record_end_time - record_start_time
                        real_fps = len(res_db['pred_xyz_29']) / \
                            duration if duration > 0 else TARGET_FPS
                        print(
                            f'Recording: {len(res_db["pred_xyz_29"])} frames, time: {duration:.2f}s, FPS: {real_fps:.2f}')
                        if abs(real_fps - TARGET_FPS) > 2:
                            try:
                                import shutil
                                temp_path = os.path.join(
                                    output_dir, 'temp_res.mp4')
                                cap2 = cv2.VideoCapture(video_path)
                                h = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                w = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
                                vw2 = cv2.VideoWriter(
                                    temp_path, fourcc, real_fps, (w, h))
                                while True:
                                    ret2, f2 = cap2.read()
                                    if not ret2:
                                        break
                                    vw2.write(f2)
                                cap2.release()
                                vw2.release()
                                shutil.move(temp_path, video_path)
                                print('res.mp4 re-encoded to real FPS!')
                            except Exception as e:
                                print(f'Error re-encoding mp4: {e}')
                    print(f'Results saved in {output_dir}')

            if recording:
                if data_predicted:
                    pred_xyz_jts_17 = pose_output.pred_xyz_jts_17.reshape(
                        17, 3).cpu().data.numpy()
                    pred_uvd_jts = pose_output.pred_uvd_jts.reshape(
                        -1, 3).cpu().data.numpy()
                    pred_xyz_jts_29 = pose_output.pred_xyz_jts_29.reshape(
                        29, 3).cpu().data.numpy()
                    pred_xyz_jts_24_struct = pose_output.pred_xyz_jts_24_struct.reshape(
                        24, 3).cpu().data.numpy()
                    pred_scores = pose_output.maxvals.cpu(
                    ).data[:, :29].reshape(29).numpy()
                    pred_camera = pose_output.pred_camera.squeeze(
                        dim=0).cpu().data.numpy()
                    pred_betas = pose_output.pred_shape.squeeze(
                        dim=0).cpu().data.numpy()
                    pred_theta = pose_output.pred_theta_mats.squeeze(
                        dim=0).cpu().data.numpy()
                    pred_phi = pose_output.pred_phi.squeeze(
                        dim=0).cpu().data.numpy()
                    pred_cam_root = pose_output.cam_root.squeeze(
                        dim=0).cpu().numpy()

                    transl = pose_output.transl.detach()[0].cpu().data.numpy()
                    focal = 1000.0
                    bbox_xywh_calc = [
                        (bbox[0] + bbox[2]) / 2,
                        (bbox[1] + bbox[3]) / 2,
                        bbox[2] - bbox[0],
                        bbox[3] - bbox[1]
                    ]
                    transl_camsys = transl * 256 / bbox_xywh_calc[2]

                    img_size = np.array(
                        (input_image.shape[0], input_image.shape[1]))

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
                    res_db['transl'].append(transl)
                    res_db['transl_camsys'].append(transl_camsys)
                    res_db['bbox'].append(np.array(bbox))
                    res_db['height'].append(img_size[0])
                    res_db['width'].append(img_size[1])
                    res_db['img_path'].append(
                        f'camera_frame_{frame_count:06d}')

                    if video_writer is not None:
                        video_writer.write(bbox_img)
                else:
                    img_size = np.array(
                        (input_image.shape[0], input_image.shape[1]))

                    res_db['pred_xyz_17'].append(np.zeros((17, 3)))
                    res_db['pred_uvd'].append(np.zeros((29, 3)))
                    res_db['pred_xyz_29'].append(np.zeros((29, 3)))
                    res_db['pred_xyz_24_struct'].append(np.zeros((24, 3)))
                    res_db['pred_scores'].append(np.zeros(29))
                    res_db['pred_camera'].append(np.zeros(3))
                    res_db['pred_betas'].append(np.zeros(10))
                    res_db['pred_thetas'].append(np.zeros((24, 3, 3)))
                    res_db['pred_phi'].append(np.zeros((23, 2)))
                    res_db['pred_cam_root'].append(np.zeros(3))
                    res_db['transl'].append(np.zeros(3))
                    res_db['transl_camsys'].append(np.zeros(3))
                    res_db['bbox'].append(np.zeros(4))
                    res_db['height'].append(img_size[0])
                    res_db['width'].append(img_size[1])
                    res_db['img_path'].append(
                        f'camera_frame_{frame_count:06d}_no_detection')

                    if video_writer is not None:
                        video_writer.write(bbox_img)

            cv2.imshow('HybrIK Preview', bbox_img)
            elapsed = time.time() - start_time
            if elapsed < frame_time:
                time.sleep(frame_time - elapsed)
            frame_count += 1

    cap.release()
    if video_writer is not None:
        video_writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
