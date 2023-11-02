import json
import re
import numpy as np
import argparse
import glob
import os
import yaml
import time
import sys
from data_tools import dataset
from mesh import write_ply, read_ply, output_3d_photo
import torch
import cv2
import imageio
import copy
from networks import Inpaint_Depth_Net, Inpaint_Edge_Net
from bilateral_filtering import sparse_bilateral_filtering
from os.path import *
from mesh import *

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='argument.yml',help='Configure of post processing')
parser.add_argument('--data_dir', type=str, default='')
parser.add_argument('--output_dir', type=str, default='')
parser.add_argument('--range',type=str, default='0,1000')
args = parser.parse_args()
args.range = [int(x) for x in args.range.split(',')]
config = yaml.safe_load(open(args.config, 'r'))
device = config["gpu_ids"]

model = None
torch.cuda.empty_cache()
print(f"Loading edge model at {time.time()}")
depth_edge_model = Inpaint_Edge_Net(init_weights=True)
depth_edge_weight = torch.load(config['depth_edge_model_ckpt'],
                                map_location=torch.device(device))
depth_edge_model.load_state_dict(depth_edge_weight)
depth_edge_model = depth_edge_model.to(device)
depth_edge_model.eval()

print(f"Loading depth model at {time.time()}")
depth_feat_model = Inpaint_Depth_Net()
depth_feat_weight = torch.load(config['depth_feat_model_ckpt'],
                                map_location=torch.device(device))
depth_feat_model.load_state_dict(depth_feat_weight, strict=True)
depth_feat_model = depth_feat_model.to(device)
depth_feat_model.eval()
depth_feat_model = depth_feat_model.to(device)
print(f"Loading rgb model at {time.time()}")
from BilateralDiffusion.scripts.inpaint_wrapper import DiffusionInpainter
rgb_model = DiffusionInpainter('LatentDiffusion',device=device,dcnn_mode=config.get('dcnn_mode'))

videos = dataset(args.data_dir,hw=(256,384))

for cnt, data in enumerate(videos):
    # data = videos[50]
    if not (cnt >= args.range[0] and cnt < args.range[1]):
        continue
    
    vname = data['scene']
    print(f'PROCESSING: {vname}')
    outdir = join(args.output_dir, vname)
    mesh_fi = os.path.join(args.output_dir, vname, 'src_mesh.ply')

    os.makedirs(outdir,exist_ok=True)
    image = data['src_img']
    H,W = image.shape[:2]
    camera_mtx = data['K']
    
    config['output_h'], config['output_w'] = data['src_img'].shape[:2]
    config['original_h'], config['original_w'] = config['output_h'], config['output_w']
    if image.ndim == 2:
        image = image[..., None].repeat(3, -1)
    if np.sum(np.abs(image[..., 0] - image[..., 1])) == 0 and np.sum(np.abs(image[..., 1] - image[..., 2])) == 0:
        config['gray_image'] = True
    else:
        config['gray_image'] = False

    depth = data['src_depth'] / 120
    disp = data['src_disp'].astype(np.float32)
    disp = disp - disp.min()
    disp = cv2.blur(disp / disp.max(), ksize=(3, 3)) * disp.max()
    disp = (disp / disp.max()) * 3.0
    depth = 1. / np.maximum(disp, 0.05)
    mean_loc_depth = depth[depth.shape[0]//2, depth.shape[1]//2]

    if not(config['load_ply'] and os.path.exists(mesh_fi)):
        vis_photos, vis_depths = sparse_bilateral_filtering(depth.copy(), image.copy(), config, num_iter=config['sparse_iter'], spdb=False)
        depth = vis_depths[-1]
        torch.cuda.empty_cache()
        print(f"GENERATING 3D SCENE...")
        t0 = time.time()
        rt_info = write_ply(  image,
                              depth,
                              camera_mtx,
                              mesh_fi,
                              config,
                              rgb_model,
                              depth_edge_model,
                              depth_edge_model,
                              depth_feat_model)
        print(f'SPEED {time.time()-t0} sec.')
    
    if config['save_ply'] is True or config['load_ply'] is True:
        print(f"LOADING 3D SCENE...")
        verts, colors, faces, Height, Width, hFov, vFov = read_ply(mesh_fi)
    else:
        verts, colors, faces, Height, Width, hFov, vFov = rt_info

    cam_mesh = netx.Graph()
    cam_mesh.graph['H'] = Height
    cam_mesh.graph['W'] = Width
    cam_mesh.graph['original_H'] = config['original_h']
    cam_mesh.graph['original_W'] = config['original_w']
    int_mtx_real_x = camera_mtx[0] * Width
    int_mtx_real_y = camera_mtx[1] * Height
    cam_mesh.graph['hFov'] = 2 * np.arctan((1. / 2.) * ((cam_mesh.graph['original_W']) / int_mtx_real_x[0]))
    cam_mesh.graph['vFov'] = 2 * np.arctan((1. / 2.) * ((cam_mesh.graph['original_H']) / int_mtx_real_y[1]))
    colors = colors[..., :3]

    fov_in_rad = max(cam_mesh.graph['vFov'], cam_mesh.graph['hFov'])
    fov = (fov_in_rad * 180 / np.pi)
    init_factor = 1
    if config.get('anti_flickering') is True:
        init_factor = 3
    if (cam_mesh.graph['original_H'] is not None) and (cam_mesh.graph['original_W'] is not None):
        canvas_w = cam_mesh.graph['original_W']
        canvas_h = cam_mesh.graph['original_H']
    else:
        canvas_w = cam_mesh.graph['W']
        canvas_h = cam_mesh.graph['H']
    canvas_size = max(canvas_h, canvas_w)
    normal_canvas = Canvas_view(fov,
                                verts,
                                faces,
                                colors,
                                canvas_size=canvas_size,
                                factor=init_factor,
                                bgcolor='gray',
                                proj='perspective')
    img = normal_canvas.render()
    backup_img, backup_all_img, all_img_wo_bound = img.copy(), img.copy() * 0, img.copy() * 0
    img = cv2.resize(img, (int(img.shape[1] / init_factor), int(img.shape[0] / init_factor)), interpolation=cv2.INTER_AREA)
    border = [0, img.shape[0], 0, img.shape[1]]
    H, W = cam_mesh.graph['H'], cam_mesh.graph['W']
    if (cam_mesh.graph['original_H'] is not None) and (cam_mesh.graph['original_W'] is not None):
        aspect_ratio = cam_mesh.graph['original_H'] / cam_mesh.graph['original_W']
    else:
        aspect_ratio = cam_mesh.graph['H'] / cam_mesh.graph['W']
    if aspect_ratio > 1:
        img_h_len = cam_mesh.graph['H'] if cam_mesh.graph.get('original_H') is None else cam_mesh.graph['original_H']
        img_w_len = img_h_len / aspect_ratio
        anchor = [0,
                  img.shape[0],
                  int(max(0, int((img.shape[1])//2 - img_w_len//2))),
                  int(min(int((img.shape[1])//2 + img_w_len//2), (img.shape[1])-1))]
    elif aspect_ratio <= 1:
        img_w_len = cam_mesh.graph['W'] if cam_mesh.graph.get('original_W') is None else cam_mesh.graph['original_W']
        img_h_len = img_w_len * aspect_ratio
        anchor = [int(max(0, int((img.shape[0])//2 - img_h_len//2))),
                  int(min(int((img.shape[0])//2 + img_h_len//2), (img.shape[0])-1)),
                  0,
                  img.shape[1]]
    anchor = np.array(anchor)
    plane_width = np.tan(fov_in_rad/2.) * np.abs(mean_loc_depth)

    imageio.imwrite(join(outdir,f'src.png'),data['src_img'])

    for tgt_id, tgt_info in enumerate(data['tgt']):
        torch.cuda.empty_cache()
        print(f'\rRENDER POSE: {tgt_id}.',end='')
        trans_mtx = tgt_info['trans_mtx']
        axis, angle = transforms3d.axangles.mat2axangle(trans_mtx[0:3, 0:3])
        trans_mtx[2, 3] = trans_mtx[2, 3]/2
        normal_canvas.rotate(axis=axis, angle=(angle*180)/np.pi)
        normal_canvas.translate(trans_mtx[:3,3])
        
        new_mean_loc_depth = mean_loc_depth - float(trans_mtx[2, 3])
        new_fov = float((np.arctan2(plane_width, np.array([np.abs(new_mean_loc_depth)])) * 180. / np.pi) * 2)
        normal_canvas.reinit_camera(new_fov)

        normal_canvas.view_changed()
        img = normal_canvas.render()
        img = cv2.GaussianBlur(img,(int(init_factor//2 * 2 + 1), int(init_factor//2 * 2 + 1)), 0)
        img = cv2.resize(img, (int(img.shape[1] / init_factor), int(img.shape[0] / init_factor)), interpolation=cv2.INTER_AREA)
        img = img[anchor[0]:anchor[1], anchor[2]:anchor[3]]
        img = img[int(border[0]):int(border[1]), int(border[2]):int(border[3])]

        normal_canvas.translate(-trans_mtx[:3,3])
        normal_canvas.rotate(axis=axis, angle=-(angle*180)/np.pi)
        normal_canvas.view_changed()

        imageio.imwrite(join(outdir,f'tgt_{tgt_id+1:03d}.png'), img)
    print('')