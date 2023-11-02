import argparse, os, sys, glob
import imageio
import time
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
from os.path import join
from ..main import instantiate_from_config
import torch.nn as nn
import cv2
from ..ldm.models.diffusion.ddim import DDIMSampler

class DiffusionInpainter():
    def __init__(self, model_dir, device=None, **kwargs):
        self.config = OmegaConf.load(join(model_dir,"models/ldm/inpainting_big/config.yaml"))
        self.config.model.params.unet_config.params.dcnn_mode = kwargs.get('dcnn_mode')
        self.config.model.params.unet_config.params.dcnn_input_level = kwargs.get('dcnn_input_level')
        self.config.model.params.unet_config.params.dcnn_output_level = kwargs.get('dcnn_output_level')
        self.model = instantiate_from_config(self.config.model)
        self.model.load_state_dict(torch.load(join(model_dir,"models/ldm/inpainting_big/last.ckpt"))["state_dict"],
                            strict=False)
        self.model = self.model.to(device)
        self.sampler = DDIMSampler(self.model)
        self.device = device
        self.depth_rate = kwargs.get('depth_rate',10)

    @torch.no_grad()
    def forward_3P(self,mask,context,rgb,edge,steps=50,unit_length=128,**kwargs):
        depth = kwargs.get('depth')
        if not isinstance(depth,torch.Tensor):
            depth = torch.from_numpy(depth)
            depth = depth.float().unsqueeze(0).unsqueeze(0).to(self.device)
        depth = depth/self.depth_rate

        n,c,h,w = rgb.size()
        residual_h = int(np.ceil(h / float(unit_length)) * unit_length - h)
        residual_w = int(np.ceil(w / float(unit_length)) * unit_length - w)
        anchor_h = residual_h//2
        anchor_w = residual_w//2
        enlarge_img = torch.full((n, c, h + residual_h, w + residual_w),0.).to(self.device)
        enlarge_mask = torch.ones((n, 1, h + residual_h, w + residual_w)).to(self.device)
        enlarge_depth = torch.full((n, 1, h + residual_h, w + residual_w),0.).to(self.device)
        enlarge_img[..., anchor_h:anchor_h+h, anchor_w:anchor_w+w] = rgb
        enlarge_depth[..., anchor_h:anchor_h+h, anchor_w:anchor_w+w] = depth
        enlarge_mask[..., anchor_h:anchor_h+h, anchor_w:anchor_w+w] = 1-context

        enlarge_mask = enlarge_mask*2.0-1.0
        enlarge_img = enlarge_img*2.0-1.0
        with torch.no_grad():
            with self.model.ema_scope():
                # encode masked image and concat downsampled mask
                c = self.model.cond_stage_model.encode(enlarge_img)
                cc = torch.nn.functional.interpolate(enlarge_mask,size=c.shape[-2:])
                cond = torch.cat((c, cc), dim=1)

                shape = (cond.shape[1]-1,)+cond.shape[2:]
                samples_ddim, _ = self.sampler.sample(S=steps,
                                                    conditioning=cond,
                                                    batch_size=cond.shape[0],
                                                    shape=shape,
                                                    verbose=False,
                                                    depth=enlarge_depth)
                x_samples_ddim = self.model.decode_first_stage(samples_ddim)

        predicted_image_raw = torch.clamp((x_samples_ddim+1.0)/2.0,min=0.0, max=1.0)
        predicted_image = predicted_image_raw[..., anchor_h:anchor_h+h, anchor_w:anchor_w+w]
        
        return predicted_image

    @torch.no_grad()
    def forward_inpaint(self, mask, context, image, position, steps=200, unit_length=128, **kwargs):
        image = image.to(self.device)
        if image.max()>1:
            image = image/255

        if isinstance(position,list):
            position = {
                'x_min':position[0],
                'x_max':position[1],
                'y_min':position[2],
                'y_max':position[3]
            }
        
        depth = kwargs.get('depth')
        if kwargs.get('global_only'):
            n,c,h,w = image.size()
            global_mask = torch.zeros((n,1,h,w)).float().to(self.device)
            global_mask[...,position['x_min']:position['x_max'], position['y_min']:position['y_max']] = mask
            global_mask[depth==0] = 1
            global_context = 1-global_mask

            global_pred = self.forward_3P(global_mask, global_context, image, None, steps, unit_length, **kwargs)
            global_pred_patch = global_pred[...,position['x_min']:position['x_max'], position['y_min']:position['y_max']]

            pred_image = global_pred_patch
        else:
            context = context.to(self.device)
            mask = mask.to(self.device)

            np_mask = mask[0].clone().detach().cpu().permute(1,2,0).numpy().astype(np.uint8)
            mask_distance = cv2.distanceTransform(src=np_mask, distanceType=cv2.DIST_L2, maskSize=5)
            mask_weight = mask_distance/mask_distance.max()
            mask_weight = np.clip(mask_weight,0,1)
            mask_weight = torch.from_numpy(mask_weight).float().to(self.device).unsqueeze(0).unsqueeze(0)

            n,c,h,w = image.size()
            global_mask = torch.zeros((n,1,h,w)).float().to(self.device)
            global_mask[...,position['x_min']:position['x_max'], position['y_min']:position['y_max']] = mask

            depth_context = torch.full_like(context,depth.max()).float().to(self.device)
            depth_context = depth[...,position['x_min']:position['x_max'], position['y_min']:position['y_max']]
            bg_threshold = depth_context[context>0].min()
            
            masked_depth = depth*(1-global_mask)
            global_context = torch.ones((n,1,h,w)).float().to(self.device)
            global_context[...,position['x_min']:position['x_max'], position['y_min']:position['y_max']] = 0
            global_context[...,position['x_min']:position['x_max'], position['y_min']:position['y_max']] = context
            global_context[masked_depth > bg_threshold] = 1
            global_rgb = global_context * image
            global_mask = (1-global_context)

            global_pred = self.forward_3P(global_mask, global_context, global_rgb, None, steps, unit_length, **kwargs)
            global_pred_patch = global_pred[...,position['x_min']:position['x_max'], position['y_min']:position['y_max']]

            patch_rgb = image[...,position['x_min']:position['x_max'], position['y_min']:position['y_max']]*context
            patch_depth = kwargs.get('depth_patch')
            patch_pred = self.forward_3P(mask, context, patch_rgb, None, steps, unit_length, depth=patch_depth)
            
            pred_image = mask_weight*global_pred_patch + (1-mask_weight)*patch_pred
        
        return pred_image




