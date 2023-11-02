# Diffuse3D: Wide-Angle 3D Photography via Bilateral Diffusion

**Diffuse3D: Wide-Angle 3D Photography via Bilateral Diffusion**            
Yutao Jiang, Yang Zhou, Yuan Liang, Wenxi Liu, Jianbo Jiao, Yuhui Quan, and Shengfeng He      
IEEE International Conference on Computer Vision (ICCV), 2023.      

**[[Paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Jiang_Diffuse3D_Wide-Angle_3D_Photography_via_Bilateral_Diffusion_ICCV_2023_paper.pdf)]   [[Demo](https://youtu.be/5mL6AMEvPSQ)]**   


## Setup
We recommend running our code with conda environment:
```bash
conda create -n d3d python=3.9
conda activate d3d
pip install -r requirements.txt
```

Next, please download the model weight using the following command:
```bash
bash download_model.sh
```

## Quick Start
The input to our model is a RGBD image, we recommend using the [DPT](https://github.com/isl-org/DPT) to generate depth image.      
The dataset directory should have the following structure:
```
dataset/
├── scene-1
│   ├── src.png
│   ├── src_depth.png
│   ├── ...
├── scene-2
│   ├── ...
├── ...
```

You can generate novel views with this command:
```bash
python inference.py --data_dir <DATASET_DIR> --output_dir <OUTPUT_DIR>
```

## License
This work is licensed under MIT License.        
If you find our work useful, please consider citing our paper:
```
@inproceedings{jiang2023diffuse3d,
  title={Diffuse3D: Wide-Angle 3D Photography via Bilateral Diffusion},
  author={Jiang, Yutao and Zhou, Yang and Liang, Yuan and Liu, Wenxi and Jiao, Jianbo and Quan, Yuhui and He, Shengfeng},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={8998--9008},
  year={2023}
}
```

## Acknowledgments
Our code is heavily borrowed from [3D Photo](https://github.com/vt-vl-lab/3d-photo-inpainting) and [LDM](https://github.com/CompVis/latent-diffusion), we thank the author for their great works.
