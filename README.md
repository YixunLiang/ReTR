# ReTR

> Code of paper 'ReTR: Modeling Rendering Via Transformer for Generalizable Neural Surface Reconstruction' (NeurIPS 2023)

## [Project Page](https://yixunliang.github.io/ReTR/) |  [Paper](https://arxiv.org/pdf/2305.18832.pdf)

>**Abstract:**
Generalizable neural surface reconstruction techniques have attracted great attention in recent years. However, they encounter limitations of low confidence depth distribution and inaccurate surface reasoning due to the oversimplified volume rendering process employed. In this paper, we present Reconstruction TRansformer (ReTR), a novel framework that leverages the transformer architecture to redesign the rendering process, enabling complex render interaction modeling. It introduces a learnable meta-ray token and utilizes the cross-attention mechanism to simulate the interaction of rendering process with sampled points and render the observed color. Meanwhile, by operating within a high-dimensional feature space rather than the color space, ReTR mitigates sensitivity to projected colors in source views. Such improvements result in accurate surface assessment with high confidence. We demonstrate the effectiveness of our approach on various datasets, showcasing how our method outperforms the current state-of-the-art approaches in terms of reconstruction quality and generalization ability. 


## Overview
<img width="1161" alt="pipeline" src="https://github.com/YixunLiang/ReTR/assets/99460842/715e4c3c-6237-443c-9747-16b425fdf52b">

## Note
<font color='red'>We will change our title following the commends from reviewers and ACs, the name in the camera ready version will be: 'ReTR: Modeling Rendering via Transformer for
Generalizable Neural Surface Reconstruction' </font> 


## Installation

### Requirements

* python 3.9
* CUDA 11.1

```
conda create --name retr python=3.9 pip
conda activate retr

pip install -r requirements.txt
```
## Sparse View Reconstruction on DTU

### Tranining

* Download pre-processed [DTU's training set](https://drive.google.com/file/d/1eDjh-_bxKKnEuz5h-HXS7EDJn59clx6V/view) and [Depths_raw](https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/cascade-stereo/CasMVSNet/dtu_data/dtu_train_hr/Depths_raw.zip) (both provided by MVSNet). Then organize the dataset as follows:
```
root_directory
├──Cameras
├──Rectified
└──Depths_raw
```
* In ``train_dtu.sh``, set `DATASET` as the root directory of dataset; set `LOG_DIR` as the directory to store the checkpoints. 

* Train the model by running `bash train_dtu.sh` on GPU.

### Evaluation
#### Download Evaluation Datas
* Download pre-processed [DTU dataset](https://drive.google.com/file/d/1cMGgIAWQKpNyu20ntPAjq3ZWtJ-HXyb4/view?usp=sharing). The dataset is organized as follows:
```
root_directory
├──cameras
    ├── 00000000_cam.txt
    ├── 00000001_cam.txt
    └── ...  
├──pair.txt
├──scan24
├──scan37
      ├── image               
      │   ├── 000000.png       
      │   ├── 000001.png       
      │   └── ...                
      └── mask                   
          ├── 000.png   
          ├── 001.png
          └── ...                
```
* Then, download the [SampleSet] for quantitative evaluation(http://roboimagedata.compute.dtu.dk/?page_id=36) and [Points](http://roboimagedata.compute.dtu.dk/?page_id=36) from DTU's website. Unzip them and place `Points` folder in `SampleSet/MVS Data/`. The structure looks like:
```
SampleSet
├──MVS Data
      └──Points
```
* Following SparseNeuS and VolRecon, you can evalute the result after your training or directly use our checkpoints by respectively running after you set paths following the comment in the scripts:
```
bash eval_dtu.sh               ## Render depth maps and images
bash tsdf_fusion.sh            ## Get the reconstructed meshes
bash clean_mesh.sh             ## Clean the raw mesh with object masks
bash eval_dtu_result.sh        ## Get the quantitative results
```
## Citation
If you find this project useful for your research, please cite: 

```
@misc{liang2023ReTR,
  title={ReTR: Modeling Rendering Via Transformer for Generalizable Neural Surface Reconstruction},
  author={Liang, Yixun and He, Hao and Chen, Ying-cong},
  journal={arXiv preprint arXiv:2305.18832},
  year={2023}
}
```
## Acknowledgement

Our code is based on [VolRecon](https://github.com/IVRL/VolRecon), Thanks for their excellent work.
