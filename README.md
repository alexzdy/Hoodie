<h1 style="font-weight: bold">
  Hoodie: Hierarchical Point Cloud and Latent Code Diffusion for Dressed Avatar Generation
</h1>


## 📌 Environment 
    conda create -n hoodie python=3.8
    conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
    conda install tensroboard
    conda install h5py
    pip3 install open3d

## 📌 Data Preparation
Please download paired human-garment point cloud dataset in the following link and place four training dataset files `Hoodie\human-garment\2048\*-2048` in human-garment.zip in our `./data` folder. 
Google Drive: **[[Google Drive]](https://drive.google.com/file/d/1P3ni1v6QvcIc2JZ1Kl8FANnfI9o0lP5X/view?usp=drive_link)**  
Baiduyun Drive: **[[Baidu Drive]](https://pan.baidu.com/s/1xPfo2tYLyfHkdVo8_ffSUw)**, Password: g7e1. 

Please note that our point cloud data is obtained based on **[Deep Fashion3D V2](https://github.com/GAP-LAB-CUHK-SZ/deepFashion3D)**.

## 📌 Training
Train the Point Cloud Diffusion and then the Joint Diffusion.
Thanks to the code repository from **[https://github.com/luost26/diffusion-point-cloud](https://github.com/luost26/diffusion-point-cloud)**.
You can set dataset using `--categories` and `--split`.

### Latent Encoder
    python train_stage1.py --categories human-upper-2048 --split upper
    python train_stage1.py --categories garment-upper-2048 --split upper

When you finished training the latent encoder, save the human and garment latents. The global latent code is a proxy representation of the point cloud for 1D diffusion training.

    python save_latents.py --categories human-upper-2048 --split upper --resume_path {human.pt path}
    python save_latents.py --categories garment-upper-2048 --split upper --resume_path {garment.pt path}

### Joint Diffusion

For modeling the joint distribution across two latent spaces:

    python cat.py
    python train_stage2.py --cat_path  --save_model

## 📌 Sampling
After training, we can implement the three main functions of Hoodie by the following operations. You can set generation category using `--categories` and set weight using `--stage1_human`, `--stage1_garment` and `--stage2_diffusion`.

### Joint Generaton
You can use the following operation for joint generation: 

    python joint-generation.py

### Conditional Inference
You can use the following operation for conditional inference:

    python conditional-inference.py

### Conditional Generation
You can use the following operation for conditional generation:

    python conditional-generation.py

## 📌 Evaluation
We use [PU-GAN](https://github.com/liruihui/PU-GAN) to unsample point cloud and then evaluate it using [torch-fidelity](https://github.com/toshas/torch-fidelity). 

