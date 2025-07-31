# AAA-Gaussians: Anti-Aliased and Artifact-Free 3D Gaussian Rendering

[Michael Steiner](https://steimich96.github.io/)<sup>&#42;</sup>,
[Thomas Köhler](https://scholar.google.com/citations?user=pMDepi0AAAAJ&hl=en)<sup>&#42;</sup>,
[Lukas Radl](https://r4dl.github.io/), 
[Felix Windisch](https://scholar.google.com/citations?user=J_Jm3Y4AAAAJ&hl=en), 
[Dieter Schmalstieg](https://www.tugraz.at/institute/icg/research/team-schmalstieg/), 
[Markus Steinberger](https://www.markussteinberger.net/)
<br> 
<sup>&#42;</sup> denotes equal contribution

[Project Page](https://derthomy.github.io/AAA-Gaussians)
| [Full Paper](https://arxiv.org/pdf/2504.12811)
| [Dataset](https://cloud.tugraz.at/index.php/s/fsDoKofW4T63xN2)

![Teaser image](assets/teaser.png)

This repository contains the official authors implementation associated with the paper "AAA-Gaussians: Anti-Aliased and Artifact-Free 3D Gaussian Rendering", which can be found [here](https://derthomy.github.io/AAA-Gaussians). 

<br>

<p align="center">
<a href="https://www.tugraz.at/en/home"><img height="90" src="assets/tugraz-logo.jpg"> </a>
<a href="https://tugraz.elsevierpure.com/en/organisations/huawei-technologies-austria-gmbh-98631"> <img height="90" src="assets/huawei-logo.jpg"> </a>
<a href="https://www.uni-stuttgart.de/en/"><img height="90" src="assets/unistuttgart-logo.jpg"> </a>
<p>

Abstract: *Although 3D Gaussian Splatting (3DGS) has revolutionized 3D reconstruction, it still faces challenges such as aliasing, projection artifacts, and view inconsistencies, primarily due to the simplification of treating splats as 2D entities. We argue that incorporating full 3D evaluation of Gaussians throughout the 3DGS 
pipeline can effectively address these issues while preserving rasterization efficiency. Specifically, we introduce an adaptive 3D smoothing filter to mitigate aliasing and present a stable view-space bounding method that eliminates popping artifacts when Gaussians extend beyond the view frustum. Furthermore, we promote tile-based culling to 3D with screen-space planes, accelerating rendering and reducing sorting costs for hierarchical rasterization. Our method achieves state-of-the-art quality on in-distribution evaluation sets and significantly outperforms other approaches for out-of-distribution views. Our qualitative evaluations further demonstrate the effective removal of aliasing, distortions, and popping artifacts, ensuring real-time, artifact-free rendering.*


<section class="section" id="BibTeX">
  <div class="container is-max-desktop content">
    <h2 class="title">BibTeX</h2>
    <pre><code>@inproceedings{steiner2025aaags,
  author    = {Steiner, Michael and K{\"o}hler, Thomas and Radl, Lukas and Windisch, Felix and Schmalstieg, Dieter and Steinberger, Markus},
  title     = {{AAA-Gaussians: Anti-Aliased and Artifact-Free 3D Gaussian Rendering}},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year      = {2025}
}</code></pre>
  </div>
</section>

## Overview
Our repository is built on [StopThePop](https://github.com/r4dl/StopThePop), which is in turn based on [Inria 3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/).
For a full breakdown on how to get the code running, please consider [3DGS's Readme](https://github.com/graphdeco-inria/gaussian-splatting/blob/main/README.md).

The project is split into submodules, each maintained in a separate github repository:

* [AAA-Gaussians-Rasterization](https://github.com/DerThomy/AAA-Gaussians-Rasterization): A clone of the [StopThePop-Rasterization](https://github.com/r4dl/StopThePop-Rasterization) which build on the original [diff-gaussian-rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization) that contains our CUDA rasterizer implementation
* [AAA-Gaussians-SIBR](https://github.com/DerThomy/AAA-Gaussians-SIBR): A clone of the [SIBR_StopThePop](https://github.com/r4dl/SIBR_StopThePop) which builds on the [SIBR Core](https://gitlab.inria.fr/sibr/sibr_core) project, containing an adapted viewer with our additional settings and functionalities

## Licensing

The majority of the projects is licensed under the ["Gaussian-Splatting License"](LICENSE.md), with the exception of:

* [StopThePop header files](submodules/diff-gaussian-rasterization/cuda_rasterizer/stopthepop): MIT License
* [FLIP](utils/flip.py): BSD-3 License

For more information refer to our [Notice](NOTICE.md) and the ["Gaussian-Splatting License"](LICENSE.md).

We also make use of additional publicly available code published under the same ["Gaussian-Splatting License"](LICENSE.md)

* [Mip Splatting](utils/mip_filter.py): https://github.com/autonomousvision/mip-splatting
* [3D Gaussian Splatting as Markov Chain Monte Carlo](train.py): https://github.com/ubc-vision/3dgs-mcmc

There are also several changes in the original source code.
If you use any of our additional functionalities, please cite our paper and link to this repository.

## Cloning the Repository

The repository contains submodules, thus please check it out with 
```shell
# HTTPS
git clone https://github.com/DerThomy/AAA-Gaussians --recursive
```

## Setup

### Local Setup

Our default, provided install method is based on Conda package and environment management:
```shell
SET DISTUTILS_USE_SDK=1 # Windows only
conda env create --file environment.yml
conda activate aaa-gs
```

> **Note:** This process assumes that you have CUDA SDK **12** installed.

Subsequently, install the CUDA rasterizer:
```shell
pip install submodules/diff-gaussian-rasterization
```

> **Note:** This can take several minutes. If you experience unreasonably long build times, consider using [StopThePop fast build mode](#stopthepop_fastbuild).

### Running

The `train.py` script takes a `.json` config file as the argument `--splatting_config`, which should contain the following information (this example is also the default `config.json`, if none is provided):

```cpp
{
  "sort_settings": 
  {
    "sort_mode": 3,    // Global (0), Per-Pixel Full (1), Per-Pixel K-Buffer (2), Hierarchical (3)
    "sort_order": 3,   /* Viewspace Z-Depth (0), Worldspace Distance (1), 
                          Per-Tile Depth at Tile Center (2), Per-Tile Depth at Max Contrib. Pos. (3) */
    "queue_sizes": 
    {
      "per_pixel": 4,  // Used for: Per-Pixel K-Buffer and Hierarchical
      "tile_2x2": 8,   // Used only for Hierarchical
      "tile_4x4": 64   // Used only for Hierarchical
    }
  },
  "culling_settings": 
  {
    "rect_bounding": true,            // Bound 2D Gaussians with a rectangle (instead of a square)
    "tight_opacity_bounding": true,   // Bound 2D Gaussians by considering their opacity value
    "tile_based_culling": true,       /* Cull Tiles where the max. contribution is below the alpha threshold;
                                          Recommended to be used together with Load Balancing*/
    "hierarchical_4x4_culling": true, // Used only for Hierarchical
  },
  "load_balancing": true,      // Use load balancing for per-tile calculations (culling, depth, and duplication)
  "proper_ewa_scaling": true,  /* For "eval_3D: false": Proper dilation of opacity, as proposed by Yu et al. ("Mip-Splatting");
                                  For "eval_3D: true": Our proposed adaptive 3D anti aliasing filter;
                                  Model also needs to be trained with this setting */
  "eval_3D": true              // Use our 3D evaluation instead of 3DGS projection
  "new_aabb": true             // Use our view space plane bounding for bounding box calculation
}
```
These values can be overwritten through the command line. 
Call `python train.py --help` to see all available options.
At the start of training, the provided arguments will be written into the output directory.
The `render.py` script uses the `config.json` in the model directory per default, with the option to overwrite through the command line.

To train different models you can create your own config file based on our config and run:

```shell
python train.py --splatting_config configs/your_config.json -s <path to COLMAP or NeRF Synthetic dataset>
```

<details>
<summary><span style="font-weight: bold;">New Command Line Arguments for train.py</span></summary>

  #### --cap_max
  The maximum number of Gaussians to densify to for training.
  #### --splatting_config
  Full config to specify the flavor of Gaussian Splatting. See ```configs/``` for pre-defined configs.
  #### --sort_mode
  Specify Sort Mode - must be one of ```{GLOBAL,PPX_FULL,PPX_KBUFFER,HIER}```
  #### --sort_order 
  Specify Sort Order - must be one of ```{Z_DEPTH,DISTANCE,PTD_CENTER,PTD_MAX}```
  #### --tile_4x4     
  Specify size of 4x4 tile queue - only needed if using sort_mode ```HIER```, only ```64``` supported.
  #### --tile_2x2 
  Specify size of 2x2 tile queue - only needed if using sort_mode ```HIER```, only ```{8,12,20}``` supported.
  #### --per_pixel {1,2,4,8,12,16,20,24}
  Specify size of per-pixel queue: If using sort_mode ```HIER```, only ```{4,8,16}``` supported. If using sort_mode ```KBUFFER```, all values are supported.
  #### --rect_bounding 
  Bound 2D Gaussians with a rectangle instead of a circle - must be one of ```{True,False}```
  #### --tight_opacity_bounding 
  Bound 2D Gaussians by considering their opacity - must be one of ```{True,False}```
  #### --tile_based_culling 
  Cull complete tiles based on opacity - must be one of ```{True,False}``` (recommended with Load Balancing)
  #### --hierarchical_4x4_culling 
  Cull Gaussians for 4x4 subtiles - must be one of ```{True,False}```, only when using sort_mode ```HIER```
  #### --load_balancing {True,False}
  Perform per-tile computations cooperatively (e.g. duplication) -  must be one of ```{True,False}```
  #### --proper_ewa_scaling 
  Dilation of 2D Gaussians (no eval_3D) / 3D Gaussians (eval_3D) -  must be one of ```{True,False}```
  #### --eval_3D
  Evaluate Gaussians in 3D as proposed by Hahlbohm et. al. ("Efficient Perspective-Correct 3D Gaussian Splatting Using Hybrid Transparency") - must be one of ```{True,False}```
  #### --new_aabb
  If eval_3D, bounds Gaussians with our novel proposed view space plane bounding - must be one of ```{True,False}```

</details>
<details>
  <summary><span style="font-weight: bold; opacity: 50%;">Original Command Line Arguments for train.py</span></summary>

  #### --source_path / -s
  Path to the source directory containing a COLMAP or Synthetic NeRF data set.
  #### --model_path / -m 
  Path where the trained model should be stored (```output/<random>``` by default).
  #### --images / -i
  Alternative subdirectory for COLMAP images (```images``` by default).
  #### --eval
  Add this flag to use a MipNeRF360-style training/test split for evaluation.
  #### --resolution / -r
  Specifies resolution of the loaded images before training. If provided ```1, 2, 4``` or ```8```, uses original, 1/2, 1/4 or 1/8 resolution, respectively. For all other values, rescales the width to the given number while maintaining image aspect. **If not set and input image width exceeds 1.6K pixels, inputs are automatically rescaled to this target.**
  #### --data_device
  Specifies where to put the source image data, ```cuda``` by default, recommended to use ```cpu``` if training on large/high-resolution dataset, will reduce VRAM consumption, but slightly slow down training. Thanks to [HrsPythonix](https://github.com/HrsPythonix).
  #### --white_background / -w
  Add this flag to use white background instead of black (default), e.g., for evaluation of NeRF Synthetic dataset.
  #### --sh_degree
  Order of spherical harmonics to be used (no larger than 3). ```3``` by default.
  #### --convert_SHs_python
  Flag to make pipeline compute forward and backward of SHs with PyTorch instead of ours.
  #### --compute_cov3D_python
  Flag to make pipeline compute forward and backward of the 3D covariance with PyTorch instead of ours.
  #### --debug
  Enables debug mode if you experience errors. If the rasterizer fails, a ```dump``` file is created that you may forward to us in an issue so we can take a look.
  #### --debug_from
  Debugging is **slow**. You may specify an iteration (starting from 0) after which the above debugging becomes active.
  #### --iterations
  Number of total iterations to train for, ```30_000``` by default.
  #### --ip
  IP to start GUI server on, ```127.0.0.1``` by default.
  #### --port 
  Port to use for GUI server, ```6009``` by default.
  #### --test_iterations
  Space-separated iterations at which the training script computes L1 and PSNR over test set, ```7000 30000``` by default.
  #### --save_iterations
  Space-separated iterations at which the training script saves the Gaussian model, ```7000 30000 <iterations>``` by default.
  #### --checkpoint_iterations
  Space-separated iterations at which to store a checkpoint for continuing later, saved in the model directory.
  #### --start_checkpoint
  Path to a saved checkpoint to continue training from.
  #### --quiet 
  Flag to omit any text written to standard out pipe. 
  #### --feature_lr
  Spherical harmonics features learning rate, ```0.0025``` by default.
  #### --opacity_lr
  Opacity learning rate, ```0.05``` by default.
  #### --scaling_lr
  Scaling learning rate, ```0.005``` by default.
  #### --rotation_lr
  Rotation learning rate, ```0.001``` by default.
  #### --noise_lr
  Noise learning rate for [MCMC](https://github.com/ubc-vision/3dgs-mcmc) densification ```0.00005```
  #### --scale_reg
  Scale regularization for [MCMC](https://github.com/ubc-vision/3dgs-mcmc) densification ```0.01```
  #### --opacity_reg
  Opacity regularization for [MCMC](https://github.com/ubc-vision/3dgs-mcmc) densification ```0.01```
  #### --position_lr_max_steps
  Number of steps (from 0) where position learning rate goes from ```initial``` to ```final```. ```30_000``` by default.
  #### --position_lr_init
  Initial 3D position learning rate, ```0.00016``` by default.
  #### --position_lr_final
  Final 3D position learning rate, ```0.0000016``` by default.
  #### --position_lr_delay_mult
  Position learning rate multiplier (cf. Plenoxels), ```0.01``` by default. 
  #### --densify_from_iter
  Iteration where densification starts, ```500``` by default. 
  #### --densify_until_iter
  Iteration where densification stops, ```15_000``` by default.
  #### --densify_grad_threshold
  Limit that decides if points should be densified based on 2D position gradient, ```0.0002``` by default.
  #### --densification_interval
  How frequently to densify, ```100``` (every 100 iterations) by default.
  #### --opacity_reset_interval
  How frequently to reset opacity, ```3_000``` by default. 
  #### --lambda_dssim
  Influence of SSIM on total loss from 0 to 1, ```0.2``` by default. 
  #### --percent_dense
  Percentage of scene extent (0--1) a point must exceed to be forcibly densified, ```0.01``` by default.
  </details>
<br>

### Evaluation

By default, the trained models use all available images in the dataset. 
To train them while withholding a test set for evaluation, use the ```--eval``` flag. 
This way, you can render training/test sets and produce error metrics as follows:

```shell
python train.py -s <path to COLMAP or NeRF Synthetic dataset> --eval # Train with train/test split
python render.py -m <path to trained model> # Generate renderings and gaussian count
python metrics.py -m <path to trained model> # Compute error metrics on renderings
```

<details>
<summary><span style="font-weight: bold; opacity: 50%;">Command Line Arguments for render.py</span></summary>

  #### --render_depth
  Flag to enable depth rendering.
  #### --skip_train
  Flag to skip rendering the training set.
  #### --skip_test
  Flag to skip rendering the test set.
  #### --model_path / -m 
  Path to the trained model directory you want to create renderings for.
  #### --quiet 
  Flag to omit any text written to standard out pipe. 

  **The below parameters will be read automatically from the model path, based on what was used for training. However, you may override them by providing them explicitly on the command line.** 

  #### --source_path / -s
  Path to the source directory containing a COLMAP or Synthetic NeRF data set.
  #### --images / -i
  Alternative subdirectory for COLMAP images (```images``` by default).
  #### --eval
  Add this flag to use a MipNeRF360-style training/test split for evaluation.
  #### --resolution / -r
  Changes the resolution of the loaded images before training. If provided ```1, 2, 4``` or ```8```, uses original, 1/2, 1/4 or 1/8 resolution, respectively. For all other values, rescales the width to the given number while maintaining image aspect. ```1``` by default.
  #### --white_background / -w
  Add this flag to use white background instead of black (default), e.g., for evaluation of NeRF Synthetic dataset.
  #### --convert_SHs_python
  Flag to make pipeline render with computed SHs from PyTorch instead of ours.
  #### --compute_cov3D_python
  Flag to make pipeline render with computed 3D covariance from PyTorch instead of ours.
</details>

<details>
<summary><span style="font-weight: bold; opacity: 50%;">Command Line Arguments for metrics.py</span></summary>

  #### --model_paths / -m 
  Space-separated list of model paths for which metrics should be computed.
</details>
<br>

We further provide the ```full_eval.py``` script.
This script specifies the routine used in our evaluation and demonstrates the use of some additional parameters, e.g., ```--images (-i)``` to define alternative image directories within COLMAP data sets.
If you have downloaded and extracted all the training data, you can run it like this:

```shell
python full_eval.py -m360 <mipnerf360 folder> -tat <tanks and temples folder> -db <deep blending folder> --config <splatting config file>
```

#### Pre-Trained Models
If you want to evaluate our [pre-trained models](https://cloud.tugraz.at/index.php/s/fsDoKofW4T63xN2), you have to download the source datsets and indicate their location to ```render.py```, just as done here:
```shell
python render.py -m <path to pre-trained model> -s <path to COLMAP dataset>
```
Alternatively, you can modify the ```source_path``` with the ```cfg_args```-file and manually insert the correct path. 

> **Note:** We included our models, which were used in our evaluation: to minimize file size, we only include the final checkpoint. We also include the final, rendered images, hence you can reproduce our results easily.

<details>
<summary><span style="font-weight: bold; opacity: 50%;">Command Line Arguments for full_eval.py</span></summary>

  #### --skip_training
  Flag to skip training stage.
  #### --skip_rendering
  Flag to skip rendering stage.
  #### --skip_metrics
  Flag to skip metrics calculation stage.
  #### --output_path
  Directory to put renderings and results in, ```./eval``` by default, set to pre-trained model location if evaluating them.
  #### --mipnerf360 / -m360
  Path to MipNeRF360 source datasets, required if training or rendering.
  #### --tanksandtemples / -tat
  Path to Tanks&Temples source datasets, required if training or rendering.
  #### --deepblending / -db
  Path to Deep Blending source datasets, required if training or rendering.
</details>
<br>

## Interactive Viewers
Following 3DGS, we provide interactive viewers for our method: remote and real-time. 
Our viewing solutions are based on the [SIBR_StopThePop](https://github.com/r4dl/SIBR_StopThePop) which extends the [SIBR](https://sibr.gitlabpages.inria.fr/) framework, developed by the GRAPHDECO group for several novel-view synthesis projects.
Our modified viewer contains additional debug modes, and options to disable several of our proposed optimizations.
The settings on startup are based on the `config.json` file in the model directory (if it exists).
The implementation is hosted [here](https://github.com/DerThomy/AAA-Gaussians-SIBR).
Hardware requirements and setup steps are identical to 3DGS, hence, refer to the [corresponding README](https://github.com/graphdeco-inria/gaussian-splatting/blob/main/README.md) for details.

### Quick Setup: Installation from Source
If you cloned with submodules (e.g., using ```--recursive```), the source code for the viewers is found in ```SIBR_viewers```.

#### Windows
CMake should take care of your dependencies.
```shell
cd SIBR_viewers
cmake -Bbuild .
cmake --build build --target install --config Release
```
You may specify a different configuration, e.g. ```RelWithDebInfo``` or ```Debug``` if you need more control during development.

#### Ubuntu 22.04
You will need to install a few dependencies before running the project setup.
```shell
# Dependencies
sudo apt install -y libglew-dev libassimp-dev libboost-all-dev libgtk-3-dev libopencv-dev libglfw3-dev libavdevice-dev libavcodec-dev libeigen3-dev libxxf86vm-dev libembree-dev
# Project setup
cd SIBR_viewers
cmake -Bbuild . -DCMAKE_BUILD_TYPE=Release # add -G Ninja to build faster
cmake --build build -j24 --target install
``` 

### Running the Interactive Viewer

After extracting or installing the viewers, you may run the compiled ```SIBR_gaussianViewer_app[_config]``` app in ```<SIBR install dir>/bin```, e.g.: 
```shell
./<SIBR install dir>/bin/SIBR_gaussianViewer_app -m <path to trained model> --rendering-size <rendering size, e.g. 1920 1080>
```

It should suffice to provide the ```-m``` parameter pointing to a trained model directory. Alternatively, you can specify an override location for training input data using ```-s```. Combine it with ```--force-aspect-ratio``` if you want the exact resolution and don't mind image distortion.

#### STOPTHEPOP_FASTBUILD
For performance reasons, StopThePop uses templates for several of their options, causing very long compile times for our submodule and SIBR.
Hence, they provide a ```STOPTHEPOP_FASTBUILD``` option in [```submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer.h```](submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer.h).
We have this option enabled by default as we found that their default config works fine. If you plan to change the configuration simply comment out
```cpp
#define STOPTHEPOP_FASTBUILD
``` 
This will however result in longer compile times!

> **Note:** For ```SIBR```, the corresponding ```CMakeLists.txt``` is located in [```SIBR_viewers/extlibs/CudaRasterizer/CudaRasterizer/CMakeLists.txt```](SIBR_viewers/extlibs/CudaRasterizer/CudaRasterizer/CMakeLists.txt), and ```rasterizer.h``` is located in [```SIBR_viewers/extlibs/CudaRasterizer/CudaRasterizer/cuda_rasterizer/rasterizer.h```](SIBR_viewers/extlibs/CudaRasterizer/CudaRasterizer/cuda_rasterizer/rasterizer.h)

## FAQ
Please consider 3DGS's FAQ, contained in [their README](https://github.com/graphdeco-inria/gaussian-splatting/blob/main/README.md). In addition, several issues are also covered on [3DGS's issues page](https://github.com/graphdeco-inria/gaussian-splatting/issues).
We will update this FAQ as issues arise.
