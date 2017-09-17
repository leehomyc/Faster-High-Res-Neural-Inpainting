## High-Resolution Image Inpainting using Multi-Scale Neural Patch Synthesis


![teaser](images/teaser.png "Sample inpainting results on held-out ImageNet images")

**[update 9/16/2017]** We increase the speed of original version by 6x (30s/image on GPU). The updates include:

1. We remove layer 12 of vgg in texture optimization.
2. In texture optimization, we have three scales and 100 iterations at each scale. Now it only computes the nearest patch at the first iteration of each scale, and re-use the nearest index in later iterations.

This greatly increases the speed at the cost of very subtle inpainting quality. 


This is the code for [High-Resolution Image Inpainting using Multi-Scale Neural Patch Synthesis](https://arxiv.org/pdf/1611.09969). Given an image, we use the content and texture network to jointly infer the missing region. This repository contains the pre-trained model for the content network and the joint optimization code, including the demo to run example images. The code is adapted from the [Context Encoders](https://github.com/pathak22/context-encoder) and [CNNMRF](https://github.com/chuanli11/CNNMRF). Please contact [Harry Yang](http://www.harryyang.org) for questions regarding the paper or the code. Note that the code is for research purpose only.

### Demo

- Install Torch:  http://torch.ch/docs/getting-started.html#_

- Clone the repository
```Shell
  git clone https://github.com/leehomyc/High-Res-Neural-Inpainting.git
```

- Download the [pre-trained models](https://drive.google.com/open?id=0BxYj-YwDqh45XzZVTXF1dnJXY28) for the content and texture networks and put them under the folder models/.

- Run the Demo
```Shell
  cd High-Res-Neural-Inpainting
  # This will use the trained model to generate the output of the content network
  th run_content_network.lua
  # This will use the trained model to run texture optimization
  th run_texture_optimization.lua
  # This will generate the final result
  th blend.lua
```


### Citation

If you find this code useful for your research, please cite:

```
@article{yang2016high,
  title={High-Resolution Image Inpainting using Multi-Scale Neural Patch Synthesis},
  author={Yang, Chao and Lu, Xin and Lin, Zhe and Shechtman, Eli and Wang, Oliver and Li, Hao},
  journal={arXiv preprint arXiv:1611.09969},
  year={2016}
}
```


