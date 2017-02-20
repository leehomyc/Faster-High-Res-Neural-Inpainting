require 'paths'
 paths.dofile('mylib/helper.lua')

 -----------------------------------------
-- Parameters:
-----------------------------------------
-- name: the name of the test, can be anything.
-- content_name: the content image located in folder "data/content"
-- style_name: the style image located in folder "data/style" 
-- ini_method: initial method, set to "image" to use the content image as the initialization; set to "random" to use random noise. 
-- max_size: maximum size of the synthesis image. Default value 384. Larger image needs more time and memory.
-- num_res: number of resolutions. Default value 3. Notice the lowest resolution image should be larger than the patch size otherwise it won't synthesize.
-- num_iter: number of iterations for each resolution. Default value 100 for all resolutions. 

-- mrf_layers: the layers for MRF constraint. Usualy layer 21 alone already gives decent results. Including layer 12 may improve the results but at significantly more computational cost.
-- mrf_weight: weight for each MRF layer. Default value 1e-4. Higher weights leads to more style faithful results.
-- mrf_patch_size: the patch size for MRF constraint. Default value 3. This value is defined seperately for each MRF layer.
-- mrf_num_rotation: To matching objects of different poses. Default value 0. This value is shared by all MRF layers. The total number of rotatoinal copies is "2 * mrf_num_rotation + 1"
-- mrf_num_scale: To matching objects of different scales. Default value 0. This value is shared by all MRF layers. The total number of scaled copies is "2 * mrf_num_scale + 1"
-- mrf_sample_stride: stride to sample mrf on style image. Default value 2. This value is defined seperately for each MRF layer.
-- mrf_synthesis_stride: stride to sample mrf on synthesis image. Default value 2. This value is defined seperately for each MRF layer.
-- mrf_confidence_threshold: threshold for filtering out bad matching. Default value 0 -- means we keep all matchings. This value is defined seperately for all layers.

-- content_layers: the layers for content constraint. Default value 23.
-- content_weight: The weight for content constraint. Default value 2e1. Increasing this value will make the result more content faithful. Decreasing the value will make the method more style faithful. Notice this value should be increase (for example, doubled) if layer 12 is included for MRF constraint,  

-- tv_weight: TV smoothness weight. Default value 1e-3.

-- mode: speed or memory. Try 'speed' if you have a GPU with more than 4GB memory, and try 'memory' otherwise. The 'speed' mode is significantly faster (especially for synthesizing high resolutions) at the cost of higher GPU memory.
-- gpu_chunck_size_1: Size of chunks to split feature maps along the channel dimension. This is to save memory when normalizing the matching score in mrf layers. Use large value if you have large gpu memory. As reference we use 256 for Titan X, and 32 for Geforce GT750M 2G.
-- gpu_chunck_size_2: Size of chuncks to split feature maps along the y dimension. This is to save memory when normalizing the matching score in mrf layers. Use large value if you have large gpu memory. As reference we use 16 for Titan X, and 2 for Geforce GT750M 2G.
-- backend: Use 'cudnn' for CUDA-enabled GPUs or 'clnn' for OpenCL.


local list_params = {
{'demo', 'fake_0001.png', 'real_0001.jpg',  'image', 512, 3, {100, 100, 100}, {12, 21}, {1e-4, 1e-4}, {3, 3}, 1, 1, {2, 2}, {2, 2}, {0, 0}, {23}, 2e1, 1e-4, 'speed', 256, 16, 'cudnn'},
{'demo', 'fake_0002.png', 'real_0002.jpg',  'image', 512, 3, {100, 100, 100}, {12, 21}, {1e-4, 1e-4}, {3, 3}, 1, 1, {2, 2}, {2, 2}, {0, 0}, {23}, 2e1, 1e-4, 'speed', 256, 16, 'cudnn'},
{'demo', 'fake_0003.png', 'real_0003.jpg',  'image', 512, 3, {100, 100, 100}, {12, 21}, {1e-4, 1e-4}, {3, 3}, 1, 1, {2, 2}, {2, 2}, {0, 0}, {23}, 2e1, 1e-4, 'speed', 256, 16, 'cudnn'},
{'demo', 'fake_0004.png', 'real_0004.jpg',  'image', 512, 3, {100, 100, 100}, {12, 21}, {1e-4, 1e-4}, {3, 3}, 1, 1, {2, 2}, {2, 2}, {0, 0}, {23}, 2e1, 1e-4, 'speed', 256, 16, 'cudnn'},
{'demo', 'fake_0005.png', 'real_0005.jpg',  'image', 512, 3, {100, 100, 100}, {12, 21}, {1e-4, 1e-4}, {3, 3}, 1, 1, {2, 2}, {2, 2}, {0, 0}, {23}, 2e1, 1e-4, 'speed', 256, 16, 'cudnn'},
{'demo', 'fake_0006.png', 'real_0006.jpg',  'image', 512, 3, {100, 100, 100}, {12, 21}, {1e-4, 1e-4}, {3, 3}, 1, 1, {2, 2}, {2, 2}, {0, 0}, {23}, 2e1, 1e-4, 'speed', 256, 16, 'cudnn'},
{'demo', 'fake_0007.png', 'real_0007.jpg',  'image', 512, 3, {100, 100, 100}, {12, 21}, {1e-4, 1e-4}, {3, 3}, 1, 1, {2, 2}, {2, 2}, {0, 0}, {23}, 2e1, 1e-4, 'speed', 256, 16, 'cudnn'},
}

run_tests(require 'transfer_CNNMRF_wrapper_2_large2_new_model_1000', list_params)