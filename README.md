## Introduction


## Installation


## Data preparation

### WIDERFace:
  1. Download WIDERFace datasets and put it under `data/retinaface`.
  2. Download annotation files from [gdrive](https://drive.google.com/file/d/1UW3KoApOhusyqSHX96yEDRYiNkd3Iv3Z/view?usp=sharing) and put them under `data/retinaface/`
 
   ```
     data/retinaface/
         train/
             images/
             labelv2.txt
         val/
             images/
             labelv2.txt
             gt/
                 *.mat
             
   ```
 

#### Annotation Format 

*please refer to labelv2.txt for detail*

For each image:
  ```
  # <image_path> image_width image_height
  bbox_x1 bbox_y1 bbox_x2 bbox_y2 (<keypoint,3>*N)
  ...
  ...
  # <image_path> image_width image_height
  bbox_x1 bbox_y1 bbox_x2 bbox_y2 (<keypoint,3>*N)
  ...
  ...
  ```
Keypoints can be ignored if there is bbox annotation only.


## Training

Example training command, with 4 GPUs:
```
CUDA_VISIBLE_DEVICES="0,1,2,3" PORT=29701 bash ./tools/dist_train.sh ./configs/scrfd/scrfd_1g.py 4
```

## WIDERFace Evaluation

We use a pure python evaluation script without Matlab.

```
GPU=0
GROUP=scrfd
TASK=scrfd_2.5g
CUDA_VISIBLE_DEVICES="$GPU" python -u tools/test_widerface.py ./configs/"$GROUP"/"$TASK".py ./work_dirs/"$TASK"/model.pth --mode 0 --out wouts
```

## Convert to ONNX

Please refer to `tools/scrfd2onnx.py`

Generated onnx model can accept dynamic input as default.

You can also set specific input shape by pass ``--shape 640 640``, then output onnx model can be optimized by onnx-simplifier.


## Inference

Please refer to `tools/scrfd.py` which uses onnxruntime to do inference.

## Network Search

For two-steps search as we described in paper, we target hard mAP on how we select best candidate models.

We provide an example for searching SCRFD-2.5GF in this repo as below.

1. For searching backbones: 

    ```
    python search_tools/generate_configs_2.5g.py --mode 1
    ```
   Where ``mode==1`` means searching backbone only. For other parameters, please check the code.
2. After step-1 done, there will be ``configs/scrfdgen2.5g/scrfdgen2.5g_1.py`` to ``configs/scrfdgen2.5g/scrfdgen2.5g_64.py`` if ``num_configs`` is set to 64.
3. Do training for every generated configs for 80 epochs, please check ``search_tools/search_train.sh``
4. Test WIDERFace precision for every generated configs, using ``search_tools/search_test.sh``.
5. Select the top accurate config as the base template(assume the 10-th config is the best), then do the overall network search. 
    ```
    python search_tools/generate_configs_2.5g.py --mode 2 --template 10
    ```
6. Test these new generated configs again and select the top accurate one(s).

