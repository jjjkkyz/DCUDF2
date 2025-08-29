# DCUDF2: Improving Efficiency and Accuracy in Extracting Zero Level Sets from Unsigned Distance Fields (TVCG 2025)
## [<a href="https://arxiv.org/abs/2408.17284" target="_blank">Paper Page</a>]

We now release main code of our algorithm. 
You can use our code in dcudf2 folder to extract mesh from unsigned distance fields.


# Install
    # we use torch to calculate gridient
    conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia

    # some referenced package
    pip install open3d trimesh matplotlib scipy scikit-image

    # use mini-cut 
    pip install PyMaxflow

    # if use 3D IoU to seperate watertight double cover mesh
    pip install git+https://github.com/facebookresearch/pytorch3d.git@stable

    # to use our train and test code, we need pyhocon to init config
    pip install pyhocon==0.3.59

# Usage
    from dcudf2.mesh_extraction import dcudf2

    query_fun = lambda pts: udf_network.udf(pts)
    resolution = 256
    threshold = 0.005

    # we have a lot default parameters, see source code for details.
    extractor = dcudf2(query_fun, resolution, threshold)
    mesh = extractor.optimize()

# Demo
    # to run our demo
    python evaluate.py --conf confs/cloth.conf --gpu 0 --dataname 564



## Citation
```
@article{chen2025dcudf2,
  title={DCUDF2: Improving Efficiency and Accuracy in Extracting Zero Level Sets from Unsigned Distance Fields},
  author={Chen, Xuhui and Yu, Fugang and Hou, Fei and Wang, Wencheng and Zhang, Zhebin and He, Ying},
  journal={IEEE Transactions on Visualization and Computer Graphics},
  year={2025},
  publisher={IEEE}
}
```

