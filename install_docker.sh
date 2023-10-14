pip install scipy
conda install cython

# pip3 install cython
pip uninstall pycocotools
echo "Y"
pip install mmpycocotools
pip uninstall mmpycocotools
echo "Y"
pip install mmpycocotools

pip install msgpack
pip install -r ./requirements/build.txt
python3 setup.py install 

pip install onnx==1.9.0
pip install onnxruntime==1.9.0
