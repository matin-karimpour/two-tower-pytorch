# Two-Tower model with pytorch

In this repository implemented the Two-Tower model with PyTorch.

run `main.py` with below options:

- -h, --help            show this help message and exit

- -c, CONFIG, --config CONFIG
                        config model and dataset path
- -o, ONNX_PATH, --onnx_path ONNX_PATH
                        ONNX model path.
- -p, PT_PATH, --pt_path PT_PATH
                        pt model path
- -d, DATA, --data DATA  load csv file for train or inference
- -s, SAVE_DIR, --save_dir SAVE_DIR
                        save model path in train mode
- -e, EPOCHS, --epochs EPOCHS
                        set epochs for train mode
- --patience PATIENCE   set patience for early stopping in train mode
- --save_onnx           save ONNX file in training mode
- --train for train model
- --onnx_predict for inference model by ONNX file
- --pt_predict for inference model by pt file