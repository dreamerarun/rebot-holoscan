set -ex
export UV_HTTP_TIMEOUT=900 UV_CONCURRENT_DOWNLOADS=4
cd ~/rebot-holoscan
~/.local/bin/uv pip install --python .venv-holoscan/bin/python torch torchvision --index-url https://download.pytorch.org/whl/cu130
~/.local/bin/uv pip install --python .venv-holoscan/bin/python opencv-python
.venv-holoscan/bin/python -c "import torch,torchvision,cv2;print(torch.__version__,torchvision.__version__,cv2.__version__,torch.cuda.is_available(),torch.cuda.get_device_name(0))"
echo VISION_INSTALL_DONE
