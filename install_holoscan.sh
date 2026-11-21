set -ex
cd ~/rebot-holoscan
~/.local/bin/uv venv --python 3.12 .venv-holoscan
~/.local/bin/uv pip install --python .venv-holoscan/bin/python holoscan-cu13==5.0.0 motorbridge==0.5.* pin numpy pyyaml pyzmq transforms3d
.venv-holoscan/bin/python -c "import holoscan,motorbridge,pinocchio,zmq;print(\"holoscan\",holoscan.__version__);print(\"OK\")"
