set -ex
export UV_HTTP_TIMEOUT=900 UV_CONCURRENT_DOWNLOADS=4 UV_HTTP_RETRIES=8
cd ~/rebot-holoscan
~/.local/bin/uv venv --allow-existing --python 3.12 .venv-isaacsim
~/.local/bin/uv pip install --python .venv-isaacsim/bin/python "isaacsim[all,extscache]==6.2.0.0" --extra-index-url https://pypi.nvidia.com --index-strategy unsafe-best-match
~/.local/bin/uv pip install --python .venv-isaacsim/bin/python pyzmq pyyaml
echo INSTALL_DONE
