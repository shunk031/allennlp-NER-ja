#!/usr/bin/env bash

set -eux

python -m pip install -U torch==1.10.2+cu111 torchvision==0.11.3+cu111 -f https://download.pytorch.org/whl/torch_stable.html
