FROM nvcr.io/nvidia/pytorch:22.12-py3

RUN pip uninstall torch -y
COPY . /workspace/vllm
RUN pip install -r /workspace/vllm/launch_deployment/requirements.txt
RUN pip install -e /workspace/vllm
RUN wget https://github.com/peak/s5cmd/releases/download/v2.2.1/s5cmd_2.2.1_Linux-64bit.tar.gz
RUN tar -xvzf s5cmd_2.2.1_Linux-64bit.tar.gz
COPY launch_deployment/vllm_server.py /workspace/vllm_server.py