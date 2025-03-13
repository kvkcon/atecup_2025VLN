FROM ac2-registry.cn-hangzhou.cr.aliyuncs.com/ac2/pytorch-ubuntu:2.3.0-cuda12.1.1-ubuntu22.04
RUN pip install flask opencv-python
RUN pip install opencv-python-headless matplotlib tqdm pyyaml psutil
RUN pip install --no-deps ultralytics
RUN mkdir -p /home/admin/atec_project
COPY checkpoints /home/admin/atec_project/checkpoints
COPY *.py /home/admin/atec_project/
COPY *.sh /home/admin/atec_project/
RUN ln -s /bin/python3 /bin/python
