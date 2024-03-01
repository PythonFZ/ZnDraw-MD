FROM pytorch/pytorch

RUN conda install -c anaconda git

RUN pip install git+https://github.com/ACEsuit/mace.git
RUN pip install mace_models
RUN pip install --upgrade git+https://github.com/zincware/zndraw

COPY ./ /workspace/simgen
WORKDIR /workspace/simgen

ENTRYPOINT [ "python", "main.py" ]
