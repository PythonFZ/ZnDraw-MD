FROM pytorch/pytorch

RUN conda install -c conda-forge packmol git

RUN pip install -r requirements.txt

WORKDIR /app
COPY . /app

ENTRYPOINT [ "python", "main.py" ]
