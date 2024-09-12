FROM pytorch/pytorch

RUN conda install -c conda-forge packmol git

WORKDIR /app
COPY . /app

RUN pip install -r requirements.txt

CMD [ "python", "main.py" ]
