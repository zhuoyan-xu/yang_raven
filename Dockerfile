FROM conda/miniconda3

RUN conda env create -f environment.yml

SHELL ["conda", "run", "-n", "RAVEN", "/bin/bash", "-c"]

CMD ["bash"]

