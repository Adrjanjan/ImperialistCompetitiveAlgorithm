FROM tensorflow/tensorflow:latest-gpu
COPY . .
RUN pip3 install -e .
RUN pip3 install sklearn matplotlib
CMD bash