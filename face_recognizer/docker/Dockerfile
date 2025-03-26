ARG REPOSITORY="docker.io"
FROM python:3.8-bookworm

RUN rm -f /var/lib/dpkg/statoverride && rm -f /var/lib/dpkg/lock && dpkg --configure -a
RUN apt-get update && apt-get install -y -o Dpkg::Options::="--force-confold" ffmpeg libsm6 libxext6

ADD requirements.txt .
RUN pip3 install --upgrade pip && pip3 install -r requirements.txt

RUN mkdir /face_recognizer_root
ENV PYTHONPATH "${PYTHONPATH}:/face_recognizer_root"
ADD face_encoder /face_recognizer_root/face_encoder
ADD tests /face_recognizer_root/tests
ADD resources /face_recognizer_root/resources
ADD models /face_recognizer_root/models
ADD VERSION /face_recognizer_root

CMD ["python3.8", "/face_recognizer_root/face_encoder/main.py"]