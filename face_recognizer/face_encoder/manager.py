import json
import os
import shutil
import tempfile
from typing import List, Tuple

import cv2
import numpy as np
import requests
from facenet_pytorch.models.inception_resnet_v1 import get_torch_home

from face_encoder.exceptions.base_exception import ServiceBaseException
from face_encoder.exceptions.s3_exception import S3Exception
from face_encoder.face_recognizer.face_recognizer_facenet import FaceRecognizerFacenet
from face_encoder.face_recognizer.face_recognizer_octuplet_loss import FaceRecognizerOctupletLoss
from face_encoder.exceptions.bad_file_format_exception import BadFileFormatException
from face_encoder.external_services.s3_client import S3Client
from face_encoder.logger import logger


def get_face_index_port() -> str:
    return os.getenv("FACE_INDEX_PORT", "4445")


def get_face_index_host() -> str:
    return os.getenv("FACE_INDEX_HOST", "localhost")


class Manager:
    def __init__(self) -> None:
        self.image_extensions = (
            'png', 'jpg', 'jpeg', 'tiff', 'bmp', 'gif', 'ppm', 'pnm', 'pgm', 'pbm', 'webp',
            'pcx', 'eps', 'sgi', 'hdr', 'pic', 'sr', 'ras', 'dib', 'jpe', 'jfif', 'j2k'
        )
        self.logger = logger
        self.s3_client = S3Client(self.logger)
        self.face_recognizer = FaceRecognizerOctupletLoss()

    def __get_image(self, path: str) -> np.ndarray:
        if not path.endswith(tuple(self.image_extensions)):
            raise BadFileFormatException(f"Unsupported input format: {splitext_(path)[1]}")  # noqa

        # TODO: some formats are not OpenCV readable
        image_rgb = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        if image_rgb is None:
            raise BadFileFormatException(f"seems file {os.path.basename(path)} not an image")

        return image_rgb

    def encode_faces_from_s3_image(self, s3_path: str) -> Tuple[List[np.ndarray], List[List]]:
        with tempfile.TemporaryDirectory() as tmp_dir:
            try:
                image_path = self.s3_client.download(s3_path=s3_path, output_dir=tmp_dir)
                self.logger.info(f"get image {image_path} from s3-storage")
            except Exception as ex:
                raise S3Exception(f"File {s3_path} not found in s3-storage. Error's msg: {ex}")

            return self.__encode_faces(image_path)

    def __encode_faces(self, image_path: str) -> Tuple[List[np.ndarray], List[List]]:
        image = self.__get_image(image_path)

        try:
            embedding, face_bbs = self.face_recognizer.get_image_embeddings(image)
            self.logger.info("faces have recognized from image")
            return embedding, face_bbs
        except Exception as exception:
            self.logger.error(f"Error in face encoder: {exception}" , exc_info=True)
            raise exception

    def insert_embeddings_to_db(self, uuid: str, embeddings: List[np.ndarray], expected_code: int = 200):
        port = get_face_index_port()
        host = get_face_index_host()

        data = {
            "uuid": uuid,
            "embeddings": embeddings
        }

        r = requests.post(f"http://{host}:{port}/insert_embeddings_to_db", json=data)

        if expected_code != 200:
            raise ServiceBaseException(msg=f"Error from face_index service: {r.content.decode()}")
        else:
            self.logger.info(f"Faces saved into database (with uuid={uuid})")
            return json.loads(r.content.decode())

    def find_similar_images(self, embeddings: List[np.ndarray],
                            topn: int, threshold: float,
                            expected_code: int = 200,
                            uuid: str = None) -> dict:

        port = get_face_index_port()
        host = get_face_index_host()

        data = {
            "embeddings": embeddings,
            "threshold": threshold,
            "topn": topn,
            "uuid": uuid
        }

        r = requests.post(f"http://{host}:{port}/find_similar_images", json=data)
        if expected_code != 200:
            raise ServiceBaseException(msg=f"Error from face_index service: {r.content.decode()}")
        else:
            return json.loads(r.content.decode())

    def get_embeddings_by_uuid(self, uuid: str = None, expected_code: int = 200):
        port = get_face_index_port()
        host = get_face_index_host()

        data = {
            "uuid": uuid
        }

        r = requests.post(f"http://{host}:{port}/get_embeddings_by_uuid", json=data)
        if expected_code != 200:
            raise ServiceBaseException(msg=f"Error from face_index service: {r.content.decode()}")
        else:
            return json.loads(r.content.decode())

    def init_index(self):
        port = get_face_index_port()
        host = get_face_index_host()
        r = requests.post(f"http://{host}:{port}/init_index")
        if r.status_code != 200:
            self.logger.error(f"I has got from api-method /init_index code={r.status_code}")
