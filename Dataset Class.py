import xmltodict
import torch
import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset
from typing import List


class2tag = {"apple": 1, "orange": 2, "banana": 3}


class FruitDataset(Dataset):
    def __init__(self, data_dir, transforms=None):
        self.image_paths = [data_dir + '/' + f for f in sorted(os.listdir(data_dir)) if f.split('.')[1] == 'jpg']
        self.box_paths = [data_dir + '/' + f for f in sorted(os.listdir(data_dir)) if f.split('.')[1] == 'xml']

        assert len(self.image_paths) == len(self.box_paths)

        self.transforms = transforms

    # Координаты прямоугольников советуем вернуть именно в формате (x_center, y_center, width, height)
    def __getitem__(self, idx):
        # image = (np.array(Image.open(self.image_paths[idx]).convert("RGB")) / 255 - 0.5) * 2
        image = np.array(Image.open(self.image_paths[idx]).convert("RGB"))
        boxes, class_labels = self.__get_boxes_from_xml(self.box_paths[idx], image.shape[1], image.shape[0])
        if self.transforms:
            transformed = self.transforms(image=image, bboxes=boxes, class_labels=class_labels)
            image = transformed['image']
            boxes = transformed['bboxes']
            class_labels = transformed['class_labels']

        else:
            image = torch.Tensor(image)
            boxes = torch.Tensor(boxes)

        return image, (boxes, class_labels)

    def __len__(self):
        return len(self.image_paths)

    def __get_boxes_from_xml(self, xml_filename: str, width, height):
        """
          Метод, который считает и распарсит (с помощью xmltodict) переданный xml
          файл и вернет координаты прямоугольников обьектов на соответсвующей фотографии
          и название класса обьекта в каждом прямоугольнике

          Обратите внимание, что обьектов может быть как несколько, так и один единственный
        """
        boxes = []
        class_labels = []

        with open(xml_filename) as f:
            dict_of_bbox = xmltodict.parse(f.read())

        if type(dict_of_bbox['annotation']['object']) is list:
            for obj in dict_of_bbox['annotation']['object']:
                box = [int(num) for num in list(obj['bndbox'].values())]
                yolo_box = self.__convert_to_yolo_box_params(box, width, height)
                boxes += [yolo_box]
                class_labels += [class2tag[obj['name']]]

        else:
            box = [int(num) for num in list(dict_of_bbox['annotation']['object']['bndbox'].values())]
            yolo_box = self.__convert_to_yolo_box_params(box, width, height)
            boxes += [yolo_box]
            class_labels += [class2tag[dict_of_bbox['annotation']['object']['name']]]

        return boxes, class_labels

    def __convert_to_yolo_box_params(self, box_coordinates: List[int], im_w, im_h):
        """
          Перейти от [xmin, ymin, xmax, ymax] к [x_center, y_center, width, height].

          Обратите внимание, что параметры [x_center, y_center, width, height] - это
          относительные значение в отрезке [0, 1]

          :param: box_coordinates - координаты коробки в формате [xmin, ymin, xmax, ymax]
          :param: im_w - ширина исходного изображения
          :param: im_h - высота исходного изображения

          :return: координаты коробки в формате [x_center, y_center, width, height]
        """

        ans = []

        ans.append((box_coordinates[0] + box_coordinates[2]) / 2 / im_w)  # x_center
        ans.append((box_coordinates[1] + box_coordinates[3]) / 2 / im_h)  # y_center

        ans.append((box_coordinates[2] - box_coordinates[0]) / im_w)  # width
        ans.append((box_coordinates[3] - box_coordinates[1]) / im_h)  # height
        return ans
