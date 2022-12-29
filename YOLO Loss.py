import torch
from torch import nn
from torch.nn import functional as F
from Utils import intersection_over_union


class YoloLoss(nn.Module):
    def __init__(self, S=7, B=2, C=3, device='cpu'):
        """
        :param: S * S - количество ячеек на которые разбивается изображение
        :param: B - количество предсказанных прямоугольников в каждой ячейке
        :param: C - количество классов
        """

        super().__init__()
        self.mse = nn.MSELoss(reduction="sum")

        self.S = S
        self.B = B
        self.C = C

        self.lambda_noobj = 0.5
        self.lambda_coord = 5

        self.device = device

    def forward(self, predictions, target):
        """ YOLO LOSS
            После батчевальни predictions и target будут иметь следующие размеры
        Args:
            predictions: (Tensor) размер [n_batch, S, S, Bx5+C], 5=len([x, y, w, h, conf]).
            target: (Tensor) размер [n_batch, S, S, Bx5+C].
        Returns:
            (Tensor): loss, sized [1, ].
        """

        batch_size = predictions.shape[0]

        mask_obj = target[:, :, :, 4] > 0  # Маска ячеек, где есть объекты [n_batch, S, S]
        mask_noobj = target[:, :, :, 4] == 0  # Маска ячеек, где объектов нет [n_batch, S, S]

        # Распространяем маску на все предикты [n_batch, S, S, Bx5+C]
        mask_obj = mask_obj.unsqueeze(-1).expand_as(target)

        # Распространяем маску на все предикты [n_batch, S, S, Bx5+C]
        mask_noobj = mask_noobj.unsqueeze(-1).expand_as(target)

        # Берем только предикты, где у нас есть объекты. [n_coord, Bx5+C]
        predict_obj = predictions[mask_obj].view(-1, self.B * 5 + self.C)

        # Собираем координаты и уверенность ВСЕХ боксов [n_coord x B, 5=len([x, y, w, h, conf])]
        # n_coord: кол-во ячеек, где есть объекты.
        bbox_pred = predict_obj[:, :5 * self.B].contiguous().view(-1, 5)
        class_pred = predict_obj[:, 5 * self.B:]  # Собираем распределение вероятности по классам [n_coord, C]

        # Делаем тоже самое, но для таргета
        target_obj = target[mask_obj].view(-1, self.B * 5 + self.C)  # [n_coord, Bx5+C]

        bbox_target = target_obj[:, :5 * self.B].contiguous().view(-1, 5)  # [n_coord x B, 5=len([x, y, w, h, conf])]
        class_target = target_obj[:, 5 * self.B:]  # [n_coord, C]

        noobj_pred = predictions[mask_noobj].view(-1, self.B * 5 + self.C)  # Предикты, где нет объектов. [n_noobj, N]
        # n_noobj: кол-во ячеек, где нет объектов.
        noobj_target = target[mask_noobj].view(-1, self.B * 5 + self.C)  # Таргеты, где нет объектов. [n_noobj, N]
        # n_noobj: кол-во ячеек, где нет объектов.
        noobj_conf_mask = torch.zeros(noobj_pred.size(), device=self.device,
                                      dtype=torch.bool)  # Вводим тензор маски, который поможет достать значение
                                                         # уверенностей [n_noobj, N]

        for b in range(self.B):
            noobj_conf_mask[:, 4 + b * 5] = 1  # noobj_conf_mask[:, 4] = 1; noobj_conf_mask[:, 9] = 1

        noobj_pred_conf = noobj_pred[noobj_conf_mask]  # [n_noobj, 2=len([conf1, conf2])]
        noobj_target_conf = noobj_target[noobj_conf_mask]  # [n_noobj, 2=len([conf1, conf2])]

        loss_noobj = F.mse_loss(noobj_pred_conf, noobj_target_conf,
                                reduction='sum')  # Лосс уверенности, где нет объектов

        response_mask_obj = torch.zeros(bbox_target.size(), device=self.device, dtype=torch.bool)  # [n_coord x B, 5]
        bbox_target_iou = torch.zeros(bbox_target.size(), device=self.device)  # [n_coord x B, 5]

        # Мы должны выбрать из B боксов тот, у которого будет наивысший IoU с таргетом
        for i in range(0, bbox_target.size(0), self.B):
            pred = bbox_pred[i:i + self.B]  # Боксы в i-той ячейке, [B, 5=len([x, y, w, h, conf])]
            pred_xyxy = torch.FloatTensor(pred.size())  # Вспомогательный тензор для переброски из
            # [B, 5=len([x, y, w, h, conf])] в [B, 5=len([x1, y1, x2, y2, conf])]

            # Так как (center_x,center_y)=pred[:, 2] и (w,h)=pred[:,2:4] нормализованны по размеру ячейки и изображения соответственно,
            # Перемасштабируем (center_x, center_y) под размер картинки, чтобы посчитать IoU корректно.
            # На самом деле это перемасштабирование не выдает настоящие координаты ячейки, так как должно быть по идее смещнение на её номер в сетке, но нас интересует IoU, которому это и не надо будет
            pred_xyxy[:, :2] = pred[:, :2] / float(self.S) - 0.5 * pred[:, 2:4]
            pred_xyxy[:, 2:4] = pred[:, :2] / float(self.S) + 0.5 * pred[:, 2:4]

            target = bbox_target[i].view(-1, 5)  # Так же достаем бокс с тагргета, [1, 5=len([x, y, w, h, conf])]
            target_xyxy = torch.FloatTensor(target.size())  # Вспомогательный тензор для переброски из
            # [B, 5=len([x, y, w, h, conf])] в [B, 5=len([x1, y1, x2, y2, conf])]

            # Так как (center_x,center_y)=pred[:, 2] и (w,h)=pred[:,2:4] нормализованны по размеру ячейки и изображения соответственно,
            # Перемасштабируем (center_x, center_y) под размер картинки, чтобы посчитать IoU корректно.
            target_xyxy[:, :2] = target[:, :2] / float(self.S) - 0.5 * target[:, 2:4]
            target_xyxy[:, 2:4] = target[:, :2] / float(self.S) + 0.5 * target[:, 2:4]

            iou = torch.Tensor(
                [intersection_over_union(pred_xyxy[i, :4], target_xyxy[0, :4]) for i in range(self.B)])  # [B, 1]

            max_iou, max_index = iou.max(0)
            max_index = max_index.data.to(self.device)

            response_mask_obj[i + max_index] = 1

            # "We want the confidence score to equal the intersection over union (IOU) between the predicted box and the ground truth"
            # Это по словам авторов статьи про YOLO.
            bbox_target_iou[i + max_index] = (max_iou).data.to(self.device)
        bbox_target_iou = bbox_target_iou.to(self.device)

        # Выбираем боксы, где у нас есть предикт с наибольшим IoU
        bbox_pred_response = bbox_pred[response_mask_obj].view(-1, 5)  # [n_response, 5]
        bbox_target_response = bbox_target[response_mask_obj].view(-1, 5)  # [n_response, 5], только первые 4=(x, y, w, h) координаты используются
        target_iou = bbox_target_iou[response_mask_obj].view(-1, 5)  # [n_response, 5], только уверенность используется (последний столбец)
        loss_xy = F.mse_loss(bbox_pred_response[:, :2], bbox_target_response[:, :2], reduction='sum')
        # в статье написано, что лучше брать лосс от корней ширины и высоты, чтобы на небольшие изменения (
        # perturbations) больше снижало у мелких боксов, чем у больших
        loss_wh = F.mse_loss(torch.sqrt(bbox_pred_response[:, 2:4]), torch.sqrt(bbox_target_response[:, 2:4]),
                             reduction='sum')

        loss_obj = F.mse_loss(bbox_pred_response[:, 4], target_iou[:, 4], reduction='sum')

        loss_class = F.mse_loss(class_pred, class_target, reduction='sum')

        # Итоговый лосс
        loss = self.lambda_coord * (loss_xy + loss_wh) + loss_obj + self.lambda_noobj * loss_noobj + loss_class
        loss = loss / float(batch_size)

        return loss
