import torch
import math
from collections import Counter

grid_size = 7
examples_per_cell = 2
nums_of_classes = 3


def batch_collate_fn(batch):
    images = [item[0].unsqueeze(0) for item in batch]
    detections = []
    for item in batch:
        det = item[1]
        det = zip(det[0], det[1])
        image_detections = torch.zeros(1, grid_size, grid_size, 5 * examples_per_cell + nums_of_classes)
        for box, cls in det:
            gx = math.floor(grid_size * box[0])
            gy = math.floor(grid_size * box[1])
            for i in range(0, 5 * examples_per_cell, 5):
                image_detections[0, gx, gy, i:i + 4] = torch.tensor(
                    [grid_size * box[0] - gx, grid_size * box[1] - gy] + list(box[2:4]))
                image_detections[0, gx, gy, i + 4] = 1
            image_detections[0, gx, gy, 5 * examples_per_cell + cls - 1] = 1
        detections.append(image_detections)

    images = torch.cat(images, 0)
    detections = torch.cat(detections, 0)
    return images, detections


def get_true_boxes(loader, grid_size=7, examples_per_cell=2, nums_of_classes=3):
    """
        Args:
            loader: (Tensor) размер [N, n_batches, S, S, 5xB+C], 5=len([x, y, w, h, conf]).
            iou_threshold: (float).
            threshold: (float)
        Returns:
            (List): bboxes_from_batch, sized [n_boxes_i, 9] 9=(len([batch_num, x_min, y_min, x_max, y_max, conf,
             prob_class1, prob_class2, prob_class3])).
            :param loader:
            :param nums_of_classes:
            :param examples_per_cell:
            :param grid_size:
    """
    all_true_boxes = torch.tensor([])

    train_idx = 0

    for batch_idx, (x, true_boxes) in enumerate(loader):
        true_boxes = true_boxes  # [S, S, 5xB+C]

        batch_size = x.shape[0]

        # Приведем true_boxes к виду, где в каждой строке батча таргет бокс
        start_of_last_box_idx = 5 * (examples_per_cell - 1)
        new_true_boxes = torch.zeros(
            (true_boxes.shape[0], true_boxes.shape[1], true_boxes.shape[2], 5 + nums_of_classes))
        for i in range(grid_size):
            for j in range(grid_size):
                new_true_boxes[:, i, j, 0] = (true_boxes[:, i, j, 0] + i) / grid_size - 0.5 * true_boxes[:, i, j,
                                                                                              2]  # x_min
                new_true_boxes[:, i, j, 1] = (true_boxes[:, i, j, 1] + j) / grid_size - 0.5 * true_boxes[:, i, j,
                                                                                              3]  # y_min
                new_true_boxes[:, i, j, 2] = (true_boxes[:, i, j, 0] + i) / grid_size + 0.5 * true_boxes[:, i, j,
                                                                                              2]  # x_max
                new_true_boxes[:, i, j, 3] = (true_boxes[:, i, j, 1] + j) / grid_size + 0.5 * true_boxes[:, i, j,
                                                                                              3]  # y_max
                new_true_boxes[:, i, j, 4:] = true_boxes[:, i, j, -(1 + nums_of_classes):]  # conf, prob_class1,
                # prob_class2,....

        new_true_boxes = new_true_boxes.view(new_true_boxes.shape[0], -1, 5 + nums_of_classes)  # [n_batch, S*S, 5+C]
        true_boxes = new_true_boxes

        for idx in range(batch_size):
            # убираем нулевые боксы
            mask_true_bbox = true_boxes[idx, :, 4] > 0  # [S*S]
            mask_true_bbox = mask_true_bbox.unsqueeze(-1).expand_as(true_boxes[idx])  # [S*S, 5+C]

            batch_true_boxes = true_boxes[idx][mask_true_bbox].view(-1, 5 + nums_of_classes)  # Берем только таргеты,
            # где у нас есть объекты. [n_bboxes, 5+C]
            batch_true_boxes = torch.cat((torch.tensor([[train_idx]] * batch_true_boxes.shape[0]), batch_true_boxes),
                                         dim=1)
            all_true_boxes = torch.cat((all_true_boxes, batch_true_boxes), dim=0)
            train_idx += 1

    return all_true_boxes


def intersection_over_union(predicted_bbox, gt_bbox) -> float:
    """
    Intersection Over Union для двух прямоугольников

    :param: dt_bbox - [x_min, y_min, x_max, y_max]
    :param: gt_bbox - [x_min, y_min, x_max, y_max]

    :return: Intersection Over Union
    """
    intersection_bbox = torch.Tensor(
        [
            max(predicted_bbox[0], gt_bbox[0]),
            max(predicted_bbox[1], gt_bbox[1]),
            min(predicted_bbox[2], gt_bbox[2]),
            min(predicted_bbox[3], gt_bbox[3]),
        ]
    )

    intersection_area = max(intersection_bbox[2] - intersection_bbox[0], 0) * max(
        intersection_bbox[3] - intersection_bbox[1], 0
    )
    area_dt = (predicted_bbox[2] - predicted_bbox[0]) * (predicted_bbox[3] - predicted_bbox[1])
    area_gt = (gt_bbox[2] - gt_bbox[0]) * (gt_bbox[3] - gt_bbox[1])

    union_area = area_dt + area_gt - intersection_area

    iou = intersection_area / union_area
    return iou


def non_max_suppression(bboxes, iou_threshold, threshold):
    """
        Args:
            bboxes: (Tensor) размер [n_bboxes, 5+C], 5=len([x_min, y_min, x_max, y_max, conf]).
            iou_threshold: (float).
            threshold: (float)
        Returns:
            (List): bboxes_from_batch, sized [n_boxes_i, 9] 9=(len([batch_num, xmin, ymin, xmax, ymax, conf, prob_class1, prob_class2, prob_class3])).
        """
    bboxes_sorted_by_conf = bboxes[
        bboxes[:, 4].sort(dim=0, descending=True)[1].data]  # батчи отсортированы независимо по конфеденсу
    new_bboxes = bboxes_sorted_by_conf[bboxes_sorted_by_conf[:, 4] > threshold]
    i = 0
    while i < len(new_bboxes):
        curr_bbox = new_bboxes[i]
        label = curr_bbox[5:].argmax(dim=0)
        j = i + 1
        while j < len(new_bboxes):
            if label == new_bboxes[j, 5:].argmax(dim=0):
                iou = intersection_over_union(curr_bbox[:4], new_bboxes[j, :4])
                if iou > iou_threshold:
                    new_bboxes = torch.cat((new_bboxes[:j], new_bboxes[j + 1:]))
                else:
                    j += 1
            else:
                j += 1
        i += 1

    return new_bboxes


def mean_average_precision(pred_boxes, true_boxes, iou_threshold=0.5, grid_size=7, examples_per_cell=2,
                           nums_of_classes=3, device='cpu', epsilon=1e-6):
    """
    Parameters: pred_boxes (list): содержит боксы вида [n_bboxes, 6+С], 6+С=(len([train_num, x_min, y_min, x_max,
    y_max, conf, prob_class1, prob_class2, prob_class3, ....]
    true_boxes (list): боксы вида [n_bboxes, 6+С],
    iou_threshold (float): трешхолд для классификации Returns: float: значение mAP для определенного  iou_threshold
    """
    average_precisions = []

    for c in range(nums_of_classes):
        detections = []
        ground_truths = []

        # идем через все боксы и берем только нужного класса
        for detection in pred_boxes:
            if detection[6:].argmax(dim=0) == c:
                detections.append(detection)

        for true_box in true_boxes:
            if true_box[6:].argmax(dim=0) == c:
                ground_truths.append(true_box)

        # Тут мы вначале будем хранить кол-во таргет боксов на данной картинке
        amount_bboxes = Counter([gt[0].int().item() for gt in ground_truths])

        # Здесь мы его чуть переобозначим и добавим на каждое изображение столько нулей сколько таргет боксов на этом изображении
        amount_bboxes1 = {}
        for key, val in amount_bboxes.items():
            amount_bboxes1[key] = torch.zeros(val)
        amount_bboxes = amount_bboxes1

        # Отсортируем по уверенности
        detections.sort(key=lambda x: x[5], reverse=True)
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)

        # Если таргет боксов данного класса нет, то скипаем
        if total_true_bboxes == 0:
            continue

        for detection_idx, detection in enumerate(detections):
            # Берем таргеты с одного изображения
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ]

            num_gts = len(ground_truth_img)
            best_iou = 0

            for idx, gt in enumerate(ground_truth_img):
                iou = intersection_over_union(
                    torch.tensor(detection[1:5]),
                    torch.tensor(gt[1:5])
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                # Так как у нас боксы отсортированы уже, то если к этому таргету уже нашли хороший бокс, то этот отлетает в FP
                if amount_bboxes[detection[0].int().item()][best_gt_idx] == 0:
                    # Добавляем в FP и пишем, что этот таргет бокс уже забит
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0].int().item()][best_gt_idx] = 1  # забит
                else:
                    FP[detection_idx] = 1

            # Тут он отлетает в FP по маленькому IoU
            else:
                FP[detection_idx] = 1

        TP_cumsum = torch.cumsum(TP, dim=0)  # на i-ой позиции сумма подпоследовательности [0,i]
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)  # очевидно
        precisions = torch.divide(TP_cumsum, (TP_cumsum + FP_cumsum + epsilon))  # очевидно по формуле
        precisions = torch.cat((torch.tensor([1]), precisions))  # добавляем изначальная точка для интегрирования
        recalls = torch.cat((torch.tensor([0]), recalls))
        # torch.trapz методом трапеций считает площадь под графиком
        average_precisions.append(torch.trapz(precisions, recalls))

    return sum(average_precisions) / len(average_precisions)


def get_bound_boxes(loader, model, iou_threshold=.5, threshold=.4, grid_size=7, examples_per_cell=2, nums_of_classes=3,
                    device='cpu'):
    """
        Args:
            loader: (Tensor) размер [N, n_batches, S, S, 5xB+C], 5=len([x, y, w, h, conf]).
            iou_threshold: (float).
            threshold: (float)
        Returns:
            (List): bboxes_from_batch, sized [n_boxes_i, 9] 9=(len([batch_num, xmin, ymin, xmax, ymax, conf, prob_class1, prob_class2, prob_class3])).
    """

    all_pred_boxes = torch.tensor([], device=device)
    all_true_boxes = torch.tensor([], device=device)

    model.eval()
    train_idx = 0

    for batch_idx, (x, true_boxes) in enumerate(loader):
        x = x.to(device)
        true_boxes = true_boxes.to(device)  # [S, S, 5xB+C]

        with torch.no_grad():
            bboxes = model(x.float())

        batch_size = x.shape[0]

        # Приведем true_boxes к виду, где в каждой строке батча таргет бокс
        start_of_last_box_idx = 5 * (examples_per_cell - 1)
        new_true_boxes = torch.zeros(
            (true_boxes.shape[0], true_boxes.shape[1], true_boxes.shape[2], 5 + nums_of_classes), device=device)
        for i in range(grid_size):
            for j in range(grid_size):
                new_true_boxes[:, i, j, 0] = (true_boxes[:, i, j, 0] + i) / grid_size - 0.5 * true_boxes[:, i, j, 2]  # x_min
                new_true_boxes[:, i, j, 1] = (true_boxes[:, i, j, 1] + j) / grid_size - 0.5 * true_boxes[:, i, j, 3]  # y_min
                new_true_boxes[:, i, j, 2] = (true_boxes[:, i, j, 0] + i) / grid_size + 0.5 * true_boxes[:, i, j, 2]  # x_max
                new_true_boxes[:, i, j, 3] = (true_boxes[:, i, j, 1] + j) / grid_size + 0.5 * true_boxes[:, i, j, 3]  # y_max
                new_true_boxes[:, i, j, 4:] = true_boxes[:, i, j, -(1 + nums_of_classes):]  # conf, probclass1, probclass2,....

        new_true_boxes = new_true_boxes.view(new_true_boxes.shape[0], -1, 5 + nums_of_classes)  # [n_batch, S*S, 5+C]
        true_boxes = new_true_boxes

        # Приведем bboxes к виду, где в каждой строке батча бокс
        bboxes_coord = bboxes[:, :, :, :-nums_of_classes]
        classes = bboxes[:, :, :, -nums_of_classes:]

        # [n_batch, S, S, (xmin, ymin, xmax, ymax, conf)]
        bboxes_xy = torch.zeros(bboxes_coord.shape, device=device)
        for i in range(grid_size):
            for j in range(grid_size):
                for ex in range(examples_per_cell):
                    bboxes_xy[:, i, j, 0 + ex * 5] = (bboxes_coord[:, i, j, 0 + ex * 5] + i) / grid_size - 0.5 * bboxes_coord[:, i, j, 2 + ex * 5]  # x_min
                    bboxes_xy[:, i, j, 1 + ex * 5] = (bboxes_coord[:, i, j, 1 + ex * 5] + j) / grid_size - 0.5 * bboxes_coord[:, i, j, 3 + ex * 5]  # y_min
                    bboxes_xy[:, i, j, 2 + ex * 5] = (bboxes_coord[:, i, j, 0 + ex * 5] + i) / grid_size + 0.5 * bboxes_coord[:, i, j, 2 + ex * 5]  # x_max
                    bboxes_xy[:, i, j, 3 + ex * 5] = (bboxes_coord[:, i, j, 1 + ex * 5] + j) / grid_size + 0.5 * bboxes_coord[:, i, j, 3 + ex * 5]  # y_max
                    bboxes_xy[:, i, j, 4 + ex * 5] = bboxes_coord[:, i, j, 4 + ex * 5]  # conf

        bboxes_coord = bboxes_xy.view(bboxes_coord.shape[0], -1,
                                      5)  # [n_batch, n_bboxes, 5], 5=(len([xmin, ymin, xmax, ymax, conf]))
        classes = classes.view(bboxes_coord.shape[0], -1,
                               nums_of_classes)  # [n_batch, n_bboxes // B, nums_of_classes], nums_of_classes=(len([prob_class1, prob_class2, prob_class3, ...]))

        new_classes = torch.zeros((bboxes_coord.shape[0], bboxes_coord.shape[1], classes.shape[2]), device=device)
        for i in range(0, bboxes_coord.shape[1], examples_per_cell):
            new_classes[:, i, :] = classes[:, i // examples_per_cell, :]
            new_classes[:, i + 1, :] = classes[:, i // examples_per_cell, :]

        bboxes = torch.cat((bboxes_coord, new_classes), -1)  # [n_batch, n_bboxes, 5 + nums_of_classes]

        for idx in range(batch_size):
            nms_boxes = non_max_suppression(
                bboxes[idx],
                iou_threshold=iou_threshold,
                threshold=threshold
            )

            if nms_boxes.any():
                nms_boxes = torch.cat((torch.tensor([[train_idx]] * nms_boxes.shape[0], device=device), nms_boxes),
                                      dim=1)
                all_pred_boxes = torch.cat((all_pred_boxes, nms_boxes), dim=0)

            # убираем нулевые боксы
            mask_true_bbox = true_boxes[idx, :, 4] > 0  # [S*S]
            mask_true_bbox = mask_true_bbox.unsqueeze(-1).expand_as(true_boxes[idx])  # [S*S, 5+C]

            # Берем только таргеты, где у нас есть объекты. [n_bboxes, 5+C]
            batch_true_boxes = true_boxes[idx][mask_true_bbox].view(-1, 5 + nums_of_classes)
            batch_true_boxes = torch.cat(
                (torch.tensor([[train_idx]] * batch_true_boxes.shape[0], device=device), batch_true_boxes), dim=1)
            all_true_boxes = torch.cat((all_true_boxes, batch_true_boxes), dim=0)

            train_idx += 1

    model.train()
    return all_pred_boxes, all_true_boxes
