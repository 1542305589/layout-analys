import cv2
import numpy as np
import torch
import torchvision

classes = {
    0: "Text",
    1: "Figure",
    2: "Table",
    3: "Equation"
}

colors = {
    0: (0, 255, 0),     # 绿色
    1: (255, 0, 0),     # 蓝色
    2: (0, 0, 255),     # 红色
    3: (51, 255, 255)   # 黄色
}

def resize_image(image, size, letterbox_image):
    """
        对输入图像进行resize
    Args:
        size:目标尺寸
        letterbox_image: bool 是否进行letterbox变换
    Returns:指定尺寸的图像
    """
    ih, iw, _ = image.shape
    # print(ih, iw)
    h, w = size
    # letterbox_image = False
    if letterbox_image:
        scale = min(w/iw, h/ih)
        nw = int(iw*scale)
        nh = int(ih*scale)
        image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
        # cv2.imshow("img", img)
        # cv2.waitKey()
        # print(image.shape)
        # 生成画布
        image_back = np.ones((h, w, 3), dtype=np.uint8) * 128
        # 将image放在画布中心区域-letterbox
        image_back[(h-nh)//2: (h-nh)//2 + nh, (w-nw)//2:(w-nw)//2+nw , :] = image
    else:
        image_back = cv2.resize(image, (640, 640))
        # cv2.imshow("img", image_back)
        # cv2.waitKey()
    return image_back


def img2input(img):
    img = np.transpose(img, (2, 0, 1))
    img = img/255
    return np.expand_dims(img, axis=0).astype(np.float32)

def std_output(pred):
    """
    将（1，84，8400）处理成（8400， 85）  85= box:4  conf:1 cls:80
    """
    pred = np.squeeze(pred)
    pred = np.transpose(pred, (1, 0))
    # pred_class = pred[..., 4:]
    # pred_conf = np.max(pred_class, axis=-1)
    # pred = np.insert(pred, 4, pred_conf, axis=-1)
    return pred

def filter(preds):
    boxes = []
    for pred in preds:
        # if (pred[2] * pred[3]) / (640 * 640) < 0.01:
        #     continue
        for conf in pred[4:]:
            if conf >= 0.25:
                boxes.append(pred.tolist())
                break

    return torch.tensor(boxes)

def xywh2xyxy(boxes):
    for box in boxes:
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        x1 = x - w / 2
        x2 = x + w / 2
        y1 = y - h / 2
        y2 = y + h / 2
        box[0] = x1
        box[1] = y1
        box[2] = x2
        box[3] = y2

    return boxes[:, :4], torch.max(boxes[:, 4:], dim=-1)


def nms(boxes, scores, cls, iou):
    new_boxes = []
    indexes = torchvision.ops.nms(boxes, scores, iou_threshold=iou).tolist()
    for index in indexes:
        new_boxes.append([boxes[index], scores[index], cls[index]])

    return new_boxes

def draw(img_path, boxes):
    img = cv2.imread(img_path)
    h, w, _ = img.shape
    font = cv2.FONT_HERSHEY_SIMPLEX  # 简单字体
    for box in boxes:
        x1, y1, x2, y2 = box[0]
        x1 = int(x1 / 640 * w)
        x2 = int(x2 / 640 * w)
        y1 = int(y1 / 640 * h)
        y2 = int(y2 / 640 * h)
        score = box[1].item()
        cls = box[2].item()
        cv2.rectangle(img, (x1, y1), (x2, y2), colors[cls], 2)
        label = f"{classes[cls]} {score:.2f}"
        text_size, baseline = cv2.getTextSize(label, font, 1.5, 2)

        text_x = x1
        text_y = y1 - baseline

        cv2.rectangle(img, (text_x, text_y - text_size[1]), (text_x + text_size[0], text_y + baseline), colors[cls], -1)
        cv2.putText(img, label, (text_x, text_y), font, 1.5, (0, 0, 0), 2)

    return img

def draw2(img_path , boxes):
    img = cv2.imread(img_path)
    font = cv2.FONT_HERSHEY_SIMPLEX  # 简单字体
    for box in boxes:
        x1, y1, x2, y2 = box[0]
        x1 = int(x1)
        x2 = int(x2)
        y1 = int(y1)
        y2 = int(y2)
        score = box[1].item()
        cls = box[2].item()
        cv2.rectangle(img, (x1, y1), (x2, y2), colors[cls], 2)
        label = f"{classes[cls]} {score:.2f}"
        text_size, baseline = cv2.getTextSize(label, font, 1.5, 2)

        text_x = x1
        text_y = y1 - baseline

        cv2.rectangle(img, (text_x, text_y - text_size[1]), (text_x + text_size[0], text_y + baseline), colors[cls], -1)
        cv2.putText(img, label, (text_x, text_y), font, 1.5, (0, 0, 0), 2)

    return img







