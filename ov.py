import openvino.runtime as ov
import numpy as np
import time
import cv2
from utils import resize_image, img2input, std_output, filter, xywh2xyxy, nms, draw
import gradio as gr
import os

classes = {
    0: "Text",
    1: "Figure",
    2: "Table",
    3: "Equation"
}


core = ov.Core()
compiled_model = core.compile_model(r"model/ov/best.xml", "CPU")
infer_request = compiled_model.create_infer_request()


def sepia(input_img):
    img = cv2.imread(input_img)
    print(f"文件名:{input_img} 分辨率:{img.shape}")
    img = resize_image(img, (640, 640), False)
    img = img2input(img)

    input_tensor = ov.Tensor(array=img, shared_memory=False)
    infer_request.set_input_tensor(input_tensor)

    t_model = time.perf_counter()
    infer_request.start_async()
    infer_request.wait()
    print(f'预测时间:{(time.perf_counter() - t_model)*1000:.4f}ms')

    output = infer_request.get_output_tensor()
    output_buffer = output.data

    output_buffer = std_output(output_buffer)

    boxes = filter(output_buffer)

    boxes, (scores, cls) = xywh2xyxy(boxes)

    new_boxes = nms(boxes, scores, cls, iou=0.45)

    img = draw(input_img, new_boxes)

    return img


demo = gr.Interface(sepia, gr.Image(type="filepath"), "image")
demo.launch()

# if __name__ == "__main__":
#     img_dir = "F:\\BaiduNetdiskDownload\\test2023\\"
#     for img in os.listdir(img_dir):
#         if os.path.isdir(img_dir + img):
#             continue
#         image = cv2.imread(img_dir+img)
#         print(f"文件名:{img} 分辨率:{image.shape}")
#         h, w, _ = image.shape
#         image = resize_image(image, (640, 640), False)
#         image = img2input(image)
#
#         input_tensor = ov.Tensor(array=image, shared_memory=False)
#         infer_request.set_input_tensor(input_tensor)
#
#         t_model = time.perf_counter()
#         infer_request.start_async()
#         infer_request.wait()
#         print(f'预测时间:{(time.perf_counter() - t_model)*1000:.4f}ms')
#
#         output = infer_request.get_output_tensor()
#         output_buffer = output.data
#
#         output_buffer = std_output(output_buffer)
#
#         boxes = filter(output_buffer)
#
#         boxes, (scores, cls) = xywh2xyxy(boxes)
#
#         new_boxes = nms(boxes, scores, cls, iou=0.45)
#
#         drawed_img = draw(img_dir+img, new_boxes)
#
#         cv2.imwrite(img_dir + "results\\" + img, drawed_img)
#
#         with open("result2.txt", 'a') as f:
#             for box, score, cls in new_boxes:
#                 cls = cls.item()
#                 x1 = int(box[0] * w / 640)
#                 y1 = int(box[1] * w / 640)
#                 x2 = int(box[2] * h / 640)
#                 y2 = int(box[3] * h / 640)
#                 f.write(f"{img} {classes[cls]} {x1} {y1} {x2} {y2}")
#                 f.write("\n")


