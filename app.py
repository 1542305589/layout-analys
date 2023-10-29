import cv2
from PIL import Image, ImageDraw, ImageFont
import gradio as gr
from ultralytics import YOLO
from utils import nms, draw2

# 加载模型
model = YOLO("model/best.pt").cuda()


def sepia(input_img):
    img_path = input_img
    input_img = cv2.imread(input_img)
    result = model.predict(source=input_img)
    new_boxes = nms(result[0].boxes.xyxy, result[0].boxes.conf, result[0].boxes.cls, iou=0.45)
    output_image = draw2(img_path, new_boxes)
    # output_image = result[0].plot()

    return output_image


demo = gr.Interface(sepia, gr.Image(type="filepath"), "image")
demo.launch()