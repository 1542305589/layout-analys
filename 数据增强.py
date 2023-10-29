import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import os
from tqdm import tqdm
import cv2
import numpy as np
from ultralytics import YOLO

transform = A.Compose(
    [
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.5),   # 随机改变图像的色彩
    A.RandomBrightnessContrast(p=0.5),   # 随机调整图像的亮度和对比度
    A.HueSaturationValue(p=0.5),  # 随机调整图像的色相、饱和度和值
    A.RandomGamma(p=0.5),  # 随机调整图像的伽马值，实现图像的明暗变化
    A.GaussianBlur(p=0.1),  # 对图像进行高斯模糊
    ],

    bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
    )

label_list = ['Text', 'Figure', 'Table', 'Equation']

images_path = r"VOCDATA/images"
labels_path = r"VOCDATA/labels"
num = 5

if __name__ == "__main__":
    for Dir in os.listdir(images_path):
        pbar = tqdm(os.listdir(images_path+'/'+Dir))
        for img_name in pbar:
            image = cv2.imread(images_path+'/'+Dir+'/'+img_name)
            txt = labels_path + '/' + Dir + '/' + img_name[:-4] +'.txt'
            bboxes = []
            labels = []
            with open(txt, "r") as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip().split(" ")
                    for j in range(len(line)):
                        if j == 0:
                            line[j] = int(line[j])
                        else:
                            line[j] = float(line[j])

                    labels.append(line[0])
                    bboxes.append(line[1:])


            for i in range(num):
                try:
                    transformed = transform(image=image, bboxes=bboxes, class_labels=labels)
                except Exception as e:
                    continue
                transformed_image = transformed["image"]
                transformed_bboxes = transformed["bboxes"]
                transformed_class_labels = transformed["class_labels"]
                aug_image = cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR)
                # for data in transformed_bboxes:
                #
                #     xcenter = int(data[0] * aug_image.shape[1])
                #     ycenter = int(data[1] * aug_image.shape[0])
                #     w = int(data[2] * aug_image.shape[1])
                #     h = int(data[3] * aug_image.shape[0])
                #
                #     xmin = int(xcenter - w / 2)
                #     ymin = int(ycenter - h / 2)
                #     xmax = int(xcenter + w / 2)
                #     ymax = int(ycenter + h / 2)
                #
                #     pt1 = (xmin, ymin)
                #     pt2 = (xmax, ymax)
                #
                #     # 绘制矩形框
                #     cv2.rectangle(aug_image, pt1, pt2, (0, 255, 0), 2)
                #
                # # 显示绘制后的图像
                # cv2.imshow("Image", aug_image)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                cv2.imwrite(images_path+'/'+Dir+'/'+f"aug{i}_"+img_name, aug_image)
                with open(labels_path + '/' + Dir + '/' + f"aug{i}_" + img_name[:-4] +'.txt', "w") as file:
                    for ii in range(len(transformed_class_labels)):
                        file.write(f"{transformed_class_labels[ii]} {transformed_bboxes[ii][0]:.6f} {transformed_bboxes[ii][1]:.6f} {transformed_bboxes[ii][2]:.6f} {transformed_bboxes[ii][3]:.6f}" + '\n')
                file.close()



