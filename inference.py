import torch
from torchvision import transforms
import cv2
import time
import numpy as np
import os
import matplotlib.pyplot as plt
from copy import deepcopy
from os.path import join

torch.cuda.set_device(9)

transform = transforms.Compose([                                 
	transforms.ToTensor(), 
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

out_channels = 384
image_size = 256
ad_object = "juice_bottle"
ad_type = "good" # good, structural_anomalies, logical_anomalies

train_output_dir = "./output/4/trainings/mvtec_loco/" + ad_object
teacher_quantiles = np.load(os.path.join(train_output_dir, "t_quantiles.npy"), allow_pickle=True).item()
teacher_mean = torch.tensor(teacher_quantiles['teacher_mean']).cuda()
teacher_std = torch.tensor(teacher_quantiles['teacher_std']).cuda()

map_quantiles = np.load(os.path.join(train_output_dir, "q_quantiles.npy"), allow_pickle=True).item()
q_st_start = torch.tensor(map_quantiles['q_st_start']).cuda()
q_st_end = torch.tensor(map_quantiles['q_st_end']).cuda()
q_ae_start = torch.tensor(map_quantiles['q_ae_start']).cuda()
q_ae_end = torch.tensor(map_quantiles['q_ae_end']).cuda()

pic_path = os.path.join(train_output_dir, "histogram.png")

def inference(score_list, time_list, ad_type="good"):

    def predict(image, teacher, student, autoencoder, teacher_mean, teacher_std,
                q_st_start=None, q_st_end=None, q_ae_start=None, q_ae_end=None):
        teacher_output = teacher(image)
        teacher_output = (teacher_output - teacher_mean) / teacher_std
        student_output = student(image)
        autoencoder_output = autoencoder(image)
        map_st = torch.mean((teacher_output - student_output[:, :out_channels])**2,
                            dim=1, keepdim=True)
        map_ae = torch.mean((autoencoder_output -
                            student_output[:, out_channels:])**2,
                            dim=1, keepdim=True)
        if q_st_start is not None:
            map_st = 0.1 * (map_st - q_st_start) / (q_st_end - q_st_start)
        if q_ae_start is not None:
            map_ae = 0.1 * (map_ae - q_ae_start) / (q_ae_end - q_ae_start)
        map_combined = 0.5 * map_st + 0.5 * map_ae
        return map_combined, map_st, map_ae

    # 获取文件夹下所有图片
    def getImages(folder_path):
        images = []
        for filename in os.listdir(folder_path):
            if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.bmp'):
                images.append(os.path.join(folder_path, filename))
        return images

    # 批量处理图片
    def processImages(images):
        for image_path in images:
            img = cv2.imread(image_path)
            
            begin = time.time()
            orig_height, orig_width, _ = img.shape 
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, dsize=(image_size, image_size), interpolation=cv2.INTER_AREA)
            input_img = transform(img)
            input_img = input_img.unsqueeze(0).cuda()

            map_combined, map_st, map_ae = predict(input_img, teacher_model, student_model, 
                                                ae_model, teacher_mean, teacher_std,
                                                q_st_start=q_st_start, q_st_end=q_st_end, 
                                                q_ae_start=q_ae_start, q_ae_end=q_ae_end)
            
            map_combined = torch.nn.functional.pad(map_combined, (4, 4, 4, 4))
            map_combined = torch.nn.functional.interpolate(
            map_combined, (orig_height, orig_width), mode='bilinear')
            map_combined = map_combined[0, 0].cpu().detach().numpy()
            ad_score = np.max(map_combined)
            end = time.time()
            
            score_list.append(ad_score)
            time_list.append((end - begin) * 1000)
            print(image_path, " score: ", ad_score, "time: ", (end - begin) * 1000, "ms")
            
            
            
    teacher_model = torch.load("./output/trainings/mvtec_loco/" + ad_object + "/teacher_final.pth").eval().cuda()
    student_model = torch.load("./output/trainings/mvtec_loco/" + ad_object + "/student_final.pth").eval().cuda()
    ae_model = torch.load("./output/trainings/mvtec_loco/" + ad_object + "/autoencoder_final.pth").eval().cuda()

    folder_path = "./mvtec_loco_anomaly_detection/" + ad_object + "/test/" + ad_type
    images = getImages(folder_path)
    processImages(images)

    max_score = max(score_list)
    min_score = min(score_list)
    avg_score = sum(score_list) / len(score_list)
    avg_time = sum(time_list) / len(time_list)

    print("Type: ", ad_type)
    print("max: ", max_score, "min: ", min_score, "avg: ", avg_score, "avg time: ", avg_time, "ms")


def plot_histogram(good_scores, broken_scores):
    plt.clf()
    # plt.hist([good_scores, broken_scores], bins=5, label=['Good', 'Broken'])
    plt.hist(good_scores, alpha=0.5, label='Good')
    plt.hist(broken_scores, alpha=0.5, label='Broken', color='crimson')
    plt.xlabel("Score")
    plt.ylabel('Count')
    plt.legend()
    plt.savefig(pic_path)
    plt.show()

good_score_list = []
broken_score_list = []
time_list = []

inference(good_score_list, time_list, ad_type=ad_type)
time_list = []
inference(broken_score_list, time_list, ad_type="broken")
plot_histogram(good_score_list, broken_score_list)




