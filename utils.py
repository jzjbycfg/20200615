import json
import matplotlib.pyplot as plt
import SimpleITK as sitk
import time
import pickle
import torch
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as transforms
from PIL import Image
import pandas as pd
import glob
import torch.nn as nn
import torch.nn.functional as F
import pickle


# save_obj(loss_dict, r'C:\Users\Administrator\Desktop\2020spring\MedicalImageAnalysis\models\c2loss_dict')
def save_obj(obj, name):  # 保存dict
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


# obj = load_obj(r'C:\Users\Administrator\Desktop\2020spring\MedicalImageAnalysis\models\c2loss_dict')
def load_obj(name):  # 加载dict
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


# 参考：https://tianchi.aliyun.com/forum/postDetail?postId=112589
def dicom_metainfo(dicm_path, list_tag):
    '''
    获取dicom的元数据信息
    :param dicm_path: dicom文件地址
    :param list_tag: 标记名称列表,比如['0008|0018',]
    :return:
    '''
    reader = sitk.ImageFileReader()
    reader.LoadPrivateTagsOn()
    reader.SetFileName(dicm_path)
    reader.ReadImageInformation()
    return [reader.GetMetaData(t) for t in list_tag]


# 参考：https://tianchi.aliyun.com/forum/postDetail?postId=112589
def dicom2array(dcm_path):
    '''
    读取dicom文件并把其转化为灰度图(np.array)
    https://simpleitk.readthedocs.io/en/master/link_DicomConvert_docs.html
    :param dcm_path: dicom文件
    :return:
    '''
    image_file_reader = sitk.ImageFileReader()
    image_file_reader.SetImageIO('GDCMImageIO')
    image_file_reader.SetFileName(dcm_path)
    image_file_reader.ReadImageInformation()
    image = image_file_reader.Execute()
    if image.GetNumberOfComponentsPerPixel() == 1:
        image = sitk.RescaleIntensity(image, 0, 255)
        if image_file_reader.GetMetaData('0028|0004').strip() == 'MONOCHROME1':
            image = sitk.InvertIntensity(image, maximum=255)
        image = sitk.Cast(image, sitk.sitkUInt8)
    img_x = sitk.GetArrayFromImage(image)[0]
    return img_x


# 参考：https://tianchi.aliyun.com/forum/postDetail?postId=113064
def get_info(trainPath, jsonPath):
    annotation_info = pd.DataFrame(columns=('studyUid', 'seriesUid', 'instanceUid', 'annotation'))
    json_df = pd.read_json(jsonPath)
    for idx in json_df.index:
        studyUid = json_df.loc[idx, "studyUid"]
        seriesUid = json_df.loc[idx, "data"][0]['seriesUid']
        instanceUid = json_df.loc[idx, "data"][0]['instanceUid']
        annotation = json_df.loc[idx, "data"][0]['annotation']
        row = pd.Series(
            {'studyUid': studyUid, 'seriesUid': seriesUid, 'instanceUid': instanceUid, 'annotation': annotation})
        annotation_info = annotation_info.append(row, ignore_index=True)
    dcm_paths = glob.glob(os.path.join(trainPath, "**", "**.dcm"))  # 具体的图片路径
    # 'studyUid','seriesUid','instanceUid'
    tag_list = ['0020|000d', '0020|000e', '0008|0018']
    dcm_info = pd.DataFrame(columns=('dcmPath', 'studyUid', 'seriesUid', 'instanceUid'))
    for dcm_path in dcm_paths:
        try:
            studyUid, seriesUid, instanceUid = dicom_metainfo(dcm_path, tag_list)  # 获取当前图像的3个id，然后再与json中的对应
            row = pd.Series(
                {'dcmPath': dcm_path, 'studyUid': studyUid, 'seriesUid': seriesUid, 'instanceUid': instanceUid})
            dcm_info = dcm_info.append(row, ignore_index=True)
        except:
            continue
    result = pd.merge(annotation_info, dcm_info, on=['studyUid', 'seriesUid', 'instanceUid'])  # 两个pd根据3个id合并
    result = result.set_index('dcmPath')['annotation']  # 然后把index设置为路径，值设置为annotation
    return result


# 从图片中获取每个椎体和椎间盘的那个小patch，作为训练样本(samples)
def get_sample(res):
    samples = []
    for i in range(len(res)):
        img_dir = res.index[0]  # 获取图片的地址
        img_arr = dicom2array(img_dir)  # 获取具体的图片数据，二维数据
        tags = res[0][0]['data']['point']  # 获取图片的标签
        for tag in tags:
            coord = tag['coord']  # 获取这个标签所在的坐标
            # 需要把每个方框中的小patch提取出来，以及对应的label
            top_left_x, top_left_y = coord[0]-16, coord[1]-16
            width, height = 32, 32  # 使目标点在正中间
            patch = img_arr[top_left_y:top_left_y+height, top_left_x:top_left_x+width]
            dic = tag['tag']
            label = list(dic.keys())[-1] + '-' + list(dic.values())[-1]
            samples.append((patch, label))
    return samples


class MyDataset(Dataset):
    # 需要自己写一个Dataset类，并且要继承从torch中import的Dataset基类，然后重写__len__和__getitem__两个方法，否则会报错
    # 此外还需要写__init__，传入数据所在路径和transform(用于数据预处理)
    def __init__(self, data_dir, transform=None):
        """
        :param data_dir: 读取的数据所在的路径
        :param transform: 数据预处理参数
        """
        self.label_name = {'vertebra-v1': 0, 'vertebra-v2': 1, 'disc-v1': 2, 'disc-v2': 3, 'disc-v3': 4, 'disc-v4': 5, 'disc-v5': 6}
        self.data_info = self.dataInfo(data_dir)  # 用来读取数据信息(数据路径，标签)
        self.transform = transform

    def __getitem__(self, index):  # 根据索引读取数据路径再读取数据
        img, label = self.data_info[index]
        img = Image.fromarray(img)
        label = self.label_name[label]  # 字符串转化为数字，注意后面各类可能还会存在类别不平衡问题

        if self.transform is not None:
            img = self.transform(img)
        else:  # 避免未作transforms而忘记把图像数据转化为tensor
            img = torch.tensor(img)
        return img, label

    def unpickle(self, file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    def __len__(self):
        return len(self.data_info)

    @staticmethod
    def dataInfo(data_dir):  # 自定义函数用来获取数据信息，输入为数据所在路径，返回为一个元组(图像路径，标签路径)
        # 先读取所有的图像数据路径
        # data_info = os.listdir(data_dir)  # 读取该路径下的所有文件，此处的data_dir就是train文件夹所对应的路径
        # for i in range(len(data_info)):
        #     data_info[i] = os.path.join(data_dir, data_info[i])
        data_info = data_dir  # 由于传入的直接是数据，所以直接返回就行了，返回的是(图像数据，标签)
        return data_info  # 返回的是一个batch_size的数据路径


class ResNet(nn.Module):  # 使用方法: model = ResNet(ResidualBlock, num_classes=10, in_channels=3)
    def __init__(self, ResidualBlock, num_classes=10, in_channels=3):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 64,  2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)   # strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class ResidualBlock(nn.Module):  # 定义了残差模块类
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),  # 批归一化，防止梯度弥散或者梯度爆炸
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out