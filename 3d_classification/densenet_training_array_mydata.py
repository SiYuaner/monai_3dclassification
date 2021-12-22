
# Copyright 2020 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# !python -c "import monai" || pip install -q "monai-weekly[nibabel, tqdm]"
# import monai.tifffile
import logging
import os
import sys
import tempfile
import shutil
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import tifffile
import monai
from monai.config import print_config
from monai.data import CacheDataset, DataLoader, ImageDataset, ZipDataset
from monai.transforms import (
    AddChannel,
    Compose,
    RandRotate90,
    Resize,
    ScaleIntensity,
    EnsureType,
    Randomizable,
    LoadImaged,
    EnsureTyped,
)
import urllib.request
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
response = urllib.request.urlopen('https://www.python.org')
import time
import settings

class Settings:
    def __init__(self, settings):

        for attr in dir(settings):
            if attr.isupper():
                setattr(self, attr, getattr(settings, attr))




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# print(response.read().decode('utf-8'))
# ————————————————
# 版权声明：本文为CSDN博主「悠闲独自在」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
# 原文链接：https://blog.csdn.net/qq_25403205/article/details/81258327


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
# print_config()

# class DatasetA(Dataset):
#     def __getitem__(self, index: int):
#         return image_data[index]
# class DatasetB(Dataset):
#     def __getitem__(self, index: int):
#         return extra_data[index]
# dataset = ZipDataset([DatasetA(), DatasetB()], transform)





def make_data_name(root, data_type, section='train'):
    # section='train' or 'val'
    # data_type = 1, 2, 3
    txt = os.path.join(root, section + '.txt')
    f = open(txt, 'r')
    contents = f.readlines()
    f.close()

    images = []
    labels = []

    for i in range(len(contents)):
        content_i = contents[i]
        img_name, label = content_i.split(' ')
        label = int(label)
        if data_type == 1:
            file_name = os.path.join(root, section, img_name, img_name + '_1.tif')
        if data_type == 2:
            file_name = os.path.join(root, section, img_name, img_name + '_2.tif')
        if data_type == 3:
            file_name = os.path.join(root, section, img_name, img_name + '.tif')
        if data_type == 4:
            file_name = os.path.join(root, section, img_name, img_name + '_3dim.tif')
        if data_type == 5:
            file_name = os.path.join(root, section, img_name, img_name + '_3dim.nii.gz')
        images.append(file_name)
        labels.append(label)
    labels = np.array(labels, dtype=np.int64)
    return images, labels


def run():
    torch.multiprocessing.freeze_support()
    print('loop')

def train(root, data_type, batch_size):
    # ----------------- train_datasets -----------------
    # root = r'H:\血管后处理文献阅读\新增参考文献\graph\tutorials\3d_classification\datasets\dataset40_nii'
    # print(os.path.relpath(root))
    images_train, labels_train = make_data_name(root, data_type=data_type, section='train')

    # Define transforms
    # train_transforms = Compose([ScaleIntensity(), AddChannel(), Resize(
    #     (96, 96, 96)), RandRotate90(), EnsureType()])
    train_transforms = Compose([ScaleIntensity(), Resize(
        (96, 96, 96)), RandRotate90(), EnsureType()])
    train_ds = ImageDataset(image_files=images_train, labels=labels_train,
                            transform=train_transforms)  # , reader='tifffile'
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=torch.cuda.is_available())

    # ----------------- val_datasets -----------------
    images_val, labels_val = make_data_name(root, data_type=data_type, section='val')
    # val_transforms = Compose(
    #     [ScaleIntensity(), AddChannel(), Resize((96, 96, 96)), EnsureType()])
    val_transforms = Compose([ScaleIntensity(), Resize((96, 96, 96)), EnsureType()])
    val_ds = ImageDataset(image_files=images_val, labels=labels_val,
                          transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=0,
                            pin_memory=torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create DenseNet121, CrossEntropyLoss and Adam optimizer
    model = monai.networks.nets.DenseNet121(
        spatial_dims=3, in_channels=3, out_channels=2).to(device)
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), settings.LR)

    # start a typical PyTorch training
    val_interval = 2
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = []
    metric_values = []

    time_now = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    log_dir = os.path.join(settings.LOG_DIR, time_now)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)


    writer = SummaryWriter(log_dir = log_dir)
    max_epochs = 50
    for epoch in range(max_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        epoch_loss = 0
        step = 0

        num_correct = 0.0
        metric_count = 0

        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = len(train_ds) // train_loader.batch_size
            print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
            writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)

            value = torch.eq(outputs.argmax(dim=1), labels)
            metric_count += len(value)
            num_correct += value.sum().item()
        metric = num_correct / metric_count
        metric_values.append(metric)


        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                num_correct = 0.0
                metric_count = 0
                for val_data in val_loader:
                    val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
                    val_outputs = model(val_images)
                    value = torch.eq(val_outputs.argmax(dim=1), val_labels)
                    # print('\n', val_labels)
                    # print(val_outputs)
                    # print(value, value.sum().item())

                    metric_count += len(value)
                    num_correct += value.sum().item()
                metric = num_correct / metric_count
                metric_values.append(metric)
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(),
                               os.path.join(log_dir, "best_metric_model_classification3d_array.pth"))
                    print("saved new best metric model")
                print(
                    "current epoch: {} current accuracy: {:.4f} "
                    "best accuracy: {:.4f} at epoch {}".format(
                        epoch + 1, metric, best_metric, best_metric_epoch
                    )
                )
                writer.add_scalar("val_accuracy", metric, epoch + 1)
    print(
        f"train completed, best_metric: {best_metric:.4f} "
        f"at epoch: {best_metric_epoch}")
    writer.close()

def get_next_im(itera):
    test_data = next(itera)
    return test_data[0].to(device), test_data[1].unsqueeze(0).to(device)



def test(test_root, model_path, save_dir, batch_size, label_avaliable=False, data_type=5):
    if not os.path.exists(os.path.join(save_dir, 'data')):
        os.makedirs(os.path.join(save_dir, 'data'))
    save_txt = os.path.join(save_dir, 'prediction.txt')
    f = open(save_txt, 'w')
    if label_avaliable:
        f.write('index   label   prediction \n')
    else:
        f.write('index   prediction \n')
    f.close()

    model = monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=3, out_channels=2).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    images_test, labels_test = make_data_name(test_root, data_type=data_type, section='train')
    test_transforms = Compose([ScaleIntensity(), Resize((96, 96, 96)), EnsureType()])
    test_ds = ImageDataset(image_files=images_test, labels=labels_test, transform=test_transforms)
    test_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=0,
                             pin_memory=torch.cuda.is_available())
    metric_values = []

    with torch.no_grad():
        num_correct = 0.0
        metric_count = 0
        index = 0
        for test_data in test_loader:
            test_images, test_labels = test_data[0].to(device), test_data[1].to(device)
            test_outputs = model(test_images)
            value = torch.eq(test_outputs.argmax(dim=1), test_labels)

            index = save_prediction(index, images_test, test_images, test_outputs, save_dir, label=test_labels)

            metric_count += len(value)
            num_correct += value.sum().item()
        metric = num_correct / metric_count
        print('average_accuracy: ', metric)
        metric_values.append(metric)

def save_prediction(index, images_test, img, prediction, save_dir, label=None):
    # 保存结果
    save_txt = os.path.join(save_dir, 'prediction.txt')
    print('batch: ', img.shape[0])

    for ii in range(img.shape[0]):
        data = img[ii,0].mul(255).byte()
        data = data.cpu().numpy()  # 这个数据是两个血管段都有的数据


        tifffile.imsave(os.path.join(save_dir, 'data', str(index).zfill(5) + '.tif'), data)
        f = open(save_txt, 'a')
        if label != None:
            f.write(str(index).zfill(5) + '-' + images_test[index] + ' ' + str(label[ii].item()) + ' ' + str(prediction.argmax(dim=1)[ii].item()) + '\n')
        else:
            f.write(str(index).zfill(5) + '-' + images_test[index] + ' ' + str(prediction.argmax(dim=1)[ii].item()) + '\n')
        f.close()
        index += 1
    return index






if __name__ == '__main__':
    run()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    print_config()

    directory = os.environ.get("MONAI_DATA_DIRECTORY")
    root_dir = tempfile.mkdtemp() if directory is None else directory
    print(root_dir)


    # ----------------- train_  -----------------
    settings = Settings(settings)

    root = settings.ROOT
    data_type = settings.DATA_TYPE
    batch_size = settings.BATCH_SIZE
    # train(root, data_type, batch_size)

    # ----------------- test  -----------------
    best_model_path = settings.BEST_MODEL_PATH
    test_root = settings.TEST_PATH
    save_dir = settings.SAVE_DIR
    test(test_root, best_model_path, save_dir, batch_size, label_avaliable = True, data_type=data_type)
