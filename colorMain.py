import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.color import lab2rgb, rgb2lab, rgb2gray
from skimage import io
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import datasets, transforms
import os, shutil, time, argparse
import cv2
import glob
import tkinter as tk
from tkinter import filedialog

# Check if GPU is available
use_gpu = torch.cuda.is_available()
training = False

par = argparse.ArgumentParser(description="Nástroj na kolorizovanie videa")
par.add_argument("-video2frames", dest = 'video2frames', action = 'store_true', help = "Konverzia videa na snímky")
par.add_argument("-frames2video", dest = 'frames2video', action = 'store_true', help = "Konverzia snímiek na video")
par.add_argument("-complet", dest = 'complet', action = 'store_true', help = "Kompletný proces konverzie videa na snímky, kolorizovanie snímkov a následné spojenie snímkov do videa")
par.add_argument("-celeb", dest = 'celeb' , action = 'store_true', help = "Načítanie checkpointu modelu trénovaného na datasete CelebA")
par.add_argument("-places", dest = 'places' , action = 'store_true', help = "Načítanie checkpointu modelu trénovaného na datasete Places365")
par.add_argument("-placeleb", dest = 'placeleb' , action = 'store_true', help = "Načítanie checkpointu modelu trénovaného na kombinovanom datasete Places365 a CelebA")


class ColorizationNet(nn.Module):
    def __init__(self, midlevel_input_size=128, global_input_size=512):
        super(ColorizationNet, self).__init__()
        # Fúzna vrstva na spojenie globálnych príznakov s príznakmi strednej úrovne
        self.midlevel_input_size = midlevel_input_size
        self.global_input_size = global_input_size
        self.fusion = nn.Linear(midlevel_input_size + global_input_size, midlevel_input_size)
        self.bn1 = nn.BatchNorm1d(midlevel_input_size)

        # Konvolučné vrstvy a upsampling
        self.deconv1_new = nn.ConvTranspose2d(midlevel_input_size, 128, kernel_size=4, stride=2, padding=1)
        self.conv1 = nn.Conv2d(midlevel_input_size, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.conv5 = nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1)
        self.upsample = nn.Upsample(scale_factor=2)

    def forward(self, midlevel_input): 
        
        # Konvolučné vrstvy a upsampling
        x = F.relu(self.bn2(self.conv1(midlevel_input)))
        x = self.upsample(x)
        x = F.relu(self.bn3(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.upsample(x)
        x = torch.sigmoid(self.conv4(x))
        x = self.upsample(self.conv5(x))
        return x


class ColorNet(nn.Module):
    def __init__(self):
        super(ColorNet, self).__init__()

        resnet_gray_model = models.resnet18()
        resnet_gray_model.conv1.weight = nn.Parameter(resnet_gray_model.conv1.weight.sum(dim=1).unsqueeze(1).data)

        # Vyňatie globálnych príznakov a príznakov strednej úrovne z ResNet-gray
        self.midlevel_resnet = nn.Sequential(*list(resnet_gray_model.children())[0:6])
        self.global_resnet = nn.Sequential(*list(resnet_gray_model.children())[0:9])
        self.fusion_and_colorization_net = ColorizationNet()

    def forward(self, input_image):

        # Posunúť vstup do ResNet-gray na získanie príznakov
        midlevel_output = self.midlevel_resnet(input_image)
        global_output = self.global_resnet(input_image)

        # Spojenie príznakov vo fúznej vrstve
        output = self.fusion_and_colorization_net(midlevel_output)
        return output



class GrayscaleImageFolder(datasets.ImageFolder):
  '''Custom images folder, which converts images to grayscale before loading'''
  def __getitem__(self, index):
    path, target = self.imgs[index]
    img = self.loader(path)
    if self.transform is not None:
      img_original = self.transform(img)
      img_original = np.asarray(img_original)
      img_lab = rgb2lab(img_original)
      img_lab = (img_lab + 128) / 255
      img_ab = img_lab[:, :, 1:3]
      img_ab = torch.from_numpy(img_ab.transpose((2, 0, 1))).float()
      img_original = rgb2gray(img_original)
      img_original = torch.from_numpy(img_original).unsqueeze(0).float()
    if self.target_transform is not None:
      target = self.target_transform(target)
    return img_original, img_ab, target

class AverageMeter(object):
  '''A handy class from the PyTorch ImageNet tutorial''' 
  def __init__(self):
    self.reset()
  def reset(self):
    self.val, self.avg, self.sum, self.count = 0, 0, 0, 0
  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count

def to_rgb(grayscale_input, ab_input, save_path=None, save_name=None):
  plt.clf() 
  color_image = torch.cat((grayscale_input, ab_input), 0).numpy()
  color_image = color_image.transpose((1, 2, 0)) 
  color_image[:, :, 0:1] = color_image[:, :, 0:1] * 100
  color_image[:, :, 1:3] = color_image[:, :, 1:3] * 255 - 128   
  color_image = lab2rgb(color_image.astype(np.float64))
  grayscale_input = grayscale_input.squeeze().numpy()
  if save_path is not None and save_name is not None: 
    plt.imsave(arr=grayscale_input, fname='{}{}'.format(save_path['grayscale'], save_name), cmap='gray')
    plt.imsave(arr=color_image, fname='{}{}'.format(save_path['colorized'], save_name))

def validate(val_loader, model, criterion, save_images, epoch):
  model.eval()

  batch_time, data_time, losses = AverageMeter(), AverageMeter(), AverageMeter()

  end = time.time()
  already_saved_images = False
  for i, (input_gray, input_ab, target) in enumerate(val_loader):
    data_time.update(time.time() - end)

    if use_gpu: input_gray, input_ab, target = input_gray.cuda(), input_ab.cuda(), target.cuda()

    output_ab = model(input_gray)
    loss = criterion(output_ab, input_ab)
    losses.update(loss.item(), input_gray.size(0))

   
    if save_images and not already_saved_images:
      already_saved_images = True
      for j in range(min(len(output_ab), 20)): 
        save_path = {'grayscale': 'outputs/gray/', 'colorized': 'outputs/color/'}
        save_name = 'img-{}-epoch-{}.jpg'.format(i * val_loader.batch_size + j, epoch)
        to_rgb(input_gray[j].cpu(), ab_input=output_ab[j].detach().cpu(), save_path=save_path, save_name=save_name)

    batch_time.update(time.time() - end)
    end = time.time()

    if i % 25 == 0:
      print('Validate: [{0}/{1}]\t'
            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
             i, len(val_loader), batch_time=batch_time, loss=losses))

  print('Finished validation.')
  return losses.avg



def color(col_loader, model, criterion, save_images):
  model.eval()
  zeros = "00000"
  count = 0

  dir = 'vystup/gray/'
  for file in os.scandir(dir):
    os.remove(file.path)

  dir = 'vystup/col/'
  for file in os.scandir(dir):
    os.remove(file.path)

  batch_time, data_time, losses = AverageMeter(), AverageMeter(), AverageMeter()
  print("Celkový počet snímkov na vyfarbenie: %d" % (len(col_loader.dataset)))
  print("Pripravujú sa dáta na kolorizovanie...")
  end = time.time()
  for i, (input_gray, input_ab, target) in enumerate(col_loader):
    data_time.update(time.time() - end)

    if use_gpu: input_gray, input_ab, target = input_gray.cuda(), input_ab.cuda(), target.cuda()

    output_ab = model(input_gray) 
    loss = criterion(output_ab, input_ab)
    losses.update(loss.item(), input_gray.size(0))

    for j in range(0, len(output_ab)): 

        if (count // 10 != 0):
            zeros = "0000"
        if (count // 100 != 0):
            zeros = "000"
        if (count // 1000 != 0):
            zeros = "00"
        if (count // 10000 != 0):
            zeros = "0"

        count = i * col_loader.batch_size + j
        save_path = {'grayscale': 'vystup/gray/','colorized': 'vystup/col/'}
        save_name = '{}{}.jpg'.format(zeros, count)
        to_rgb(input_gray[j].cpu(), ab_input=output_ab[j].detach().cpu(), save_path=save_path, save_name=save_name)

    batch_time.update(time.time() - end)
    end = time.time()
    
    print('Kolorizovaná várka: [{0}/{1}]\t'
        'Čas {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        'Strata {loss.val:.5f} ({loss.avg:.5f})\t'.format(
            i+1, len(col_loader), batch_time=batch_time, loss=losses))

  print("Kolorizovanie ukončené..")
  return losses.avg



def train(train_loader, model, criterion, optimizer, epoch):
  print('Starting training epoch {}'.format(epoch))
  model.train()
  
  batch_time, data_time, losses = AverageMeter(), AverageMeter(), AverageMeter()

  end = time.time()
  for i, (input_gray, input_ab, target) in enumerate(train_loader):

    if use_gpu: input_gray, input_ab, target = input_gray.cuda(), input_ab.cuda(), target.cuda()

    data_time.update(time.time() - end)

    output_ab = model(input_gray) 
    loss = criterion(output_ab, input_ab) 
    losses.update(loss.item(), input_gray.size(0))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    batch_time.update(time.time() - end)
    end = time.time()

    if i % 25 == 0:
      print('Epocha: [{0}][{1}/{2}]\t'
            'Čas {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
            'Strata {loss.val:.5f} ({loss.avg:.5f})\t'.format(
              epoch, i, len(train_loader), batch_time=batch_time,
             data_time=data_time, loss=losses)) 

  print('Skončená trénovacia epocha {}'.format(epoch))

def toFrames(video_path):

    vidcap = cv2.VideoCapture(video_path)
    success,image = vidcap.read()
    total = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(vidcap.get(cv2.CAP_PROP_FPS))
    if success:
        print("Video sa načítalo.")
    progres = 0
    count = 0
    zeros = "00000"

    while success:
        if (count // 10 != 0):
            zeros = "0000"
        if (count // 100 != 0):
            zeros = "000"
        if (count // 1000 != 0):
            zeros = "00"
        if (count // 10000 != 0):
            zeros = "0"

        print("Video sa rozdeľuje na snímky. Progres: %d%%" % (progres), end='\r')
        path ="kolorizuj/col/%s%d.jpg" % (zeros,count)     
        cv2.imwrite(path, image)        
        success,image = vidcap.read()
        count += 1
        progres = count / total * 100
        progres = int(progres)

    print("Konverzia videa na snímky úspešne ukončená.")
    return fps

def toVideo(video_path, fps):
    img_array = []
    progres = 0
    if fps == 0:
        fps = input("Zadaj framerate (počet snímkov za sekundu) produkovaného videa:")
    fps = int(fps)

    for filename in glob.glob('vystup/col/*.jpg'):
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)


    out = cv2.VideoWriter(video_path,cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
 
    for i in range(len(img_array)):
        print("Snímky sa spájajú do videa. Progres: %d%%" % (progres), end='\r')
        out.write(img_array[i])
        progres = i / len(img_array) * 100
        progres = int(progres)
    out.release()
    print("Video bolo uložené na adrese: %s" % video_path)

def getDim():
    pic = glob.glob('kolorizuj/col/*')
    first = cv2.imread(pic[0])
    height, width, ch = first.shape

    if height > width: 
        if width > 720:
            ratio = width / 720
            new_height = int(height / ratio)
            remain = new_height % 8
            new_height = new_height - remain
            new_width = 720
        else: 
            remain = width % 8
            new_width = width - remain
            remain = height % 8
            new_height = height - remain
    elif width == height:
        remain = width % 8
        new_width = width - remain
        remain = height % 8
        new_height = height - remain
    else:
        if height > 720:
            ratio = height / 720
            new_width = int(width / ratio)
            remain = new_width % 8
            new_width = new_width - remain
            new_height = 720
        else: 
            remain = width % 8
            new_width = width - remain
            remain = height % 8
            new_height = height - remain

    return new_height, new_width


if __name__ == '__main__':
    fps = 0
    model = ColorNet()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0)
    args = par.parse_args()
    height = 0
    width = 0

    if use_gpu: 
        criterion = criterion.cuda()
        model = model.cuda()

    os.makedirs('vystup/col', exist_ok=True)
    os.makedirs('vystup/gray', exist_ok=True)
    os.makedirs('kolorizuj', exist_ok=True)
    os.makedirs('kolorizuj/col', exist_ok=True)
    os.makedirs('checkpointy', exist_ok=True)


    save_images = True
    epochs = 100

    if training:
        # Training
        train_transforms = transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip()])
        train_imagefolder = GrayscaleImageFolder('C:\\Users\\hrebe\\Desktop\\Diplomovka\\data\\Placeleb\\train', train_transforms)
        train_loader = torch.utils.data.DataLoader(train_imagefolder, batch_size=64, shuffle=True, num_workers = 8)

        # Validation 
        val_transforms = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])
        val_imagefolder = GrayscaleImageFolder('C:\\Users\\hrebe\\Desktop\\Diplomovka\\data\\Placeleb\\val' , val_transforms)
        val_loader = torch.utils.data.DataLoader(val_imagefolder, batch_size=64, shuffle=False, num_workers = 8)


        for epoch in range(epochs):
          train(train_loader, model, criterion, optimizer, epoch)
          with torch.no_grad():
            losses = validate(val_loader, model, criterion, save_images, epoch)
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': losses,
                    }, 'checkpoints/placesbigmodel-epoch-{}-losses-{:.5f}.pth'.format(epoch+1,losses))
    else:


        if args.celeb or args.places or args.placeleb or args.video2frames or args.frames2video or args.complet:



            if args.video2frames or args.complet:
                root = tk.Tk()
                root.withdraw()
                file_path = filedialog.askopenfilename(title='Zvoľte video na kolorizovanie')

                dir = 'kolorizuj/col/'
                for file in os.scandir(dir):
                    os.remove(file.path)

                fps = toFrames(file_path)
                


            height, width = getDim()
            col_transforms = transforms.Compose([transforms.Resize((height, width))])
            col_imagefolder = GrayscaleImageFolder('kolorizuj/', col_transforms)
            col_loader = torch.utils.data.DataLoader(col_imagefolder, batch_size=16, shuffle=False, num_workers = 8)

            if args.celeb:
                if use_gpu:
                    model.load_state_dict(torch.load('checkpointy/celeba_checkpoint.pth'))
                else:
                    model.load_state_dict(torch.load('checkpointy/celeba_checkpoint.pth', map_location=torch.device('cpu')))
                print("Checkpoint CelebA úspešne načítaný..")
                with torch.no_grad():
                 losses = color(col_loader, model, criterion, save_images)

            elif args.places: 
                if use_gpu:
                    checkpoint = torch.load('checkpointy/places365_checkpoint.pth')
                else:
                    checkpoint = torch.load('checkpointy/places365_checkpoint.pth', map_location=torch.device('cpu'))
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                epoch = checkpoint['epoch']
                loss = checkpoint['loss']
                print("Checkpoint Places365 úspešne načítaný..")
                with torch.no_grad():
                 losses = color(col_loader, model, criterion, save_images)

            elif args.placeleb: 
                if use_gpu:
                    checkpoint = torch.load('checkpointy/placeleb_checkpoint.pth')
                else:
                    checkpoint = torch.load('checkpointy/placeleb_checkpoint.pth', map_location=torch.device('cpu'))
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                epoch = checkpoint['epoch']
                loss = checkpoint['loss']
                print("Checkpoint zmiešaného datasetu PlaCeleb úspešne načítaný..")
                with torch.no_grad():
                 losses = color(col_loader, model, criterion, save_images)

            else:
                print('Nebol zvolený žiaden checkpoint..')

            if args.frames2video or args.complet:
                root = tk.Tk()
                root.withdraw()
                file_path = filedialog.asksaveasfilename(title='Uložte kolorizované video', defaultextension='.mp4')
                toVideo(file_path, fps)

        else: 
            print('Nebola zvolená žiadna akcia. Pre zobrazenie dostupných argumentov spustite script s parametrom -h')
            