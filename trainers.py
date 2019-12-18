import os
from datetime import datetime
import time
from abc import ABCMeta, abstractmethod
import itertools
import json

import numpy as np
from skimage.io import imsave, imread
from skimage.transform import resize
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from models import CycleGAN
from dataset import UnpairedImageDataset, ReplayBuffer
from losses import CriterionWGANgp

class BaseTrainer(metaclass=ABCMeta):
    def __init__(self, params: dict):
        self.domain_X = params["domain_X"]
        self.domain_Y = params["domain_Y"]
        if "test_X" in params.keys():
            self.test_X = params["test_X"]
        else:
            self.test_X = None

        if "test_Y" in params.keys():
            self.test_Y = params["test_Y"]
        else:
            self.test_Y = None

        self.data_dir = "./data"

        self.max_epoch = params["max_epoch"]
        self.batch_size = params["batch_size"]
        self.image_size = params["image_size"]
        self.learning_rate = params["learning_rate"]

        model_name = params["model_type"]
        dt_now = datetime.now()
        dt_seq = dt_now.strftime("%y%m%d_%H%M")
        self.result_dir = os.path.join("./result", f"{dt_seq}_{model_name}")
        self.weight_dir = os.path.join(self.result_dir, "weights")
        self.sample_dir = os.path.join(self.result_dir, "sample")
        os.makedirs(self.result_dir, exist_ok=True)
        os.makedirs(self.weight_dir, exist_ok=True)
        os.makedirs(self.sample_dir, exist_ok=True)
        
        with open(os.path.join(self.result_dir, "params.json"), mode="w") as f:
            json.dump(params, f)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @abstractmethod
    def train(self, model):
        pass


class CycleGANTrainer(BaseTrainer):
    def __init__(self, params):
        super(CycleGANTrainer, self).__init__(params)
        self.lambda_cyc = params["lambda_cyc"]
        self.num_channels = params["num_channels"]

    def train(self):
        model = CycleGAN(self.num_channels)
        self.cast_model(model)
        print("Constructed CycleGAN model.")

        # Construct Dataloader
        dataset = UnpairedImageDataset(self.domain_X, self.domain_Y, self.image_size)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, num_workers=4, drop_last=True)

        buffer_X = ReplayBuffer()
        buffer_Y = ReplayBuffer()

        print("Construced dataloader.")

        # Construct Optimizers
        optimizer_G = Adam(itertools.chain(model.G_XY.parameters(), model.G_YX.parameters()), 
                           lr=self.learning_rate, betas=(0.5, 0.999))
        optimizer_D_X = Adam(model.D_X.parameters(),
                            lr=self.learning_rate, betas=(0.5, 0.999))
        optimizer_D_Y = Adam(model.D_Y.parameters(),
                            lr=self.learning_rate, betas=(0.5, 0.999))

        lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(self.max_epoch, 0,
                                                                                           self.max_epoch//2).step)
        lr_scheduler_D_X = torch.optim.lr_scheduler.LambdaLR(optimizer_D_X, lr_lambda=LambdaLR(self.max_epoch, 0,
                                                                                           self.max_epoch//2).step)
        lr_scheduler_D_Y = torch.optim.lr_scheduler.LambdaLR(optimizer_D_Y, lr_lambda=LambdaLR(self.max_epoch, 0,
                                                                                           self.max_epoch//2).step)

        #domain_loss_X = CriterionWGANgp(model.D_X)
        #domain_loss_Y = CriterionWGANgp(model.D_Y)
        domain_loss_X = nn.BCELoss()
        domain_loss_Y = nn.BCELoss()
        cycle_consistency_loss = nn.L1Loss()
        target_real = torch.ones((self.batch_size, 1), dtype=torch.float32).to(device=self.device)
        target_fake = torch.zeros((self.batch_size, 1), dtype=torch.float32).to(device=self.device)

        start_time = time.time()
        for epoch in range(1, self.max_epoch + 1):
            print(f"epoch {epoch} Starts.")
            for batch in dataloader:
                images_X, images_Y = self.cast_images(batch)
                
                #
                # train generator
                #
    
                XY = model.G_XY(images_X)
                XYX = model.G_YX(XY)
                YX = model.G_YX(images_Y)
                YXY = model.G_XY(YX)

                #loss_dy = domain_loss_Y.gen_loss(XY)
                #loss_dx = domain_loss_X.gen_loss(YX)
                loss_dy = domain_loss_Y(model.D_Y(XY), target_real)
                loss_dx = domain_loss_X(model.D_X(YX), target_real)
                loss_cyc_x = cycle_consistency_loss(XYX, images_X)
                loss_cyc_y = cycle_consistency_loss(YXY, images_Y)

                loss_G = loss_dx + loss_dy + self.lambda_cyc * (loss_cyc_x + loss_cyc_y)

                optimizer_G.zero_grad()
                loss_G.backward()
                optimizer_G.step()

                #
                # train discriminators
                #
                XY = (model.G_XY(images_X)).detach()
                XY = buffer_Y.push_and_pop(XY)
                YX = (model.G_YX(images_Y)).detach()
                YX = buffer_X.push_and_pop(YX)
                # train discriminator X
                #loss_D_X = domain_loss_X.dis_loss(images_X, YX)
                loss_D_X = domain_loss_X(model.D_X(YX), target_fake) + domain_loss_X(model.D_X(images_X), target_real)
                optimizer_D_X.zero_grad()
                loss_D_X.backward()
                optimizer_D_X.step()

                # train discriminator Y
                #loss_D_Y = domain_loss_Y.dis_loss(images_Y, XY)
                loss_D_Y = domain_loss_Y(model.D_Y(XY), target_fake) + domain_loss_Y(model.D_Y(images_Y), target_real)
                optimizer_D_Y.zero_grad()
                loss_D_Y.backward()
                optimizer_D_Y.step()
                print(f"loss_D_X: {loss_D_X.item():.4f}, loss_D_Y: {loss_D_Y.item():.4f}, loss_G: {loss_G.item():.4f}")

            # data shuffle at end of epoch
            dataloader.dataset.shuffle()
            
            # save generator's weights
            self.save_weights(model, epoch)

            # save sample image
            ori_image_X = images_X[0].detach()
            translated_image_X = XY[0].detach()
            ori_image_Y = images_Y[0].detach()
            translated_image_Y = YX[0].detach()

            sample_tile = torch.cat([torch.cat([ori_image_X, translated_image_X], dim=2), torch.cat([ori_image_Y, translated_image_Y], dim=2)], dim=1)
            sample_tile = sample_tile.permute((1, 2, 0))

            self.save_sample(sample_tile, epoch)

            # test sample
            if self.test_X is not None:
                filelist = os.listdir(os.path.join(self.data_dir, self.test_X))
                for filename in filelist:
                    image = imread(os.path.join(self.data_dir, self.test_X, filename))
                    image = ((image.transpose((2, 0, 1))) / 127.5 - 1).astype(np.float32)
                    image = torch.from_numpy(image).to(self.device).unsqueeze(0)
                    with torch.no_grad():
                        translated_image = model.G_XY(image)
                    image = np.squeeze(image.to("cpu").numpy(), axis=0)
                    translated_image = np.squeeze(translated_image.to("cpu").numpy(), axis=0)
                    translated_image = resize(translated_image, image.shape)
                    image = np.concatenate([image, translated_image], axis=2)
                    image = ((image + 1) * 127.5).astype(np.uint8)
                    image = np.transpose(image, (1, 2, 0))
                    imsave(os.path.join(self.sample_dir, f"{epoch:03}_{filename}"), image)

            if self.test_X is not None:
                filelist = os.listdir(os.path.join(self.data_dir, self.test_Y))
                for filename in filelist:
                    image = imread(os.path.join(self.data_dir, self.test_Y, filename))
                    image = ((image.transpose((2, 0, 1))) / 127.5 - 1).astype(np.float32)
                    image = torch.from_numpy(image).to(self.device).unsqueeze(0)
                    with torch.no_grad():
                        translated_image = model.G_YX(image)
                    image = np.squeeze(image.to("cpu").numpy(), axis=0)
                    translated_image = np.squeeze(translated_image.to("cpu").numpy(), axis=0)
                    translated_image = resize(translated_image, image.shape)
                    image = np.concatenate([image, translated_image], axis=2)
                    image = ((image + 1) * 127.5).astype(np.uint8)
                    image = np.transpose(image, (1, 2, 0))
                    imsave(os.path.join(self.sample_dir, f"{epoch:03}_{filename}"), image)

            elapsed = time.time() - start_time
            print(f"Epoch {epoch} ends. (total time: {elapsed:.2f})")

            lr_scheduler_G.step()
            lr_scheduler_D_X.step()
            lr_scheduler_D_Y.step()

    def save_weights(self, model, epoch):
        torch.save(model.G_XY.state_dict(), os.path.join(self.weight_dir, f"Generator_{self.domain_X}2{self.domain_Y}_{epoch:04}.pth"))
        torch.save(model.G_YX.state_dict(), os.path.join(self.weight_dir, f"Generator_{self.domain_Y}2{self.domain_X}_{epoch:04}.pth"))

    def save_sample(self, sample, epoch):
        sample = sample.to("cpu").numpy()
        sample = ((sample + 1) * 127.5).astype(np.uint8)
        imsave(os.path.join(self.sample_dir, f"{epoch}.png"), sample)

    def cast_images(self, batch):
        image_X, image_Y = batch
        image_X = image_X.to(self.device)
        image_Y = image_Y.to(self.device)
        return image_X, image_Y

    def cast_model(self, model):
        model.G_XY.to(self.device)
        model.G_YX.to(self.device)
        model.D_X.to(self.device)
        model.D_Y.to(self.device)


class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)              

