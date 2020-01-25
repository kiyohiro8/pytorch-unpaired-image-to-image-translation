import os
from datetime import datetime
import time
from abc import ABCMeta, abstractmethod
import itertools
import json
from distutils.util import strtobool

import numpy as np
from skimage.io import imsave, imread
from skimage.transform import resize
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

import models
from dataset import UnpairedImageDataset, ReplayBuffer
from losses import CriterionWGANgp, LSGAN


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
        self.result_dir = os.path.join("./result", f"{dt_seq}_{model_name}_{self.domain_X}2{self.domain_Y}")
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
        self.use_idt = params["use_idt"]
        if self.use_idt:
            print("use identity loss")
        else:
            print("not use identity loss")
        self.gen_num_channels = params["gen_num_channels"]
        self.dis_num_channels = params["dis_num_channels"]
    def train(self):
        from models.cyclegan import CycleGAN
        model = CycleGAN(self.gen_num_channels, self.dis_num_channels)
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
                           lr=self.learning_rate, betas=(0.5, 0.99))
        optimizer_D_X = Adam(model.D_X.parameters(),
                            lr=self.learning_rate*4, betas=(0.5, 0.99))
        optimizer_D_Y = Adam(model.D_Y.parameters(),
                            lr=self.learning_rate*4, betas=(0.5, 0.99))

        lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(self.max_epoch, 0,
                                                                                           self.max_epoch//2).step)
        lr_scheduler_D_X = torch.optim.lr_scheduler.LambdaLR(optimizer_D_X, lr_lambda=LambdaLR(self.max_epoch, 0,
                                                                                           self.max_epoch//2).step)
        lr_scheduler_D_Y = torch.optim.lr_scheduler.LambdaLR(optimizer_D_Y, lr_lambda=LambdaLR(self.max_epoch, 0,
                                                                                           self.max_epoch//2).step)

        target_real = torch.ones((self.batch_size, 1, 1, 1), dtype=torch.float32).to(device=self.device)
        target_fake = torch.zeros((self.batch_size, 1, 1, 1), dtype=torch.float32).to(device=self.device)
        domain_loss = LSGAN(target_real, target_fake)

        cycle_consistency_loss = nn.L1Loss()
        idt_loss = nn.L1Loss()
        
        target_real = torch.ones((self.batch_size, 1, 1, 1), dtype=torch.float32).to(device=self.device)
        target_fake = torch.zeros((self.batch_size, 1, 1, 1), dtype=torch.float32).to(device=self.device)

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

                loss_dy = domain_loss(model.D_Y(XY), is_real=True)
                loss_dx = domain_loss(model.D_X(YX), is_real=True)
                loss_cyc_x = cycle_consistency_loss(XYX, images_X)
                loss_cyc_y = cycle_consistency_loss(YXY, images_Y)
                if self.use_idt:
                    loss_idt_x = idt_loss(model.G_XY(images_Y), images_Y)
                    loss_idt_y = idt_loss(model.G_YX(images_X), images_X)

                loss_G = loss_dx + loss_dy + self.lambda_cyc * (loss_cyc_x + loss_cyc_y)

                if self.use_idt:
                    loss_G += 10 * (loss_idt_x + loss_idt_y)

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
                loss_D_X = domain_loss(model.D_X(YX), is_real=False) + domain_loss(model.D_X(images_X), is_real=True)
                optimizer_D_X.zero_grad()
                loss_D_X.backward()
                optimizer_D_X.step()

                # train discriminator Y
                loss_D_Y = domain_loss(model.D_Y(XY), is_real=False) + domain_loss(model.D_Y(images_Y), is_real=True)
                optimizer_D_Y.zero_grad()
                loss_D_Y.backward()
                optimizer_D_Y.step()
                print(f"loss_D_X: {loss_D_X.item():.4f}, loss_D_Y: {loss_D_Y.item():.4f}, loss_G: {loss_G.item():.4f}")

            # data shuffle at end of epoch
            dataloader.dataset.shuffle()
            
            # save generator's weights
            if epoch % 10 == 0:
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
                    image = image.astype(np.float32)
                    image = ((image.transpose((2, 0, 1))) / 127.5 - 1)
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

            if self.test_Y is not None:
                filelist = os.listdir(os.path.join(self.data_dir, self.test_Y))
                for filename in filelist:
                    image = imread(os.path.join(self.data_dir, self.test_Y, filename))
                    image = image.astype(np.float32)
                    image = ((image.transpose((2, 0, 1))) / 127.5 - 1)
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


class MUNITTrainer(BaseTrainer):
    def __init__(self, params):
        super(MUNITTrainer, self).__init__(params)
        self.lambda_image = params["lambda_image"]
        self.lambda_c = params["lambda_content"]
        self.lambda_s = params["lambda_style"]
        self.use_idt = params["use_idt"]
        if self.use_idt:
            print("use identity loss")
        else:
            print("not use identity loss")
        self.num_channels = params["num_channels"]
        self.style_dim = params["style_dim"]

    def train(self):
        from models.munit import MUNIT
        model = MUNIT(self.num_channels, self.style_dim)
        self.cast_model(model)
        print("Constructed MUNIT model.")

        # Construct Dataloader
        dataset = UnpairedImageDataset(self.domain_X, self.domain_Y, self.image_size)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, num_workers=4, drop_last=True)
        test_dataset = UnpairedImageDataset(self.domain_X+"_test", self.domain_Y+"_test", self.image_size)
        test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, num_workers=2, drop_last=True)

        buffer_X = ReplayBuffer()
        buffer_Y = ReplayBuffer()

        print("Construced dataloader.")

        # Construct Optimizers
        optimizer_G = Adam(itertools.chain(model.content_enc_X.parameters(), 
                                           model.content_enc_Y.parameters(),
                                           model.style_enc_X.parameters(),
                                           model.style_enc_Y.parameters(),
                                           model.dec_X.parameters(),
                                           model.dec_Y.parameters()), 
                           lr=self.learning_rate, betas=(0.5, 0.99))
        optimizer_D_X = Adam(model.dis_X.parameters(),
                            lr=self.learning_rate*4, betas=(0.5, 0.99))
        optimizer_D_Y = Adam(model.dis_Y.parameters(),
                            lr=self.learning_rate*4, betas=(0.5, 0.99))

        lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(self.max_epoch, 0,
                                                                                           self.max_epoch//2).step)
        lr_scheduler_D_X = torch.optim.lr_scheduler.LambdaLR(optimizer_D_X, lr_lambda=LambdaLR(self.max_epoch, 0,
                                                                                           self.max_epoch//2).step)
        lr_scheduler_D_Y = torch.optim.lr_scheduler.LambdaLR(optimizer_D_Y, lr_lambda=LambdaLR(self.max_epoch, 0,
                                                                                           self.max_epoch//2).step)
        target_real = torch.ones((self.batch_size, 1, 1, 1), dtype=torch.float32).to(device=self.device)
        target_fake = torch.zeros((self.batch_size, 1, 1, 1), dtype=torch.float32).to(device=self.device)
        domain_loss = LSGAN(target_real, target_fake)
        reconstruction_loss = nn.L1Loss()

        start_time = time.time()
        for epoch in range(1, self.max_epoch + 1):
            print(f"epoch {epoch} Starts.")
            for batch in dataloader:
                images_X, images_Y = self.cast_images(batch)

                #
                # training encoders and decoders
                #
                content_X = model.content_enc_X(images_X)
                style_X = model.style_enc_X(images_X)
                content_Y = model.content_enc_Y(images_Y)
                style_Y = model.style_enc_Y(images_Y)

                rec_X = model.dec_X(content_X, style_X)
                rec_Y = model.dec_Y(content_Y, style_Y)
                
                fake_X = model.dec_X(content_Y, style_X)
                fake_Y = model.dec_Y(content_X, style_Y)

                rec_c_X = model.content_enc_Y(fake_Y)
                rec_c_Y = model.content_enc_X(fake_X)

                rec_s_Y = model.style_enc_Y(fake_Y)
                rec_s_X = model.style_enc_X(fake_X)

                loss_gan = domain_loss(model.dis_X(fake_X), is_real=True) + \
                           domain_loss(model.dis_Y(fake_Y), is_real=True)

                loss_image_rec = reconstruction_loss(rec_X, images_X) + \
                                 reconstruction_loss(rec_Y, images_Y)

                loss_content_rec = reconstruction_loss(rec_c_X, content_X) + \
                                   reconstruction_loss(rec_c_Y, content_Y)

                loss_style_rec = reconstruction_loss(rec_s_X, style_X) + \
                                 reconstruction_loss(rec_s_Y, style_Y)

                loss_G = loss_gan + \
                         self.lambda_image * loss_image_rec + \
                         self.lambda_c * loss_content_rec + \
                         self.lambda_s * loss_style_rec

                optimizer_G.zero_grad()
                loss_G.backward()
                optimizer_G.step()

                fake_X = fake_X.detach()
                fake_Y = fake_Y.detach()

                #
                # training discriminators
                #
                fake_X_D = buffer_X.push_and_pop(fake_X)
                fake_Y_D = buffer_Y.push_and_pop(fake_Y)

                loss_D_X = domain_loss(model.dis_X(images_X), is_real=True) + \
                           domain_loss(model.dis_X(fake_X_D), is_real=False)

                optimizer_D_X.zero_grad()
                loss_D_X.backward()
                optimizer_D_X.step()
                
                loss_D_Y = domain_loss(model.dis_Y(images_Y), is_real=True) + \
                           domain_loss(model.dis_Y(fake_Y_D), is_real=False)
                
                optimizer_D_Y.zero_grad()
                loss_D_Y.backward()
                optimizer_D_Y.step()

                print(f"loss_D_X: {loss_D_X.item():.4f}, loss_D_Y: {loss_D_Y.item():.4f}, loss_G: {loss_G.item():.4f}")
                
            # data shuffle at end of epoch
            dataloader.dataset.shuffle()

            # save generator's weights
            if epoch % 10 == 0:
                self.save_weights(model, epoch)

            # save sample images
            images_X = images_X[:self.batch_size]
            images_Y = images_Y[:self.batch_size]
            fake_sample_X, fake_sample_Y = self.generate_convert_sample(images_X, images_Y, model)
            self.save_sample(fake_sample_Y, self.domain_X, self.domain_Y, epoch)
            self.save_sample(fake_sample_X, self.domain_Y, self.domain_X, epoch)

            for i, batch in enumerate(test_dataloader):
                images_X, images_Y = self.cast_images(batch)
                fake_sample_X, fake_sample_Y = self.generate_convert_sample(images_X, images_Y, model)
                self.save_sample(fake_sample_Y, self.domain_X, self.domain_Y, epoch, i)
                self.save_sample(fake_sample_X, self.domain_Y, self.domain_X, epoch, i)

            elapsed = time.time() - start_time
            print(f"Epoch {epoch} ends. (total time: {elapsed:.2f})")

            lr_scheduler_G.step()
            lr_scheduler_D_X.step()
            lr_scheduler_D_Y.step()


    def generate_convert_sample(self, images_X, images_Y, model):
        content_X = model.content_enc_X(images_X)
        content_Y = model.content_enc_Y(images_Y)
        style_X = model.style_enc_X(images_X)
        style_Y = model.style_enc_Y(images_Y)
        
        blank = np.zeros((3, self.image_size, self.image_size))

        content_list_X = [content_X[i].unsqueeze(0) for i in range(self.batch_size)]
        content_list_Y = [content_Y[i].unsqueeze(0) for i in range(self.batch_size)]
        style_list_X = [style_X[i].unsqueeze(0) for i in range(self.batch_size)]
        style_list_Y = [style_Y[i].unsqueeze(0) for i in range(self.batch_size)]

        fake_X_list = []
        fake_Y_list = []
        
        for i in range(self.batch_size):
            fake_X_col_list = []
            fake_Y_col_list = []
            for j in range(self.batch_size):
                with torch.no_grad():
                    fake_X = (model.dec_X(content_list_Y[i], style_list_X[j])).squeeze(0)
                    fake_Y = (model.dec_Y(content_list_X[i], style_list_Y[j])).squeeze(0)
                fake_X = fake_X.to("cpu").numpy()
                fake_Y = fake_Y.to("cpu").numpy()

                fake_X_col_list.append(fake_X)
                fake_Y_col_list.append(fake_Y)
            
            fake_X_col = np.concatenate(fake_X_col_list, axis=2)
            fake_Y_col = np.concatenate(fake_Y_col_list, axis=2)
            fake_X_list.append(fake_X_col)
            fake_Y_list.append(fake_Y_col)

        fake_X = np.concatenate(fake_X_list, axis=1)
        fake_Y = np.concatenate(fake_Y_list, axis=1)

        images_X = images_X.to("cpu").numpy()
        images_Y = images_Y.to("cpu").numpy()

        images_X_list = [images_X[i] for i in range(self.batch_size)]
        images_Y_list = [images_Y[i] for i in range(self.batch_size)]

        images_X_columns = np.concatenate(images_X_list, axis=2)
        images_Y_columns = np.concatenate(images_Y_list, axis=2)
        images_X_rows = np.concatenate(images_X_list, axis=1)
        images_Y_rows = np.concatenate(images_Y_list, axis=1)

        fake_X_columns = np.concatenate([blank, images_X_columns], axis=2)
        fake_X_rows = np.concatenate([images_Y_rows, fake_X], axis=2)
        fake_X = np.concatenate([fake_X_columns, fake_X_rows], axis=1)

        fake_Y_columns = np.concatenate([blank, images_Y_columns], axis=2)
        fake_Y_rows = np.concatenate([images_X_rows, fake_Y], axis=2)
        fake_Y = np.concatenate([fake_Y_columns, fake_Y_rows], axis=1)

        fake_X = np.transpose(fake_X, (1, 2, 0))
        fake_Y = np.transpose(fake_Y, (1, 2, 0))

        return fake_X, fake_Y
                
    def save_sample(self, sample, X_name, Y_name, epoch, count=None):
        sample = ((sample + 1) * 127.5).astype(np.uint8)
        if count is None:
            imsave(os.path.join(self.sample_dir, f"{epoch}_{X_name}2{Y_name}.png"), sample)
        else:
            imsave(os.path.join(self.sample_dir, f"{epoch}_{X_name}2{Y_name}_{count}.png"), sample)

    def save_weights(self, model, epoch):
        torch.save(model.content_enc_X.state_dict(), os.path.join(self.weight_dir, f"ContentEncoder_{self.domain_X}_{epoch:04}.pth"))
        torch.save(model.style_enc_X.state_dict(), os.path.join(self.weight_dir, f"StyleEncoder_{self.domain_X}_{epoch:04}.pth"))
        torch.save(model.dec_X.state_dict(), os.path.join(self.weight_dir, f"Decoder_{self.domain_X}_{epoch:04}.pth")) 
        torch.save(model.content_enc_Y.state_dict(), os.path.join(self.weight_dir, f"ContentEncoder_{self.domain_Y}_{epoch:04}.pth"))
        torch.save(model.style_enc_Y.state_dict(), os.path.join(self.weight_dir, f"StyleEncoder_{self.domain_Y}_{epoch:04}.pth"))
        torch.save(model.dec_Y.state_dict(), os.path.join(self.weight_dir, f"Decoder_{self.domain_Y}_{epoch:04}.pth"))      

    def cast_images(self, batch):
        image_X, image_Y = batch
        image_X = image_X.to(self.device)
        image_Y = image_Y.to(self.device)
        return image_X, image_Y       

    def cast_model(self, model):
        model.style_enc_X.to(self.device)
        model.style_enc_Y.to(self.device)
        model.content_enc_X.to(self.device)
        model.content_enc_Y.to(self.device)
        model.dec_X.to(self.device)
        model.dec_Y.to(self.device)
        model.dis_X.to(self.device)
        model.dis_Y.to(self.device)


class AttentionGuidedGANTrainer(BaseTrainer):
    def __init__(self, params):
        super(AttentionGuidedGANTrainer, self).__init__()
        self.lambda_cyc = params["lambda_cyc"]
        self.attention_gan = params["attention_gan"]
        self.attention = params["attention"]
        self.use_idt = params["use_idt"]
        if self.use_idt:
            print("use identity loss")
        else:
            print("not use identity loss")
        self.gen_num_channels = params["gen_num_channels"]
        self.dis_num_channels = params["dis_num_channels"]

    def train(self):
        #model = models.AttentionGuidedGAN(self.num_channels)
        #self.cast_model(model)
        print("Constructed Attention Guided GAN model.")

    def cast_model(self, model):
        pass


class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)              

