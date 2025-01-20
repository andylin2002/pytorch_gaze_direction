import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from src.archs import Discriminator, Generator, vgg_16
from src.data_loader import ImageData
from torch.utils.data import TensorDataset, DataLoader

from torchvision.utils import save_image

from utils.ops import l1_loss, content_loss, style_loss
from utils.eyes_catch import eyes_catch
from utils.paste import paste

from tqdm import tqdm
from itertools import cycle

#Learning Rate
D_LR = 5e-5
G_LR = 2e-4

#D's hyperparameter
D_ADV_WEIGHT = 1
D_REG_WEIGHT = 2

#G's hyperparameter
G_ADV_WEIGHT = 1
G_REG_WEIGHT = 2
G_RECON_WEIGHT = 1
G_FEAT_WEIGHT = 1500

AUGMENT = True

class Model(nn.Module):
    def __init__(self, params):
        super(Model, self).__init__()
        self.generator = Generator(style_dim=2)
        self.discriminator = Discriminator(params)

        if params.vgg_path:
            vgg_dict = torch.load(params.vgg_path, map_location=torch.device('cpu'), weights_only=True)
            self.vgg_dict = vgg_dict
            for param in self.vgg_dict.values():
                param.requires_grad = False
            print(f"Successfully loaded pretrained weights from {params.vgg_path}")

        self.params = params
        self.global_step = torch.tensor(0, dtype=torch.int32, requires_grad=False)
        self.lr = params.lr

        def init_weights(m):
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight) # Xavier initialization
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias) # Initialize bias to 0

        self.generator.apply(init_weights)
        self.discriminator.apply(init_weights)

    def data_loader(self):
        
        hps = self.params

        image_data_class = ImageData(load_size=hps.image_size,
                                     channels=3,
                                     data_path=hps.data_path,
                                     ids=hps.ids)
        image_data_class.preprocess()

        train_dataset_num = len(image_data_class.train_images)
        test_dataset_num = len(image_data_class.test_images)

        ########## train_data ##########
        train_images = []
        train_angles_r = []
        train_labels = []
        train_images_t = []
        train_angles_g = []

        tqdm.write(f"train dataset number: {train_dataset_num}")
        for each in tqdm(range(train_dataset_num)):
            image_data_class.train_images[each], image_data_class.train_angles_r[each], image_data_class.train_labels[each], image_data_class.train_images_t[each], image_data_class.train_angles_g[each] = image_data_class.image_processing(
                image_data_class.train_images[each],
                image_data_class.train_angles_r[each],
                image_data_class.train_labels[each],
                image_data_class.train_images_t[each],
                image_data_class.train_angles_g[each],
                augment=AUGMENT
            )

        train_images = torch.stack(image_data_class.train_images) if isinstance(image_data_class.train_images[0], torch.Tensor) else torch.tensor(image_data_class.train_images, dtype=torch.float32)
        train_angles_r = torch.tensor(image_data_class.train_angles_r, dtype=torch.float32)
        train_labels = torch.tensor(image_data_class.train_labels, dtype=torch.float32)
        train_images_t = torch.stack(image_data_class.train_images_t) if isinstance(image_data_class.train_images_t[0], torch.Tensor) else torch.tensor(image_data_class.train_images_t, dtype=torch.float32)
        train_angles_g = torch.tensor(image_data_class.train_angles_g, dtype=torch.float32)

        train_dataset = TensorDataset(
            train_images,
            train_angles_r,
            train_labels,
            train_images_t,
            train_angles_g
        )

        ########## test_data ##########
        test_images = []
        test_angles_r = []
        test_labels = []
        test_images_t = []
        test_angles_g = []

        tqdm.write(f"test dataset number: {test_dataset_num}")
        for each in tqdm(range(test_dataset_num)):
            image_data_class.test_images[each], image_data_class.test_angles_r[each], image_data_class.test_labels[each], image_data_class.test_images_t[each], image_data_class.test_angles_g[each] = image_data_class.image_processing(
                image_data_class.test_images[each],
                image_data_class.test_angles_r[each],
                image_data_class.test_labels[each],
                image_data_class.test_images_t[each],
                image_data_class.test_angles_g[each],
                augment=False
            )

        test_images = torch.stack(image_data_class.test_images) if isinstance(image_data_class.test_images[0], torch.Tensor) else torch.tensor(image_data_class.test_images, dtype=torch.float32)
        test_angles_r = torch.tensor(image_data_class.test_angles_r, dtype=torch.float32)
        test_labels = torch.tensor(image_data_class.test_labels, dtype=torch.float32)
        test_images_t = torch.stack(image_data_class.test_images_t) if isinstance(image_data_class.test_images_t[0], torch.Tensor) else torch.tensor(image_data_class.test_images_t, dtype=torch.float32)
        test_angles_g = torch.tensor(image_data_class.test_angles_g, dtype=torch.float32)

        test_dataset = TensorDataset(
            test_images,
            test_angles_r,
            test_labels,
            test_images_t,
            test_angles_g
        )
        
        train_loader = DataLoader(train_dataset, batch_size=hps.batch_size, shuffle=True, num_workers=8, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=hps.batch_size, shuffle=False, num_workers=8, drop_last=True)

        return train_loader, test_loader, train_dataset_num
    
    def adv_loss(self, images_r, images_g):

        hps = self.params

        # Discriminator output for real and generated samples
        gan_real, reg_real = self.discriminator(images_r)
        gan_fake, reg_fake = self.discriminator(images_g)

        # Generate interpolated samples
        eps = torch.rand((hps.batch_size, 1, 1, 1), device=images_r.device)
        interpolated = eps * images_r + (1. - eps) * images_g
        interpolated = interpolated.requires_grad_()

        gan_inter, _ = self.discriminator(interpolated)

        # Compute gradient penalty
        grad = torch.autograd.grad(
            outputs=gan_inter,
            inputs=interpolated,
            grad_outputs=torch.ones_like(gan_inter),
            create_graph=True,
            retain_graph=True
        )[0]

        slopes = torch.sqrt(torch.sum(grad**2, dim=[1, 2, 3]))
        gp = torch.mean((slopes - 1)**2)

        # Discriminator Loss
        adv_d_loss = (-torch.mean(gan_real) +
                torch.mean(gan_fake) +
                10.0 * gp)

        # Generator Loss
        adv_g_loss = -torch.mean(gan_fake)

        # Regression Loss
        reg_d_loss = F.mse_loss(self.angles_r, reg_real)
        reg_g_loss = F.mse_loss(self.angles_g, reg_fake)

        return adv_d_loss, adv_g_loss, reg_d_loss, reg_g_loss, gp

    def feat_loss(self, image_g, image_t):
        
        # Define VGG model and required layer names
        content_layers = ["conv5"]  # conv5_3
        style_layers = [
            "conv1",   # conv1_2
            "conv2",   # conv2_2
            "conv3",  # conv3_3
            "conv4"   # conv4_3
        ]

        _, end_points_image_g = vgg_16(self, image_g)
        _, end_points_image_t = vgg_16(self, image_t)

        c_loss = content_loss(end_points_image_g, end_points_image_t, content_layers)
        s_loss = style_loss(end_points_image_g, end_points_image_t, style_layers)

        return c_loss, s_loss

    def d_loss_calculator(self, images_r, angles_g):

        images_g = self.generator(images_r, angles_g)

        # regression losses and adversarial losses
        (self.adv_d_loss, self.adv_g_loss, self.reg_d_loss,
        self.reg_g_loss, self.gp) = self.adv_loss(images_r, images_g)

        if (self.it + 1) % 600 == 0:
            print(f"adv_d_loss = {self.adv_d_loss:<10.5f}")
            print(f"reg_d_loss = {self.reg_d_loss:<10.5f}")
            print(" ")


        return D_ADV_WEIGHT * self.adv_d_loss + D_REG_WEIGHT * self.reg_d_loss

    def g_loss_calculator(self, images_r, angles_r, images_t, angles_g):

        images_g = self.generator(images_r, angles_g)

        images_recon = self.generator(images_g, angles_r)

        # reconstruction loss
        self.recon_loss = l1_loss(images_r, images_recon)

        # content loss and style loss
        self.c_loss, self.s_loss = self.feat_loss(images_g, images_t)

        # regression losses and adversarial losses
        (self.adv_d_loss, self.adv_g_loss, self.reg_d_loss,
        self.reg_g_loss, self.gp) = self.adv_loss(images_r, images_g)

        if (self.it + 1) % 600 == 0:
            print(f"adv_g_loss = {self.adv_g_loss:<10.5f}")
            print(f"reg_g_loss = {self.reg_g_loss:<10.5f}")
            print(f"recon_loss = {self.recon_loss:<10.5f}")
            print(f"feat_loss = {self.s_loss + self.c_loss:<10.5f}")
            print(" ")

        return G_ADV_WEIGHT * self.adv_g_loss + G_REG_WEIGHT * self.reg_g_loss + G_RECON_WEIGHT * self.recon_loss + \
                                        G_FEAT_WEIGHT * (self.s_loss + self.c_loss)
    
    def optimizer(self, model, lr):

        hps = self.params

        if hps.optimizer == 'sgd':
            return optim.SGD(model.parameters(), lr=lr)
        if hps.optimizer == 'adam':
            return optim.Adam(model.parameters(),
                            lr=lr,
                            betas=(hps.adam_beta1, hps.adam_beta2),
                            weight_decay=1e-3)
        raise AttributeError("attribute 'optimizer' is not assigned!")

    def add_optimizer(self):

        g_op = self.optimizer(self.generator, lr=G_LR)
        d_op = self.optimizer(self.discriminator, lr=D_LR)

        return d_op, g_op

    def train(self, conti = False):

        hps = self.params

        if torch.cuda.is_available():
            device = torch.device("cuda")
            torch.backends.cudnn.benchmark = True
        else:
            device = torch.device("cpu")
        print(f"Using device: {device}")
        self.device = device

        # to device
        self.generator = self.generator.to(device)
        self.discriminator = self.discriminator.to(device)

        self.d_op, self.g_op = self.add_optimizer()
        self.d_op.zero_grad()
        self.g_op.zero_grad()

        current_model_path = os.path.join(hps.log_dir, "current_model.ckpt")
        if conti:
            checkpoint = torch.load(current_model_path)
            self.generator.load_state_dict(checkpoint['current_generator'])
            self.discriminator.load_state_dict(checkpoint['current_discriminator'])
            self.g_op.load_state_dict(checkpoint['optimizer_generator'])
            self.d_op.load_state_dict(checkpoint['optimizer_discriminator'])
            print(f". ݁₊ ⊹ . ݁ ⟡ ݁ . ⊹ ₊ ݁.Loaded previous model. ݁₊ ⊹ . ݁ ⟡ ݁ . ⊹ ₊ ݁.")

        best_model_loss = float('inf')

        (train_iter, test_iter, train_size) = self.data_loader() # Load iterators for training, validation, and test datasets

        num_epoch = hps.epochs
        self.num_epoch = num_epoch
        batch_size = hps.batch_size
        frequency = 3
        num_iter = train_size // batch_size # num_iter = 1102

        d_op_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.d_op, mode='min', factor=0.95, patience=3, verbose=True)
        g_op_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.g_op, mode='min', factor=0.95, patience=3, verbose=True)
        
        try:
            for epoch in range(num_epoch):
                self.epoch = epoch

                for it, (train_batch, test_batch) in enumerate(tqdm(zip(train_iter, cycle(test_iter)), total=num_iter, desc=f"Epoch: {epoch + 1}/{num_epoch}")):
                    self.it = it
                    transformed_d_test_loss = float('inf')
                    transformed_g_test_loss = float('inf')

                    train_batch = [t.to(device) for t in train_batch]
                    test_batch = [t.to(device) for t in test_batch]

                    # Unpack training data
                    self.x_r, self.angles_r, self.labels, self.x_t, self.angles_g = train_batch
                    '''
                    self.x_r: torch.Size([32, 3, 64, 64]),
                    self.angles_r: torch.Size([32, 2]),
                    self.labels: torch.Size([32]),
                    self.x_t: torch.Size([32, 3, 64, 64]),
                    self.angles_g: torch.Size([32, 2])
                    '''

                    # Unpack test data
                    self.x_test_r, self.angles_test_r, self.labels_test, self.x_test_t, self.angles_test_g = test_batch
                    '''
                    self.x_test_r: torch.Size([32, 3, 64, 64]),
                    self.angles_test_r: torch.Size([32, 2]),
                    self.labels_test: torch.Size([32]),
                    self.x_test_t: torch.Size([32, 3, 64, 64]),
                    self.angles_test_g: torch.Size([32, 2])
                    '''

                    # Training Discriminator
                    self.d_op.zero_grad()
                    self.d_loss_calculator(self.x_r, self.angles_g).backward()

                    if (it + 1) % frequency == 0:
                        self.d_op.step()
                        
                        '''Train Generator (execute every frequency steps)'''
                        self.g_op.zero_grad()

                        self.g_loss_calculator(self.x_r, self.angles_r, self.x_t, self.angles_g).backward()
                        self.g_op.step()

                    del train_batch, test_batch
                    torch.cuda.empty_cache()

                    # Record summaries and save model
                    if (it + 1) % hps.summary_steps == 0:
                        self.global_step = epoch * num_iter + it

                        d_test_loss = self.d_loss_calculator(self.x_test_r, self.angles_test_g).to(device)
                        transformed_d_test_loss = torch.exp(d_test_loss / 100).to(device)
                        g_test_loss = self.g_loss_calculator(self.x_test_r, self.angles_test_r, self.x_test_t, self.angles_test_g).to(device)
                        transformed_g_test_loss = torch.exp(g_test_loss / 100).to(device)
                        tqdm.write(f"generator test loss: {transformed_g_test_loss:<10.2f}, discriminator test loss: {transformed_d_test_loss:<10.2f}")

                        # Define metrics to evaluate model performance
                        challenger_loss = (transformed_g_test_loss + transformed_d_test_loss * (transformed_g_test_loss + transformed_d_test_loss)) # g + d * (g + d)
                        if challenger_loss < best_model_loss:
                            best_model_loss = challenger_loss
                            tqdm.write(f". ݁₊ ⊹ . ݁ ⟡ ݁ . ⊹ ₊ ݁.New lowest loss at step: {self.global_step}. ݁₊ ⊹ . ݁ ⟡ ݁ . ⊹ ₊ ݁.")

                        # Save model weights
                        torch.save({
                                'current_generator': self.generator.state_dict(),
                                'current_discriminator': self.discriminator.state_dict(),
                                'optimizer_generator': self.g_op.state_dict(),
                                'optimizer_discriminator': self.d_op.state_dict()
                            }, current_model_path)

                        del d_test_loss, transformed_d_test_loss, g_test_loss, transformed_g_test_loss, challenger_loss
                        torch.cuda.empty_cache()

                # Adjust learning rate at the end of each epoch
                d_op_scheduler.step(transformed_d_test_loss)
                g_op_scheduler.step(transformed_g_test_loss)

        except KeyboardInterrupt:
            print("Training interrupted.")

    def eval(self):

        hps = self.params

        checkpoint_path = os.path.join(hps.log_dir, 'current_model.ckpt')
        checkpoint = torch.load(checkpoint_path)
        self.generator.load_state_dict(checkpoint['current_generator'])
        self.discriminator.load_state_dict(checkpoint['current_discriminator'])

        eval_dir = os.path.join(hps.log_dir, 'eval')
        os.makedirs(eval_dir, exist_ok=True)
        
        for file_name in os.listdir(hps.client_pictures_dir): # for each picture
            each_eval_dir = os.path.join(eval_dir, f'eval_{file_name}')
            os.makedirs(each_eval_dir, exist_ok=True)

            picture_eyes_patch, eyes_position, size = eyes_catch(hps, file_name) # picture_eyes_patch.shape = [eyes' number, 3, 64, 64]

            '''========== Gaze Adjustment =========='''
            custom_values = torch.tensor([0, 0])
            gaze_angles = torch.zeros(len(picture_eyes_patch), 2)
            gaze_angles[:] = custom_values
            '''========== Gaze Adjustment =========='''

            with torch.no_grad():
                generated_image = self.generator(picture_eyes_patch, gaze_angles) # generated_image.shape = [number, 3, 64, 64]
                
                print(file_name)
                test_gan, test_reg = self.discriminator(picture_eyes_patch)
                print(f"original: gan = {test_gan}, reg = {test_reg}")
                test_gan, test_reg = self.discriminator(generated_image)
                print(f"generated: gan = {test_gan}, reg = {test_reg}")
            
            paste(hps, file_name, generated_image, eyes_position, size)
            
            for i, new_img in enumerate(generated_image):
                if i % 2 == 0: # left eye
                    file_name = os.path.join(each_eval_dir, f'generated_{i // 2}_L.png')
                    save_image(new_img, file_name)
                else: # right eye
                    file_name = os.path.join(each_eval_dir, f'generated_{i // 2}_R.png')
                    save_image(new_img, file_name)

        print("Evaluation finished.")
        