import torch
import itertools
import random
from .base_model import BaseModel
from . import networks3D

scaler = torch.cuda.amp.GradScaler()
import numpy as np

from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM
class ImagePool():
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)
        return return_images


class CycleGANModel(BaseModel):
    def name(self):
        return 'CycleGANModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # default CycleGAN did not use dropout
        parser.set_defaults(no_dropout=True)
        if is_train:
            parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for generators and discriminators')
            parser.add_argument('--lambda_A', type=float, default=5.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=5.0,
                                help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_sup', type=float, default=0, help='supervised loss')
            parser.add_argument('--lambda_sim', type=float, default=1, help='weight for similarity loss')
            parser.add_argument('--lambda_identity', type=float, default=0, help='use identity mapping. Setting lambda_identity other than 0 has an effect of '
                                                                                   'scaling the weight of the identity mapping loss. For example, if the weight of the'
                                                                                   ' identity loss should be 10 times smaller than the weight of the reconstruction loss, '
                                                                                   'please set lambda_identity = 0.1')
            '''
            adjust the weight of correlation coefficient loss
            '''
            parser.add_argument('--lambda_co_A', type=float, default=2,
                                help='weight for correlation coefficient loss (A -> B)')
            parser.add_argument('--lambda_co_B', type=float, default=2,
                                help='weight for correlation coefficient loss (B -> A )')

        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.patch_size = opt.patch_size
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B', 'similarity', 'classificaton', 'supervise']
        # self.loss_names = ['D_A', 'G_A', 'cycle_A', 'cor_coe_GA', 'D_B', 'G_B', 'cycle_B', 'cor_coe_GB']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        visual_names_A = ['real_A', 'fake_B', 'rec_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B']
        if self.isTrain and self.opt.lambda_identity > 0.0:
            visual_names_A.append('idt_A')
            visual_names_B.append('idt_B')

        self.visual_names = visual_names_A + visual_names_B
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']

        # load/define networks
        # The naming conversion is different from those used in the paper
        # Code (paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = networks3D.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,   # nc number channels
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_B = networks3D.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD_A = networks3D.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = networks3D.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks3D.GANLoss(use_lsgan=not opt.no_lsgan).to(self.device)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            self.L1 = torch.nn.L1Loss()
            self.L2 = torch.nn.MSELoss()
            self.I2half = torch.nn.AvgPool3d(2)
            self.I2quarter = torch.nn.AvgPool3d(4)
            self.BCEwithlogt = torch.nn.BCEWithLogitsLoss()
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = []
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        self.real_A = input[0 if AtoB else 1].to(self.device)
        self.real_B = input[1 if AtoB else 0].to(self.device)
        # self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        with torch.cuda.amp.autocast():
            #self.fake_B, self.fake_B_half, self.fake_B_quarter, self.fake_B_cl, self.fake_B_class_feature, self.fake_B_common_feature = self.netG_A(self.real_A)
            #self.rec_A, _, _, _, _, _ = self.netG_B(self.fake_B)

            #self.fake_A, self.fake_A_half, self.fake_A_quarter, self.fake_A_cl, self.fake_A_class_feature, self.fake_A_common_feature  = self.netG_B(self.real_B)
            #self.rec_B, _, _, _, _, _ = self.netG_A(self.fake_A)
            self.fake_B, self.fake_B_cl, self.fake_B_class_feature, self.fake_B_common_feature = self.netG_A(self.real_A)
            self.rec_A, _, _, _ = self.netG_B(self.fake_B)

            self.fake_A, self.fake_A_cl, self.fake_A_class_feature, self.fake_A_common_feature = self.netG_B(self.real_B)
            self.rec_B, _, _, _ = self.netG_A(self.fake_A)

    def backward_D_basic(self, netD, real, fake, fake_half, fake_quarter):
        # Real
        lambda_GAN = self.opt.lambda_GAN
        with torch.cuda.amp.autocast():
            real_half = self.I2half(real)
            real_quarter = self.I2quarter(real)
            #pred_real = netD(real, real_half, real_quarter)
            pred_real = netD(real, None, None)
            loss_D_real = self.criterionGAN(pred_real, True)
            # Fake
            #pred_fake = netD(fake.detach(), fake_half.detach(), fake_quarter.detach())
            pred_fake = netD(fake.detach(), None, None)
            loss_D_fake = self.criterionGAN(pred_fake, False)
            # Combined loss
            loss_D = (loss_D_real + loss_D_fake) * 0.5 * lambda_GAN

        # backward
        # loss_D.backward()
        scaler.scale(loss_D).backward()
        return loss_D

    def backward_D_A(self):
        fake_B = self.fake_B_pool.query(self.fake_B)
        #self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B, self.fake_B_half, self.fake_B_quarter)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B, None, None)

    def backward_D_B(self):
        fake_A = self.fake_A_pool.query(self.fake_A)
        #self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A, self.fake_A_half, self.fake_A_quarter)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A, None, None)

    def backward_G(self):
        lambda_GAN = self.opt.lambda_GAN
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        lambda_sup = self.opt.lambda_sup
        lambda_sim = self.opt.lambda_sim

        with torch.cuda.amp.autocast():
            # Identity loss
            if lambda_idt > 0:
                # G_A should be identity if real_B is fed.
                #self.idt_A, _, _, _, _, _ = self.netG_A(self.real_B)
                self.idt_A, _, _, _ = self.netG_A(self.real_B)
                self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
                # G_B should be identity if real_A is fed.
                #self.idt_B, _, _, _, _, _ = self.netG_B(self.real_A)
                self.idt_B, _, _, _ = self.netG_B(self.real_A)
                self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
            else:
                self.loss_idt_A = 0
                self.loss_idt_B = 0

            # GAN loss D_A(G_A(A))
            #self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B, self.fake_B_half, self.fake_B_quarter), True) * lambda_GAN
            self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B, None, None), True) * lambda_GAN

            # GAN loss D_B(G_B(B))
            #self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A, self.fake_A_half, self.fake_A_quarter), True) * lambda_GAN
            self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A, None, None), True) * lambda_GAN

            # Forward cycle loss
            self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A

            # Backward cycle loss
            self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B

            self.loss_supervise = (self.L1(self.fake_A, self.real_A) + self.L1(self.fake_B, self.real_B)) * lambda_sup

            self.loss_similarity = (self.L2(self.fake_B_common_feature, self.fake_B_class_feature).mean() + self.L2(self.fake_A_common_feature, self.fake_A_class_feature).mean()) * lambda_sim

            self.loss_classificaton = self.BCEwithlogt(self.fake_B_cl, torch.zeros_like(self.fake_B_cl)) * 2
            # ------------------------------------------------------------ ---------------------------
            # ----------------------------------------------------------------------------------------
            # combined loss
            self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A \
                          + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B \
                          + self.loss_similarity + self.loss_classificaton \
                          + self.loss_supervise

            #self.loss_G = self.loss_supervise

        scaler.scale(self.loss_G).backward()


    def backward_G_pd(self):
        lambda_GAN = self.opt.lambda_GAN
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        lambda_sup = self.opt.lambda_sup
        lambda_sim = self.opt.lambda_sim

        with torch.cuda.amp.autocast():
            # Identity loss
            if lambda_idt > 0:
                # G_A should be identity if real_B is fed.
                #self.idt_A, _, _, _, _, _ = self.netG_A(self.real_B)
                self.idt_A, _, _, _ = self.netG_A(self.real_B)
                self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
                # G_B should be identity if real_A is fed.
                #self.idt_B, _, _, _, _, _ = self.netG_B(self.real_A)
                self.idt_B, _, _, _ = self.netG_B(self.real_A)
                self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
            else:
                self.loss_idt_A = 0
                self.loss_idt_B = 0

            # GAN loss D_A(G_A(A))
            #self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B, self.fake_B_half, self.fake_B_quarter), True) * lambda_GAN
            self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B, None, None), True) * lambda_GAN

            # GAN loss D_B(G_B(B))
            #self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A, self.fake_A_half, self.fake_A_quarter), True) * lambda_GAN
            self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A, None, None), True) * lambda_GAN

            # Forward cycle loss
            self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A

            # Backward cycle loss
            self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B

            self.loss_supervise = (self.L1(self.fake_A, self.real_A) + self.L1(self.fake_B, self.real_B)) * lambda_sup

            self.loss_similarity = (self.L2(self.fake_B_common_feature, self.fake_B_class_feature).mean() + self.L2(self.fake_A_common_feature, self.fake_A_class_feature).mean()) * lambda_sim

            self.loss_classificaton = self.BCEwithlogt(self.fake_B_cl, torch.ones_like(self.fake_B_cl)) * 2
            # ------------------------------------------------------------ ---------------------------
            # ----------------------------------------------------------------------------------------
            # combined loss
            self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A \
                          + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B \
                          + self.loss_similarity + self.loss_classificaton \
                          + self.loss_supervise

            #self.loss_G = self.loss_supervise


        scaler.scale(self.loss_G).backward()

    def optimize_parameters(self):
        # forward
        self.forward()
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)
        self.optimizer_G.zero_grad()
        self.backward_G_pd()
        # self.optimizer_G.step()

        scaler.step(self.optimizer_G)

        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()
        self.backward_D_A()
        self.backward_D_B()

        scaler.step(self.optimizer_D)
        scaler.update()

    def optimize_parameters_pd(self):
        # forward
        self.forward()
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)
        self.optimizer_G.zero_grad()
        self.backward_G_pd()

        scaler.step(self.optimizer_G)

        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()
        self.backward_D_A()
        self.backward_D_B()

        scaler.step(self.optimizer_D)
        scaler.update()

