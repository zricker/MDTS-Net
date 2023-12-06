import sys
from utils.NiftiDataset import *
import utils.NiftiDataset as NiftiDataset
from torch.utils.data import DataLoader
from options.train_options import TrainOptions
from torch.cuda import amp
# from logger import *
import time
from models import create_model
from utils.visualizer import Visualizer
# from test import inference
import torchvision
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
if __name__ == '__main__':
    cuda = True if torch.cuda.is_available() else False
    scaler = amp.GradScaler(enabled=cuda)  # 自動混合精度
    # writer = SummaryWriter(log_dir='./log/')
    # -----  Loading the init options -----
    opt = TrainOptions().parse()

    # -----  Transformation and Augmentation process for the data  -----
    min_pixel = int(opt.min_pixel * ((opt.patch_size[0] * opt.patch_size[1] * opt.patch_size[2]) / 100))
    trainTransforms = [
                NiftiDataset.Resample(opt.new_resolution, opt.resample),
                NiftiDataset.Augmentation(),
                NiftiDataset.Padding((opt.patch_size[0], opt.patch_size[1], opt.patch_size[2])),
                NiftiDataset.RandomCrop((opt.patch_size[0], opt.patch_size[1], opt.patch_size[2]), opt.drop_ratio, min_pixel),
                ]

    train_set = NifitDataSet(opt.data_path, which_direction='AtoB', transforms=trainTransforms, shuffle_labels=False, train=True)
    parkinson_train_set = NifitDataSet(opt.parkinson_data_path, which_direction='AtoB', transforms=trainTransforms, shuffle_labels=False, train=True)

    print('lenght train list:', len(train_set))
    print('lenght parkinson_train_set list:', len(parkinson_train_set))
    train_loader = DataLoader(train_set, batch_size=opt.batch_size, shuffle=False, num_workers=opt.workers, pin_memory=True)  # Here are then fed to the network with a defined batch size
    parkinson_train_loader = DataLoader(parkinson_train_set, batch_size=opt.batch_size, shuffle=False, num_workers=opt.workers, pin_memory=True)
    # -----------------------------------------------------
    model = create_model(opt)  # creation of the model
    model.setup(opt)
    if opt.epoch_count > 1:
        model.load_networks(opt.epoch_count)
    visualizer = Visualizer(opt)
    total_steps = 0
    # summary(model.netG_A, input_size=(1, 104, 124, 104))
    # summary(model.netG_B, input_size=(1, 104, 124, 104))
    # summary(model.netD_A, input_size=(1, 104, 124, 104))
    # summary(model.netD_B, input_size=(1, 104, 124, 104))
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        for i, data in enumerate(train_loader):
            iter_start_time = time.time()
            if total_steps % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            visualizer.reset()
            total_steps += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)

            model.optimize_parameters()


            if total_steps % opt.print_freq == 0:
                losses = model.get_current_losses()
                t = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t, t_data)

            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                model.save_networks('latest')

            iter_data_time = time.time()

        for i, data in enumerate(parkinson_train_loader):
            iter_start_time = time.time()
            if total_steps % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            visualizer.reset()
            total_steps += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)

            model.optimize_parameters_pd()
            #print(i)

            if total_steps % opt.print_freq == 0:
                losses = model.get_current_losses()
                t = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t, t_data)

            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                model.save_networks('latest')

            iter_data_time = time.time()

        model.update_learning_rate()
        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

        # for name, param in model.netG_A.named_parameters():
        #     if param.grad.size(dim=0) >= 2:
        #         writer.add_histogram(tag=name + '_grad', values=param.grad, global_step=epoch)
        #         writer.add_histogram(tag=name + '_data', values=param.data, global_step=epoch)

        # for name, param in model.netD_A.named_parameters():
        #     if param.grad.size(dim=0) >= 2:
        #         writer.add_histogram(tag=name + '_grad', values=param.grad, global_step=epoch)
        #         writer.add_histogram(tag=name + '_data', values=param.data, global_step=epoch)

    # writer.close()











