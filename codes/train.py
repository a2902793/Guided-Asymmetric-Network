import os.path, math, argparse, random, logging, torch
from timeit import default_timer as timer
import options.options as option
from utils import util
from data import create_dataloader, create_dataset
from models import create_model

# import torch.profiler

def main():
    # options
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, required=False, default='options/train/DualSR_pretrain.json', help='Path to option JSON file.')
    opt = option.parse(parser.parse_args().opt, is_train=True)
    opt = option.dict_to_nonedict(opt)  # Convert to NoneDict, which return None for missing key.

    # train from scratch OR resume training
    if opt['path']['resume_state']:  # resuming training
        resume_state = torch.load(opt['path']['resume_state'])
    else:  # training from scratch
        resume_state = None
        util.mkdir_and_rename(opt['path']['experiments_root'])  # rename old folder if exists
        util.mkdirs((path for key, path in opt['path'].items() if not key == 'experiments_root'
                     and 'pretrain_model' not in key and 'resume' not in key))

    # config loggers. Before it, the log will not work
    util.setup_logger(None, opt['path']['log'], 'train', level=logging.DEBUG, screen=True)
    util.setup_logger('val', opt['path']['log'], 'val', level=logging.DEBUG)
    logger = logging.getLogger('base')

    if resume_state:
        logger.debug('Resuming training from epoch: {}, iter: {}.'.format(
            resume_state['epoch'], resume_state['iter']))
        option.check_resume(opt)  # check resume options

    logger.debug(option.dict2str(opt))
    # tensorboard logger
    # if opt['use_tb_logger'] and 'debug' not in opt['name']:
    #     from tensorboardX import SummaryWriter
    #     tb_logger = SummaryWriter(log_dir='../tb_logger/' + opt['name'])

    # random seed
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    # logger.debug('Random seed: {}'.format(seed))
    util.set_random_seed(seed)

    torch.backends.cudnn.benckmark = True
    # torch.backends.cudnn.deterministic = True

    # create train and val dataloader
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = create_dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['batch_size']))
            logger.debug('Number of train images: {:,d}, iters: {:,d}'.format(
                len(train_set), train_size))
            total_iters = int(opt['train']['niter'])
            total_epochs = int(math.ceil(total_iters / train_size))
            logger.debug('Total epochs needed: {:d} for iters {:,d}'.format(
                total_epochs, total_iters))
            train_loader = create_dataloader(train_set, dataset_opt)
        elif phase == 'val':
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt)
            logger.debug('Number of val images in [{:s}]: {:d}'.format(dataset_opt['name'],
                                                                      len(val_set)))
        else:
            raise NotImplementedError('Phase [{:s}] is not recognized.'.format(phase))
    assert train_loader is not None

    # create model
    model = create_model(opt)

    # resume training
    if resume_state:
        start_epoch = resume_state['epoch']
        current_step = resume_state['iter']
        model.resume_training(resume_state)  # handle optimizers and schedulers
    else:
        current_step = 0
        start_epoch = 0

    logger.info('Start training from epoch: {:d}, iter: {:d}'.format(start_epoch, current_step))

    # with torch.profiler.profile(
    #     schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
    #     on_trace_ready=torch.profiler.tensorboard_trace_handler('./tb_logger'),
    #     record_shapes=True,
    #     with_stack=True
    # ) as prof:
    for epoch in range(start_epoch, total_epochs):
        # start = timer()
        for _, train_data in enumerate(train_loader):
            current_step += 1
            if current_step > total_iters:
                break

            # training
            model.feed_data(train_data)
            model.optimize_parameters(current_step)

            # update learning rate
            model.update_learning_rate()

            # log
            if current_step % opt['logger']['print_freq'] == 0:
                logs = model.get_current_log()
                message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(
                    epoch, current_step, model.get_current_learning_rate())
                for k, v in logs.items():
                    message += '{:s}: {:.4e} '.format(k, v)
                    # tensorboard logger
                    # if opt['use_tb_logger'] and 'debug' not in opt['name']:
                    #     tb_logger.add_scalar(k, v, current_step)
                logger.debug(message)

            # validation
            if current_step % opt['train']['val_freq'] == 0:
                # start = timer()
                avg_psnr_SRCNN = 0.0
                avg_psnr_GAN = 0.0
                avg_psnr = 0.0
                idx = 0
                for val_data in val_loader:
                    idx += 1
                    img_name = os.path.splitext(os.path.basename(val_data['LR_path'][0]))[0]
                    img_dir = os.path.join(opt['path']['val_images'], img_name)
                    util.mkdir(img_dir)

                    model.feed_data(val_data)
                    # model.feed_data2(val_data)
                    model.test()

                    visuals = model.get_current_visuals()
                    SRCNN_img = util.tensor2img(visuals['fake_LF'])  # SRCNN做的
                    GAN_img = util.tensor2img(visuals['fake_HF'])  # GAN做的
                    sr_img = util.tensor2img(visuals['SR'])  # uint8
                    gt_img = util.tensor2img(visuals['HR'])  # uint8

                    # # Save SR images for reference
                    # save_img_path = os.path.join(img_dir, '{:s}_{:d}.png'.format(\
                    #     img_name, current_step))
                    
                    # # util.save_img(SRCNN_img, os.path.join(img_dir, 'SRCNN_{:s}_{:d}.png'.format(img_name, current_step)))
                    # # util.save_img(GAN_img, os.path.join(img_dir, 'GAN_{:s}_{:d}.png'.format(img_name, current_step)))
                    # util.save_img(sr_img, save_img_path)

                    # calculate PSNR
                    crop_size = opt['scale']

                    SRCNN_img = SRCNN_img / 255.
                    GAN_img = GAN_img / 255.
                    gt_img = gt_img / 255.
                    sr_img = sr_img / 255.

                    cropped_SRCNN_img = SRCNN_img[crop_size:-crop_size, crop_size:-crop_size, :]
                    cropped_GAN_img = GAN_img[crop_size:-crop_size, crop_size:-crop_size, :]
                    cropped_sr_img = sr_img[crop_size:-crop_size, crop_size:-crop_size, :]
                    cropped_gt_img = gt_img[crop_size:-crop_size, crop_size:-crop_size, :]
                    
                    avg_psnr_SRCNN += util.calculate_psnr(cropped_SRCNN_img * 255, cropped_gt_img * 255)
                    avg_psnr_GAN += util.calculate_psnr(cropped_GAN_img * 255, cropped_gt_img * 255)
                    avg_psnr += util.calculate_psnr(cropped_sr_img * 255, cropped_gt_img * 255)
                    
                avg_psnr_SRCNN = avg_psnr_SRCNN / idx
                avg_psnr_GAN = avg_psnr_GAN / idx 
                avg_psnr = avg_psnr / idx

                # log
                # logger.debug('# Validation # PSNR: {:.4e} PSNR_SRCNN: {:.4e} PSNR_GAN: {:.4e}'.format(avg_psnr, avg_psnr_SRCNN, avg_psnr_GAN))
                logger.debug('# Validation # PSNR: {:.4e} '.format(avg_psnr))
                logger_val = logging.getLogger('val')  # validation logger
                logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e} psnr_SRCNN: {:.4e} psnr_GAN: {:.4e}'.format(
                    epoch, current_step, avg_psnr, avg_psnr_SRCNN, avg_psnr_GAN))
                # end = timer()
                # logger.info(f'validation: {end - start} seconds')

                # logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e}'.format(
                #     epoch, current_step, avg_psnr))
                # tensorboard logger
                # if opt['use_tb_logger'] and 'debug' not in opt['name']:
                #     tb_logger.add_scalar('psnr', avg_psnr, current_step)
            
            # save models and training states
            if current_step % opt['logger']['save_checkpoint_freq'] == 0:
                logger.debug('Saving models and training states.')
                model.save(current_step)
                model.save_training_state(epoch, current_step)
            # prof.step()
        # end = timer()
        # logger.info(f'epoch({epoch}): {end - start} seconds')

    logger.debug('Saving the final model.')
    model.save('latest')
    logger.debug('End of training.')
    print("End of train.py")


if __name__ == '__main__':
    main()
