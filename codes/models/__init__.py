import logging
logger = logging.getLogger('base')


def create_model(opt):
    model = opt['model']
    if model == 'DualSR':
        from .DualSR_model import DualSRModel as M
    elif model == 'DualSR_pretrain':
        from .DualSR_pretrain import DualSR_pretrain as M
    else:
        raise NotImplementedError('Model [{:s}] not recognized.'.format(model))
    m = M(opt)
    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m
