import logging
logger = logging.getLogger('base')


def create_model(opt):
    model = opt['model']
    if model == 'DualSR':
        from .DualSR_model import DualSRModel as M
    elif model == 'DualSR_pretrain':
        from .DualSR_pretrain import DualSR_pretrain as M
    else:
        raise NotImplementedError(f'Model [{model}] not recognized.')
    m = M(opt)
    logger.info(f'Model [{m.__class__.__name__}] is created.')
    return m
