import torch
import os
from loguru import logger

def select_device2(id, force_cpu=False):
    cuda = False if force_cpu else torch.cuda.is_available()
    device = torch.device('cuda:{}'.format(id) if cuda else 'cpu')

    if not cuda:
        logger.info('Using CPU')
    if cuda:
        c = 1024 ** 2  # bytes to MB
        ng = torch.cuda.device_count()
        x = [torch.cuda.get_device_properties(i) for i in range(ng)]
        logger.info("Using CUDA device0 _CudaDeviceProperties(name='%s', total_memory=%dMB)" %
              (x[0].name, x[0].total_memory / c))
        if ng > 0:
            # torch.cuda.set_device(0)  # OPTIONAL: Set GPU ID
            for i in range(1, ng):
                logger.info("           device%g _CudaDeviceProperties(name='%s', total_memory=%dMB)" %
                      (i, x[i].name, x[i].total_memory / c))

    return device

def select_device(device='', batch_size=None):
    # device = 'cpu' or '0' or '0,1,2,3'
    cpu_request = type(device) == str and device.lower() == 'cpu'
    if device and not cpu_request:  # if device requested other than 'cpu'
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable
        assert torch.cuda.is_available(), 'CUDA unavailable, invalid device %s requested' % device  # check availablity

    cuda = False if cpu_request else torch.cuda.is_available()
    if cuda:
        c = 1024 ** 2  # bytes to MB
        ng = torch.cuda.device_count()
        if ng > 1 and batch_size:  # check that batch_size is compatible with device_count
            assert batch_size % ng == 0, 'batch-size %g not multiple of GPU count %g' % (batch_size, ng)
        x = [torch.cuda.get_device_properties(i) for i in range(ng)]
        s = f'Using torch {torch.__version__} '
        for i in range(0, ng):
            if i == 1:
                s = ' ' * len(s)
            logger.info("%sCUDA:%g (%s, %dMB)" % (s, i, x[i].name, x[i].total_memory / c))
    else:
        logger.info(f'Using torch {torch.__version__} CPU')

    return torch.device('cuda:0' if cuda else 'cpu')

if __name__ == "__main__":
    _ = select_device(0)