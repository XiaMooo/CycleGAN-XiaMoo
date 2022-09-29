import torch
from options.test_options import TestOptions
from models import create_model


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    device = torch.device(
        f"cuda:{','.join(str(i) for i in opt.gpu_ids)}" if len(opt.gpu_ids) > 1 or opt.gpu_ids[0] >= 0 else "cpu"
    )
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    model = model.netG.module

    torch.jit.save(
        torch.jit.trace(
            model,
            torch.rand(1, opt.input_nc, opt.crop_size, opt.crop_size).to(device)
        ), f"{opt.name}.jit.pt")

    torch.save(model.state_dict(), f"{opt.name}.pt")