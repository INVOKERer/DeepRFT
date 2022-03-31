import numpy as np

def get_parameter_number(net):
    total_num = sum(np.prod(p.size()) for p in net.parameters())
    trainable_num = sum(np.prod(p.size()) for p in net.parameters() if p.requires_grad)
    print('Total: ', total_num)
    print('Trainable: ', trainable_num)


if __name__=='__main__':
    from DeepRFT_MIMO import DeepRFT_flops as Net
    import torch
    from ptflops import get_model_complexity_info
    with torch.cuda.device(0):
        net = Net()
        macs, params = get_model_complexity_info(net, (3, 256, 256), as_strings=True,
                                                 print_per_layer_stat=True, verbose=True)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))
