from collections import OrderedDict
from .res50 import *
from.vgg19 import *
from .i3d import *
import torch 

def freeze_bn(net):
    for module in net.modules():
        if isinstance(module, torch.nn.modules.BatchNorm2d):
            for i in module.parameters():
                i.requires_grad = False
def release_bn(net):
    for module in net.modules():
        if isinstance(module, torch.nn.modules.BatchNorm2d):
            for i in module.parameters():
                i.requires_grad = True

def purge_classifier(state):
    del state["logits.conv3d.weight"]
    del state["logits.conv3d.bias"]
    return state

def init_model(cfg):
    num_classes =cfg.num_classes
    if cfg.extra:
        num_classes +=1
    if cfg.model=='res50':
        model = Res50(num_classes, bottleneck=cfg.bottleneck, extra=cfg.extra).cuda()
    elif cfg.model =='vgg19':
        model =VGG19(num_classes, bottleneck=cfg.bottleneck, extra=cfg.extra).cuda()
    elif cfg.model =='i3d':
        model = InceptionI3d(in_channels=3)

        if cfg.backbone_pretrain == "kinetics":
            model.load_state_dict(
                purge_classifier(
                    torch.load(
                        "../pretrained_models/i3d_rgb_imagenet+kinetics.pt"
                    )
                )
            )

    if cfg.fix_bn:
        freeze_bn(model)
    else:
        release_bn(model)


    if cfg.init_weight != 'None':
        params = torch.load(cfg.init_weight)
        print('Model restored with weights from : {}'.format(cfg.init_weight))
        try:
            model.load_state_dict(params, strict=True)

        except Exception as e:
            temp = OrderedDict()
            for k,v in params.items():
                name = k[7:]
                temp[name] = v
            model.load_state_dict(temp)

    if cfg.multi_gpu:
        model = nn.DataParallel(model)
    if cfg.train:
        model = model.train().cuda()
        print('Mode --> Train')
    else:
        model = model.eval().cuda()
        print('Mode --> Eval')
    return model

