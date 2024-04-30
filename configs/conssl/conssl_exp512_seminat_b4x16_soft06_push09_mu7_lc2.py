train = dict(
    eval_step=1024,
    total_steps=1024*512,
    trainer=dict(
        type='ConSSL',
        threshold=0.6,
        T=1.0,
        temperature=0.07,
        lambda_u=1.0,
        lambda_contrast=2.0,
        lambda_sim=1.5,
        contrast_with_softlabel=True,
        contrast_left_out=True,
        contrast_with_thresh=0.9,
        loss_x=dict(type='cross_entropy', reduction='mean'),
        loss_u=dict(type='cross_entropy', reduction='none')))
num_classes = 810
seed = 1
model = dict(
    type='resnet18',
    low_dim=64,
    num_class=810,
    proj=True,
    width=1,
    in_channel=3)
seminat_mean = [0.4732, 0.4828, 0.3779]
seminat_std = [0.2348, 0.2243, 0.2408]
data = dict(
    type='TxtDatasetSSL',
    num_workers=5,
    batch_size=16,
    l_anno_file='./data/semi-inat2021/l_train/anno.txt',
    u_anno_file='./data/semi-inat2021/u_train/u_train.txt',
    v_anno_file='./data/semi-inat2021/val/anno.txt',
    mu=7,
    lpipelines=[[{
        'type': 'RandomHorizontalFlip',
        'p': 0.5
    }, {
        'type': 'RandomResizedCrop',
        'size': 224,
        'scale': (0.2, 1.0)
    }, {
        'type': 'ToTensor'
    }, {
        'type': 'Normalize',
        'mean': [0.4732, 0.4828, 0.3779],
        'std': [0.2348, 0.2243, 0.2408]
    }]],
    upipelinse=[[{
        'type': 'RandomHorizontalFlip'
    }, {
        'type': 'Resize',
        'size': 256
    }, {
        'type': 'CenterCrop',
        'size': 224
    }, {
        'type': 'ToTensor'
    }, {
        'type': 'Normalize',
        'mean': [0.4732, 0.4828, 0.3779],
        'std': [0.2348, 0.2243, 0.2408]
    }],
                [{
                    'type': 'RandomHorizontalFlip'
                }, {
                    'type': 'RandomResizedCrop',
                    'size': 224,
                    'scale': (0.2, 1.0)
                }, {
                    'type': 'RandAugmentMC',
                    'n': 2,
                    'm': 10
                }, {
                    'type': 'ToTensor'
                }, {
                    'type': 'Normalize',
                    'mean': [0.4732, 0.4828, 0.3779],
                    'std': [0.2348, 0.2243, 0.2408]
                }],
                [{
                    'type': 'RandomResizedCrop',
                    'size': 224,
                    'scale': (0.2, 1.0)
                }, {
                    'type': 'RandomHorizontalFlip'
                }, {
                    'type':
                    'RandomApply',
                    'transforms': [{
                        'type': 'ColorJitter',
                        'brightness': 0.4,
                        'contrast': 0.4,
                        'saturation': 0.4,
                        'hue': 0.1
                    }],
                    'p':
                    0.8
                }, {
                    'type': 'RandomGrayscale',
                    'p': 0.2
                }, {
                    'type': 'ToTensor'
                }]],
    vpipeline=[
        dict(type='Resize', size=256),
        dict(type='CenterCrop', size=224),
        dict(type='ToTensor'),
        dict(
            type='Normalize',
            mean=[0.4732, 0.4828, 0.3779],
            std=[0.2348, 0.2243, 0.2408])
    ],
    eval_step=1024)
scheduler = dict(
    type='cosine_schedule_with_warmup',
    num_warmup_steps=0,
    num_training_steps=524288)
ema = dict(use=True, pseudo_with_ema=False, decay=0.999)
amp = dict(use=False, opt_level='O1')
log = dict(interval=50)
ckpt = dict(interval=1)
evaluation = dict(eval_both=True)
optimizer = dict(
    type='SGD', lr=0.03, momentum=0.9, weight_decay=0.001, nesterov=True)
resume = ''
