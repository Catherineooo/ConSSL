train = dict(
    eval_step=1024,
    total_steps=1024*512,
    trainer=dict(
        type='ConSSL',
        threshold=0.95,
        T=1.0,
        temperature=0.07,
        lambda_u=1.0,
        lambda_contrast=1.0,
        lambda_sim=0.4,
        contrast_with_softlabel=True,
        loss_x=dict(type='cross_entropy', reduction='mean'),
        loss_u=dict(type='cross_entropy', reduction='none')))
num_classes = 100
seed = 1
model = dict(
    type='wideresnet',
    depth=28,
    widen_factor=8,
    dropout=0,
    num_classes=100,
    proj=True)
cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)
data = dict(
    type='CIFAR100SSL',
    num_workers=4,
    num_labeled=10000,
    num_classes=100,
    batch_size=16,
    expand_labels=False,
    mu=7,
    root='./data/CIFAR',
    lpipelines=[[{
        'type': 'RandomHorizontalFlip'
    }, {
        'type': 'RandomCrop',
        'size': 32,
        'padding': 4,
        'padding_mode': 'reflect'
    }, {
        'type': 'ToTensor'
    }, {
        'type': 'Normalize',
        'mean': (0.5071, 0.4867, 0.4408),
        'std': (0.2675, 0.2565, 0.2761)
    }]],
    upipelinse=[[{
        'type': 'RandomHorizontalFlip'
    }, {
        'type': 'RandomCrop',
        'size': 32,
        'padding': 4,
        'padding_mode': 'reflect'
    }, {
        'type': 'ToTensor'
    }, {
        'type': 'Normalize',
        'mean': (0.5071, 0.4867, 0.4408),
        'std': (0.2675, 0.2565, 0.2761)
    }],
                [{
                    'type': 'RandomHorizontalFlip'
                }, {
                    'type': 'RandomCrop',
                    'size': 32,
                    'padding': 4,
                    'padding_mode': 'reflect'
                }, {
                    'type': 'RandAugmentMC',
                    'n': 2,
                    'm': 10
                }, {
                    'type': 'ToTensor'
                }, {
                    'type': 'Normalize',
                    'mean': (0.5071, 0.4867, 0.4408),
                    'std': (0.2675, 0.2565, 0.2761)
                }],
                [{
                    'type': 'RandomResizedCrop',
                    'size': 32
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
        dict(type='ToTensor'),
        dict(
            type='Normalize',
            mean=(0.5071, 0.4867, 0.4408),
            std=(0.2675, 0.2565, 0.2761))
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
