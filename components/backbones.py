def get_backbone(name, n_way, h, w):
    if name.lower() == 'conv32':
        h, w = h // 2 // 2 // 2 // 2, w // 2 // 2 // 2 // 2
        config = [
            ('conv2d', [32, 3, 3, 3, 1, 1]),
            ('relu', [True]),
            ('bn', [32]),
            ('max_pool2d', [2, 2, 0]),
            ('conv2d', [32, 32, 3, 3, 1, 1]),
            ('relu', [True]),
            ('bn', [32]),
            ('max_pool2d', [2, 2, 0]),
            ('conv2d', [32, 32, 3, 3, 1, 1]),
            ('relu', [True]),
            ('bn', [32]),
            ('max_pool2d', [2, 2, 0]),
            ('conv2d', [32, 32, 3, 3, 1, 1]),
            ('relu', [True]),
            ('bn', [32]),
            ('max_pool2d', [2, 2, 0]),
            ('flatten', []),
            ('linear', [n_way, 32 * h * w])
        ]
    elif name.lower() == 'conv32_br':
        h, w = h // 2 // 2 // 2 // 2, w // 2 // 2 // 2 // 2
        config = [
            # ('conv2d', [32, 3, 3, 3, 1, 1]),
            # ('bn', [32]),
            # ('relu', [True]),
            # ('max_pool2d', [2, 2, 0]),
            # ('conv2d', [32, 32, 3, 3, 1, 1]),
            # ('bn', [32]),
            # ('relu', [True]),
            # ('max_pool2d', [2, 2, 0]),
            # ('conv2d', [32, 32, 3, 3, 1, 1]),
            # ('bn', [32]),
            # ('relu', [True]),
            # ('max_pool2d', [2, 2, 0]),
            # ('conv2d', [32, 32, 3, 3, 1, 1]),
            # ('bn', [32]),
            # ('relu', [True]),
            # ('max_pool2d', [2, 2, 0]),
            # ('flatten', []),
            # ('linear', [n_way, 32 * h * w])
            ('conv2d', [32, 3, 3, 3, 1, 1]),
            ('bn', [32]),
            ('max_pool2d', [2, 2, 0]),
            ('relu', [True]),
            ('conv2d', [32, 32, 3, 3, 1, 1]),
            ('bn', [32]),
            ('max_pool2d', [2, 2, 0]),
            ('relu', [True]),
            ('conv2d', [32, 32, 3, 3, 1, 1]),
            ('bn', [32]),
            ('max_pool2d', [2, 2, 0]),
            ('relu', [True]),
            ('conv2d', [32, 32, 3, 3, 1, 1]),
            ('bn', [32]),
            ('max_pool2d', [2, 2, 0]),
            ('relu', [True]),
            ('flatten', []),
            ('linear', [n_way, 32 * h * w])
        ]
    elif name.lower() == 'conv64_br':
        h, w = h // 2 // 2 // 2 // 2, w // 2 // 2 // 2 // 2
        config = [
            ('conv2d', [64, 3, 3, 3, 1, 1]),
            ('bn', [64]),
            ('max_pool2d', [2, 2, 0]),
            ('relu', [True]),
            ('conv2d', [64, 64, 3, 3, 1, 1]),
            ('bn', [64]),
            ('max_pool2d', [2, 2, 0]),
            ('relu', [True]),
            ('conv2d', [64, 64, 3, 3, 1, 1]),
            ('bn', [64]),
            ('max_pool2d', [2, 2, 0]),
            ('relu', [True]),
            ('conv2d', [64, 64, 3, 3, 1, 1]),
            ('bn', [64]),
            ('max_pool2d', [2, 2, 0]),
            ('relu', [True]),
            ('flatten', []),
            ('linear', [n_way, 64 * h * w])
        ]
    elif name.lower() == 'resnet12':
        h, w = h // 2 // 2 // 2 // 2, w // 2 // 2 // 2 // 2
        config = [
            # [84,84,3]
            ('identity_in', [64, 3, 1, 1, 1, 0]),
            ('conv2d', [64, 3, 3, 3, 1, 1]),
            ('bn', [64]),
            ('leakyrelu', [True]),
            ('conv2d', [64, 64, 3, 3, 1, 1]),
            ('bn', [64]),
            ('leakyrelu', [True]),
            ('conv2d', [64, 64, 3, 3, 1, 1]),
            ('bn', [64]),
            # [84,84,64]
            ('identity_out', []),
            ('leakyrelu', [True]),
            ('max_pool2d', [2, 2, 0]),

            # [42,42,64]
            ('identity_in', [128, 64, 1, 1, 1, 0]),
            ('conv2d', [128, 64, 3, 3, 1, 1]),
            ('bn', [128]),
            ('leakyrelu', [True]),
            ('conv2d', [128, 128, 3, 3, 1, 1]),
            ('bn', [128]),
            ('leakyrelu', [True]),
            ('conv2d', [128, 128, 3, 3, 1, 1]),
            ('bn', [128]),
            # [42,42,128]
            ('identity_out', []),
            ('leakyrelu', [True]),
            ('max_pool2d', [2, 2, 0]),

            # [21,21,128]
            ('identity_in', [256, 128, 1, 1, 1, 0]),
            ('conv2d', [256, 128, 3, 3, 1, 1]),
            ('bn', [256]),
            ('leakyrelu', [True]),
            ('conv2d', [256, 256, 3, 3, 1, 1]),
            ('bn', [256]),
            ('leakyrelu', [True]),
            ('conv2d', [256, 256, 3, 3, 1, 1]),
            ('bn', [256]),
            # [21,21,256]
            ('identity_out', []),
            ('leakyrelu', [True]),
            ('max_pool2d', [2, 2, 0]),

            # [10,10,256]
            ('identity_in', [512, 256, 1, 1, 1, 0]),
            ('conv2d', [512, 256, 3, 3, 1, 1]),
            ('bn', [512]),
            ('leakyrelu', [True]),
            ('conv2d', [512, 512, 3, 3, 1, 1]),
            ('bn', [512]),
            ('leakyrelu', [True]),
            ('conv2d', [512, 512, 3, 3, 1, 1]),
            ('bn', [512]),
            # [10,10,512]
            ('identity_out', []),
            ('leakyrelu', [True]),
            ('max_pool2d', [2, 2, 0]),

            # [5,5,512]
            ('flatten', []),
            ('linear', [n_way, 512 * h * w])
        ]
    return config
