CONFIGS_ = {
    # input_size,hidden_size,output_size,num_layers,dropout_prob,bidirectional,MLP_layers,encoder_type,property_size
    'vendor-purchased-2019':(512,256,512,2,0.2,True,3,"transformer",26),
    'Twibot-20':(512,256,512,2,0.2,True,3,"transformer",26),
    # layers,heads,hidden_size,ff_size,dropout
    'transformer_config':(2,8,512,512,0.2),
    # num_convs,kernel_size,stride,hiddensize,outsize=numberkernels,drop_out,sqe_len
    'cnn_config':(4,[2,3,4,5],2,512,2,0.2,50),
}

# temporary roundabout to evaluate sensitivity of the generator
GENERATORCONFIGS = {
    # hidden_dimension, latent_dimension, input_channel, n_class, noise_dim
    'vendor-purchased-2019':(256,512,2,64),
    'Twibot-20':(256,512,2,64),
}



RUNCONFIGS = {
    'vendor-purchased-2019':
        {
            'ensemble_lr': 1e-2,
            'ensemble_batch_size': 64,
            'ensemble_epochs': 300,
            'num_pretrain_iters': 20,
            'property_size':26,
            'train_teacher_epochs':20,
            'ensemble_alpha': 1,    # teacher loss (server side)
            'ensemble_beta': 1,     # adversarial student loss
            'ensemble_eta': 1,      # diversity loss
            'unique_labels': 2,    # available labels
            'generative_alpha': 10, # used to regulate user training
            'generative_beta': 10, # used to regulate user training
            'weight_decay': 1e-2
        },
        'Twibot-20':
        {
            'ensemble_lr': 1e-2,
            'ensemble_batch_size': 64,
            'ensemble_epochs': 300,
            'num_pretrain_iters': 20,
            'train_teacher_epochs':20,
            'property_size':26,
            'ensemble_alpha': 1,    # teacher loss (server side)
            'ensemble_beta': 1,     # adversarial student loss
            'ensemble_eta': 1,      # diversity loss
            'unique_labels': 2,    # available labels
            'generative_alpha': 10, # used to regulate user training
            'generative_beta': 10, # used to regulate user training
            'weight_decay': 1e-2
        },
}

