# FedACK
The code for paper "Cross Platforms Linguals and Models Social Bot Detection via Federated Adversarial Contrastive Knowledge Distillation." on Web Conference 2023 (WWW).

# Federated learning module
The code implementation of the federated learning module is stored in the FLAgorithms directory.
The model directory stored the backbone model design for social bot detection.
The implementation of server-side and client-side code for federated learning is stored in the servers/users directories.

# ~Federated learning module
The codes stored in infer,  loss, util, and model directories are used to train cross-lingual module.

The main.py is used for training FL module for social bots detection.
```python
python main.py
```
To train the federated bot detection codes.

The train.py is used for pre-traing cross-lingual module.

As for the processed data please contact us [email](mailto:dao@mail.ustc.edu.cn) and attach the information about your organization and the purpose of the data. 
The download data contains a datas directory containing processed data files, and a train_gpu3_warmup8000_latent256_kl800000_split1.0 directory contains our pre-trained cross-lingual model.

# Model
![Model Structure](model.png)

# Results

![Main Results](result.png)

# Citation
```
Coming Soon!
```

---

Have a nice day.
