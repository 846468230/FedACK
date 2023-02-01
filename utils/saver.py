import json

import torch
import os
import datetime


class Saver(object):
    def __init__(self, opt):
        self.opt = opt
        self.ckpt_names = []
        self.rouge1_results = []
        self.rouge2_results = []
#        self.model_path = opt.model_path + datetime.datetime.now().strftime("-%y%m%d-%H%M%S")
        self.model_path = opt.model_path + "_gpu" + str(opt.gpu) + "_warmup" + str(opt.warm_up) + "_latent" + str(opt.latent_dim) + "_kl" + str(opt.kl_annealing_steps) + "_split" + str(opt.split)
        self.max_to_keep = opt.max_to_keep
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

        with open(os.path.join(self.model_path, "params.json"), "w", encoding="UTF-8") as log:
            log.write(json.dumps(vars(opt), indent=4) + "\n")

    def save(self, save_dict, step,bleu_src_to_tar_real,bleu_tar_to_src_real,bleu_src_to_tar_fake,bleu_tar_to_src_fake,rouge1,rouge2,rouge3,rouge4,loss, typ):

        with open(os.path.join(self.model_path, "log"), "a", encoding="UTF-8") as log:
            log.write("%s\t" % datetime.datetime.now())
            log.write("type: %s\t" % typ)
            log.write("step: %6d\t" % step)
            log.write("loss: %.2f\t" % loss)
            log.write("BLEU StT_Real: %3.2f\t" % bleu_src_to_tar_real)
            log.write(" BLEU TtS_Real: %3.2f\t" % bleu_tar_to_src_real)
            log.write("BLEU StT_Fake: %3.2f\t " % bleu_src_to_tar_fake)
            log.write("BLEU TtS_Fake: %3.2f\t" % bleu_tar_to_src_fake)
            log.write("Rouge1: %3.2f\t" % rouge1)
            log.write("Rouge2: %3.2f\t" % rouge2)
            log.write("Rouge3: %3.2f\t" % rouge3)
            log.write("Rouge4: %3.2f\t" % rouge4)
            log.write("\n")

        filename = "checkpoint-step-%06d" % step
        full_filename = os.path.join(self.model_path, filename)
        self.ckpt_names.append(full_filename)
        torch.save(save_dict, full_filename)

        if 0 < self.max_to_keep < len(self.ckpt_names):
            earliest_ckpt = self.ckpt_names.pop(0)
            if os.path.exists(earliest_ckpt):
                os.remove(earliest_ckpt)