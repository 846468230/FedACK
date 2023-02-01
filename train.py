# -*- coding: utf-8 -*-
import logging
import torch.cuda
import os
import torch
from model import EModel
from utils import parseopt, get_device, Saver,calculate_bleu,printing_opt,calculate_rouge
from loss import WarmAdam, LabelSmoothingLoss
from data import build_dataset
from infer import beam_search

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)
opt = parseopt.parse_train_args()
device = get_device(opt.device)
logging.info("\n" + printing_opt(opt))
saver = Saver(opt)

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info('Trainable', str(trainable_num), str(total_num))
    return {'Total': total_num, 'Trainable': trainable_num}
def valid(model, criterion, valid_dataset, step,typ):
    model.eval()
    total_loss, total = 0.0, 0
    src_to_tar_total_loss = 0.0
    tar_to_src_total_loss = 0.0
    tar_map_to_src_to_tar_loss = 0.0
    src_map_to_tar_to_src_loss = 0.0
    hypothesis_tar_real, references_tar_real = [], []
    hypothesis_src_real, references_src_real = [], []
    hypothesis_tar_fake, hypothesis_src_fake = [], []

    for batch in valid_dataset:
        src_to_tgt_real, tgt_to_src_real, tgt_map_src_to_tgt, src_map_tgt_to_src, genloss = model.test_forward(
            batch.src, batch.tgt)
        loss1 = criterion(src_to_tgt_real, batch.tgt)
        loss2 = criterion(tgt_to_src_real, batch.src)
        loss3 = criterion(tgt_map_src_to_tgt, batch.tgt)
        loss4 = criterion(src_map_tgt_to_src, batch.src)
        src_to_tar_total_loss += loss1.data
        tar_to_src_total_loss += loss2.data
        tar_map_to_src_to_tar_loss += loss3.data
        src_map_to_tar_to_src_loss += loss4.data
        # loss = criterion(scores, batch.tgt)
        total_loss = (
                             src_to_tar_total_loss + tar_to_src_total_loss + tar_map_to_src_to_tar_loss + src_map_to_tar_to_src_loss) / 4
        total += 1

        # if opt.tf:
        _, prediction_tar_real = src_to_tgt_real.topk(k=1, dim=-1)
        _, prediction_src_real = tgt_to_src_real.topk(k=1, dim=-1)
        _, prediction_tar_fake = tgt_map_src_to_tgt.topk(k=1, dim=-1)
        _, prediction_src_fake = src_map_tgt_to_src.topk(k=1, dim=-1)
        # else:
        #     predictions = beam_search(opt, model, batch.src, valid_dataset.fields)

        hypothesis_tar_real += [valid_dataset.fields["tgt"].decode(p) for p in prediction_tar_real]
        references_tar_real += [valid_dataset.fields["tgt"].decode(t) for t in batch.tgt]
        hypothesis_src_real += [valid_dataset.fields["src"].decode(p) for p in prediction_src_real]
        references_src_real += [valid_dataset.fields["src"].decode(t) for t in batch.src]
        hypothesis_tar_fake += [valid_dataset.fields["src"].decode(p) for p in prediction_tar_fake]
        hypothesis_src_fake += [valid_dataset.fields["src"].decode(p) for p in prediction_src_fake]
    print(list(zip(hypothesis_tar_real, references_tar_real))[:3])
    print(list(zip(hypothesis_src_real, references_src_real))[:3])
    print(list(zip(hypothesis_tar_fake, references_tar_real))[:3])
    print(list(zip(hypothesis_src_fake, references_src_real))[:3])
    bleu_src_to_tar_real = calculate_bleu(hypothesis_tar_real, references_tar_real)
    bleu_tar_to_src_real = calculate_bleu(hypothesis_src_real, references_src_real)
    bleu_src_to_tar_fake = calculate_bleu(hypothesis_tar_fake, references_tar_real)
    bleu_tar_to_src_fake = calculate_bleu(hypothesis_src_fake, references_src_real)
    rouge1,_= calculate_rouge(hypothesis_tar_real, references_tar_real)
    rouge2,_= calculate_rouge(hypothesis_src_real, references_src_real)
    rouge3,_ = calculate_rouge(hypothesis_tar_fake, references_tar_real)
    rouge4,_ = calculate_rouge(hypothesis_src_fake, references_src_real)
    print(
        "type: %s \t loss: %.2f\t BLEU StT_Real: %3.2f\t BLEU TtS_Real: %3.2f\t BLEU StT_Fake: %3.2f\t BLEU TtS_Fake: %3.2f\tRouge1: %3.2f\tRouge2: %3.2f\tRouge3: %3.2f\tRouge4: %3.2f" % (
        typ, total_loss / total, bleu_src_to_tar_real, bleu_tar_to_src_real, bleu_src_to_tar_fake, bleu_tar_to_src_fake,
        rouge1, rouge2, rouge3, rouge4))
    logging.info("type: %s \t loss: %.2f\t BLEU StT_Real: %3.2f\t BLEU TtS_Real: %3.2f\t BLEU StT_Fake: %3.2f\t BLEU TtS_Fake: %3.2f\tRouge1: %3.2f\tRouge2: %3.2f\tRouge3: %3.2f\tRouge4: %3.2f" % (
        typ, total_loss / total, bleu_src_to_tar_real, bleu_tar_to_src_real, bleu_src_to_tar_fake, bleu_tar_to_src_fake,
        rouge1, rouge2, rouge3, rouge4))
    checkpoint = {"model": model.state_dict(), "opt": opt}
    saver.save(checkpoint, step, bleu_src_to_tar_real, bleu_tar_to_src_real, bleu_src_to_tar_fake, bleu_tar_to_src_fake,
               rouge1, rouge2, rouge3, rouge4, total_loss / total, typ)


def train(model, criterion, optimizer, train_dataset, valid_dataset, test_dataset,train_dis):
    total_loss = 0.0
    model.zero_grad()
    dis_total_loss = 0.0
    gen_total_loss = 0.0
    for i, batch in enumerate(train_dataset):
        # train dis
        for j in range(train_dis):
            model.discriminator.optimizer.zero_grad()
            _, _, _, _, dis_loss = model(batch.src, batch.tgt, True, 0)
            dis_loss.backward()
            dis_total_loss += dis_loss.data
            model.discriminator.optimizer.step()
        # train  gen 增加一个mapper之后的相似性损失， 也可以家在 encoder decoder上 但是先要改 valid 看看mapper效果。不行的话再加
        _, _, _, _, genloss = model(batch.src, batch.tgt, True, 1)  # 带padding的词句子索引矩阵
        model.mapper.optimizer.zero_grad()
        genloss.backward()
        gen_total_loss += genloss.data
        model.mapper.optimizer.step()
        # train encoder and decoder
        src_to_tgt_real, tgt_to_src_real, tgt_map_src_to_tgt, src_map_tgt_to_src, genloss = model(batch.src, batch.tgt,
                                                                                                  True,
                                                                                                  2)  # 带padding的词句子索引矩阵
        loss1 = criterion(src_to_tgt_real, batch.tgt)
        loss2 = criterion(tgt_to_src_real, batch.src)
        loss3 = criterion(tgt_map_src_to_tgt, batch.tgt)
        loss4 = criterion(src_map_tgt_to_src, batch.src)
        # loss = criterion(scores, batch.tgt)
        loss = (loss1 + loss2 + loss3 + loss4) / 4
        loss.backward()
        total_loss += loss.data

        if (i + 1) % opt.grad_accum == 0:
            optimizer.step()
            model.zero_grad()

            if optimizer.n_step % opt.report_every == 0:
                mean_loss = total_loss / opt.report_every / opt.grad_accum
                mean_dis_loss = dis_total_loss / opt.report_every / opt.grad_accum / opt.train_dis
                mean_gen_loss = gen_total_loss / opt.report_every / opt.grad_accum
                print("step: %7d\t loss: %7f\t dis_loss: %7f\t gen_loss: %7f" % (
                optimizer.n_step, mean_loss, mean_dis_loss, mean_gen_loss))
                logging.info("step: %7d\t loss: %7f\t dis_loss: %7f\t gen_loss: %7f" % (optimizer.n_step, mean_loss, mean_dis_loss, mean_gen_loss))
                total_loss = dis_total_loss = gen_total_loss = 0.0

            if optimizer.n_step % opt.save_every == 0:
                with torch.set_grad_enabled(False):
                    valid(model, criterion, valid_dataset, optimizer.n_step, "valid")
                    if optimizer.n_step > 10000 and optimizer.n_step % (2 * opt.save_every) == 0:
                        valid(model,criterion, test_dataset, optimizer.n_step, "test")
                model.train()
        if optimizer.n_step % 85000 == 0:
            logging.info("Training DONE all steps  %7d" % optimizer.n_step)
            break;
        del loss, dis_loss,genloss


def main():
    logging.info("Build dataset...")
    train_dataset = build_dataset(opt, opt.train, opt.vocab, device, train=True, testdata=False, V=opt.max_test_data)
    valid_dataset = build_dataset(opt, opt.valid, opt.vocab, device, train=False, testdata=False, V=opt.max_test_data)
    test_dataset = build_dataset(opt, opt.test, opt.vocab, device, train=False, testdata=False, V=opt.max_test_data)
    fields = valid_dataset.fields = train_dataset.fields = test_dataset.fields
    logging.info("Build model...")
    pad_ids = {"src": fields["src"].pad_id, "tgt": fields["tgt"].pad_id}
    vocab_sizes = {"src": len(fields["src"].vocab), "tgt": len(fields["tgt"].vocab)}
    checkpoint_num = []
    if os.path.exists(opt.model_path + "_gpu" + str(opt.gpu) + "_warmup" + str(opt.warm_up) + "_latent" + str(
        opt.latent_dim) + "_kl" + str(opt.kl_annealing_steps) + "_split" + str(opt.split)):
        files = os.listdir(opt.model_path + "_gpu" + str(opt.gpu) + "_warmup" + str(opt.warm_up) + "_latent" + str(
            opt.latent_dim) + "_kl" + str(opt.kl_annealing_steps) + "_split" + str(opt.split))
        for fil in files:
            if not os.path.isdir(fil) and len(fil) > 20:
                checkpoint_num.append(int(fil.split("-")[-1]))
        if len(checkpoint_num) > 0:
            opt.train_from = opt.model_path + "_gpu" + str(opt.gpu) + "_warmup" + str(opt.warm_up) + "_latent" + str(
                opt.latent_dim) + "_kl" + str(opt.kl_annealing_steps) + "_split" + str(
                opt.split) + "/checkpoint-step-%06d" % max(checkpoint_num)
    model = EModel.load_model(opt, device, pad_ids, vocab_sizes).to(device)

    logging.info("model parameters:", get_parameter_number(model))
    for name, parameters in model.named_parameters():
        print(name, ':', str(parameters.size()))

    criterion = LabelSmoothingLoss(opt.label_smoothing, vocab_sizes["tgt"], pad_ids["tgt"]).to(device)
    n_step = int(opt.train_from.split("-")[-1]) if opt.train_from else 1
    optimizer = WarmAdam(model.parameters(), opt.lr, opt.hidden_size, opt.warm_up, n_step)
    logging.info("start training...")
    train(model, criterion, optimizer, train_dataset, valid_dataset,test_dataset,opt.train_dis)

if __name__ == '__main__':
    main()