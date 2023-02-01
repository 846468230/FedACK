# -*- coding: utf-8 -*-
import logging
import argparse
from FLAlgorithms.utils.model_utils import create_model
from utils import printing_opt
from FLAlgorithms.servers import FedACK
from FLAlgorithms.servers import FedAvg,FedProx,FedDistill,FedEnsemble
from FLAlgorithms.utils.visualize_decision_boundary import visualize_decision_boundary
import sys
import torch
from utils import get_device
logger=None
def create_server_n_user(args, i):
    device = get_device(args.device)
    model = create_model(args.model, args.dataset, device,args.only_properties,args.visualize_boundary)
    if ('FedACK' in args.algorithm):
        server = FedACK(args, model, i,device,logger)
    elif 'FedProx' in args.algorithm:
        server = FedProx(args, model, i, device, logger)
    elif ('FedAvg' in args.algorithm):
        server = FedAvg(args, model, i,device,logger)
    elif 'FedDistill' in args.algorithm:
        server = FedDistill(args, model, i, device, logger)
    elif 'FedEnsemble' in args.algorithm:
        server = FedEnsemble(args, model, i, device, logger)
    else:
        print("Algorithm {} has not been implemented.".format(args.algorithm))
        exit()
    return server


def run_job(args, i):
    torch.manual_seed(i)
    print("\n\n         [ Start training iteration {} ]           \n\n".format(i))
    # Generate model
    server = create_server_n_user(args, i)
    if args.train:
        server.train(args)
        if args.visualize_boundary:
            visualize_decision_boundary(server)
            sys.exit(0)
        server.test()

def main(args):
    for i in range(args.times):
        run_job(args, i)
    print("Finished training.")


parser = argparse.ArgumentParser()

parser.add_argument("--model", type=str, default="cnn",help="the tweet feature encoder cnn or transformer")
parser.add_argument("--train", type=int, default=1, choices=[0,1])
parser.add_argument("--gen_batch_size", type=int, default=256, help='number of samples from generator')
parser.add_argument("--learning_rate", type=float, default=0.01, help="Local learning rate")
parser.add_argument("--personal_learning_rate", type=float, default=0.001, help="Personalized learning rate to caculate theta aproximately using K steps")
parser.add_argument("--ensemble_lr", type=float, default=0.01, help="Ensemble learning rate.")
parser.add_argument("--beta", type=float, default=1.0, help="Average moving parameter for pFedMe, or Second learning rate of Per-FedAvg")
parser.add_argument("--lamda", type=int, default=1, help="Regularization term")
parser.add_argument("--mix_lambda", type=float, default=0.1, help="Mix lambda for FedMXI baseline")
parser.add_argument("--embedding", type=int, default=0, help="Use embedding layer in generator network")
parser.add_argument("--num_users", type=int, default=10, help="Number of Users per round")
parser.add_argument("--K", type=int, default=1, help="Computation steps")
parser.add_argument("--times", type=int, default=3, help="running time")
parser.add_argument("--visualize_boundary", type=bool, default=False, help="draw the users' model's decision boundary")
parser.add_argument("--only_properties", type=bool, default=False, help="only use the users' properties")
parser.add_argument("--test_generalization", type=bool, default=False, help="whether test the generalization")
# "cresci-2015": 0, "botometer-feedback-2019": 1, "cresci-rtbust-2019": 2, "gilani-2017": 3,"vendor-purchased-2019": 4, "varol-2017": 5
parser.add_argument("--algorithm", type=str, default="FedACK",choices=["FedACK","FedAvg","FedProx","FedDistill","FedEnsemble"])
parser.add_argument("--dataset", type=str, default="Twibot-20",choices=["Twibot-20","vendor-purchased-2019"])
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--noniid_alpha", type=float, default=0.05, choices=[0.1,0.5,1,0.05], help="the noniid parametor of dataset partition from dirichlet distribution")
parser.add_argument("--device", type=int, default=7, choices=[0,1,2,3,4,5,6,7], help="run device (cpu | cuda)")
parser.add_argument("--mode", type=int,default=0,choices=[0,1],help="share all model (0) or only partial extractor model (1) or only_gen (2)")
parser.add_argument("--num_glob_iters", type=int, default=100)
parser.add_argument("--local_epochs", type=int, default=5)
parser.add_argument("--use_adam",type=bool, default=True, help="Optimizor use adam if true learning rate need to be set 0.001")
parser.add_argument("--result_path", type=str, default="results", help="directory path to save results")

args = parser.parse_args()
if (args.algorithm in ['FedAvg','FedProx',"FedDistill","FedEnsemble"]) or args.mode == 0 :
    args.mode = 'all'
if args.test_generalization:
    args.num_users = 5
# elif (args.algorithm =='FedGKD') or args.mode == 1:
#     args.mode = 'partial'
# else:
#     args.mode = 'partial'
logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)
logging.info("\n" + printing_opt(args))
logger = logging
print("=" * 80)
print("Summary of training process:")
print("Algorithm    : {}".format(args.algorithm))
print("Batch size   : {}".format(args.batch_size))
print("Dataset      : {}".format(args.dataset))
print("Noniid alpha : {}".format(args.noniid_alpha))
print("Only use property    : {}".format(args.only_properties))
print("Model type   : {}".format(args.model))
print("Share model mode     : {}".format(args.mode))
print("Learing rate     : {}".format(args.learning_rate))
print("Use Adam     : {}".format(args.use_adam))
print("Ensemble learing rate    : {}".format(args.ensemble_lr))
print("Average Moving       : {}".format(args.beta))
print("Subset of users      : {}".format(args.num_users))
print("Number of global rounds     : {}".format(args.num_glob_iters))
print("Number of local rounds      : {}".format(args.local_epochs))
print("Local Model       : {}".format(args.model))
print("Device            : {}".format(args.device))
print("Test generalization    : {}".format(args.test_generalization))
print("=" * 80)
main(args)
