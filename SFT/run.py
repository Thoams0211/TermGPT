import argparse
import os
import torch
import matplotlib.pyplot as plt

from src.trainer import CustomTrainer


if __name__ == "__main__":

    # close wandb
    os.environ["WANDB_DISABLED"]="true"

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", type=str, required=True, help="Traing dataset: [QCA/Rules]")
    args, _ = parser.parse_known_args()
    config_path = f"./setup/config_{args.mode}.json"

    # config trainer & train
    trainer = CustomTrainer.from_config_file(config_path)
    trainer.train()

    # plot loss
    losses = trainer.lossCallBack.losses
    plt.plot(losses)
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig(f'./figures/loss.png')

    # save model
    trainer.save_model()