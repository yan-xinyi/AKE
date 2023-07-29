# -*- coding: utf-8 -*-
import logging
from utils import *
from config import *

torch.cuda.set_device(0)                                                # Setting the training device to the gpu's cuda

if __name__ == '__main__':

  for i in range(run_times):                                            # Setting the number of repeat runs

    # Print the results to the log              
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(lineno)d - %(levelname)s: %(message)s',
                        datefmt='%m-%d %H:%M',
                        filename=save_path + fs_name + '.txt',          # save path of data
                        filemode='a')                                   # 'w' for overwrites the write, 'a' for continues to write

    logging.info("Start with the feature combination of:"+ fs_name)     # load the model
    model_name = get_model(model_type)                                            # Choose which model to run
    # Train the model
    best_P, best_R, best_F, best_epoch = model_name()

    logging.info("The best P first 20 epoch is:" + str(best_P))         # Print the result into the log file
    logging.info("The best R first 20 epoch is:" + str(best_R))
    logging.info("The best F first 20 epoch is:" + str(best_F))
    logging.info("The best epoch of first 20 epoch is:" + str(best_epoch))
    logging.info("The " + str(i + 1) + "fold is end.")

