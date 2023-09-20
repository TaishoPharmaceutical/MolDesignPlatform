import sys
sys.path.append("../../")

from EnsembleModel import get_ensemble_model, result_name, make_result_tables
from GTs.train import train_model, d_model, dff
from GTs.Utils.DataUtils import MAX_LEN
from GTs.train import sep_train_and_test_data
from glob import glob

import argparse
import pandas as pd
import pickle
from multiprocessing import Process

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

def predict_df(pred_df, ggt_path, data_path, reg_tasks,
               class_tasks, ggt_max_len=MAX_LEN,
               smiles_column = "SMILES", sub_pred=False):

    """
    data_path : The data is needed for converting z_score to original value
    """
    ensemble_model = get_ensemble_model(ggt_path, data_path,
                                      reg_tasks, class_tasks, ggt_max_len)

    smiles = list(pred_df["SMILES"])
    predictions = ensemble_model.predict(smiles, sub_pred)

    tasks = reg_tasks + class_tasks

    for i,p in enumerate(predictions):
        for j,t in enumerate(tasks):
            pred_df[t+result_name[i]] = p[j].numpy()

    return pred_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--df", help="path: data file", type=str)
    parser.add_argument("--save", help="folder path: save train model", type=str)
    parser.add_argument("--log", help="folder path: save model logs", type=str)
    parser.add_argument("--gpu", help="using gpu number", type=str)
    parser.add_argument("--zscore", help="using z_score in traing", type=bool, default=True)
    parser.add_argument("--batch", help="batch size of a training", type=int, default=200)
    parser.add_argument("--epoch", help="epoch", type=int, default=100)

    args = parser.parse_args()

    data_file = args.df
    save_folder = args.save
    log_folder = args.log
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

    os.makedirs(f"{save_folder}/data", exist_ok=True)
    os.makedirs(f"{save_folder}/result", exist_ok=True)

    train_file = save_folder + "/data/train.csv"
    test_file = save_folder + "/data/test.csv"
    copy_file = save_folder + "/data/data.csv"
    save_file_name = save_folder + "/data/mean_std.csv"

    df = pd.read_csv(data_file)
    df.to_csv(copy_file)

    train_df, test_df, reg_tasks, cls_tasks = sep_train_and_test_data(data_file, save_file_name, test_rate=0.1)
    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)

    print("Complete loading: classifications:%s regressions:%s"%(",".join(cls_tasks), ",".join(reg_tasks)))

    info = []
    info.append(reg_tasks)
    info.append(cls_tasks)
    info.append(d_model)
    info.append(dff)
    with open(save_folder + "/data/info", "wb") as pf:
        pickle.dump(info, pf)

    if args.zscore:
        reg_tasks_z = [x+"z" for x in reg_tasks]

    tasks = reg_tasks_z+cls_tasks

    p = Process(target=train_model, args=(save_folder, reg_tasks_z, cls_tasks, "pooling", args.batch, args.epoch))
    p.start()
    p.join()

    p = Process(target=train_model, args=(save_folder, reg_tasks_z, cls_tasks, "normal", args.batch, args.epoch))
    p.start()
    p.join()

    ggt_path = sorted(glob(save_folder + "/result/ggt_model*"))[0].split(".")[0]
    print(ggt_path)

    test_file = save_folder + "/data/test.csv"
    pred_df=pd.read_csv(test_file)

    df = predict_df(pred_df, ggt_path, data_file, reg_tasks, cls_tasks, sub_pred=True)
    df.to_csv(test_file, index=False)

    make_result_tables(df, save_folder+"/result", reg_tasks, cls_tasks)
