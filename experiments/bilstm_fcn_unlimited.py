import datetime
import os, sys, copy
import pickle

import pandas as pd
import numpy as np
import torch

from active_learning.labeler import Labeler
from active_learning.telemetry import ActiveLearningTelemetry
from active_learning.utils import active_learning_loop

from datasets.conll2003 import Conll2003Dataset
from models.supervised.supervised_al import SupervisedModelAL

if __name__ == "__main__":

    device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")
    print(f"Device: {device}")
    models_dir = 'experiments/models/saves/'

    # ----------------------------------- Prepare dataset ------------------------------------

    TASK_TYPE = 'NER' if len(sys.argv) < 2 else sys.argv[1]
    LABELED_UNLABELED_RATIO = 0.1
    train_dataset, test_dataset = Conll2003Dataset(task=TASK_TYPE).train_test_split()
    ltrain_dataset, utrain_dataset = train_dataset.split(LABELED_UNLABELED_RATIO)
    labeled_train_x, labeled_train_y = ltrain_dataset.x_embeddings, ltrain_dataset.y
    unlabeled_train_x, unlabeled_train_y = utrain_dataset.x_embeddings, utrain_dataset.y
    test_dataset_x, test_dataset_y = test_dataset.x_embeddings, test_dataset.y

    print(f"labelled size: {len(labeled_train_y)}", f"Unlabelled size: {len(unlabeled_train_y)}",
          f"test size: {len(test_dataset_y)})")
    # ------------------------------------ create model ---------------------------------------
    dropout_rate = 0.4
    model = SupervisedModelAL(train_dataset.max_tag_idx + 1, 100, 2, dropout_rate, dropout_rate)

    if device.type == 'cuda':
        model = model.cuda()
    model_kwargs = dict(device=device, repeats=250)

    # ------------------------------ setup active learning ------------------------------------
    al_batch_size = 100 if len(sys.argv) < 3 else int(sys.argv[2])
    labeler = Labeler()
    print(f"al_batch_size {al_batch_size}")

    # ------------------------------ train model ----------------------------------------------
    batch_size = 200
    epochs = 10
    for epoch_idx in range(epochs):
        print("--------- EPOCH: %d -------------" % (epoch_idx + 1))
        model.train_epoch(labeled_train_x, labeled_train_y, batch_size, device, 0.01)
        print(f"Labeled train score: {model.score(labeled_train_x, labeled_train_y, batch_size, device)}")
        print(f"Unlabeled train score: {model.score(unlabeled_train_x, unlabeled_train_y, batch_size, device)}")
        print(f"Test score:  {model.score(test_dataset_x, test_dataset_y, batch_size, device)}")

    now_str = datetime.datetime.now().strftime("%d%M%Y_%H%M%S")
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    path = f'supervised_{epochs}epochs_{now_str}.pkl'
    torch.save(model, os.path.join(models_dir, path))

    score = model.score(test_dataset_x, test_dataset_y, batch_size, device)
    telemetries_list = []
    for strategy in ['total_token_entropy', 'least_confident', 'token_entropy', 'nbest_sequence_entropy',
                     'margin', 'random']:
        telemetry = ActiveLearningTelemetry(strategy)
        telemetry.add_point(0, np.zeros((1, 5)), np.zeros((1, 5)), None, score)
        labeled_train_x, labeled_train_y = copy.deepcopy(ltrain_dataset.x_embeddings), copy.deepcopy(ltrain_dataset.y)
        unlabeled_train_x, unlabeled_train_y = copy.deepcopy(utrain_dataset.x_embeddings), copy.deepcopy(
            utrain_dataset.y)
        sim = copy.deepcopy(utrain_dataset.sim.tolist())
        test_dataset_x, test_dataset_y = test_dataset.x_embeddings, test_dataset.y
        model = torch.load(os.path.join(models_dir, path))
        for i in range(20):
            print(telemetry)
            indices, annotated_cnt, pseudolabeled_ratios, threshold_ratios, batch = active_learning_loop(
                labeled_train_x, labeled_train_y,
                unlabeled_train_x, unlabeled_train_y, sim, 1,
                model, model_kwargs, strategy, labeler, al_batch_size)
            scores_list = []
            for unit in range(10):
                m = copy.deepcopy(model)
                m.train_epoch(labeled_train_x, labeled_train_y, batch_size, device=device, lr=0.01)
                scores = m.score(test_dataset_x, test_dataset_y, batch_size=batch_size, device=device)
                scores_list.append(scores)
            model = m
            telemetry.add_point(annotated_cnt, pseudolabeled_ratios, threshold_ratios, batch, scores_list)
        with open(f'telemetry_{strategy}.pkl', 'wb') as output:
            pickle.dump(telemetry, output, pickle.DEFAULT_PROTOCOL)
        telemetries_list.append(telemetry)
    telemetries = pd.concat([t.to_df() for t in telemetries_list])
    telemetries.to_csv(f"bilstm_batchsize_{al_batch_size}_dropout_{dropout_rate}_{TASK_TYPE}_glove.csv")
