#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import time
import os
import numpy as np
from postproc.KNN import KNN


class User():
  def __init__(self, ARCH, DATA, datadir, logdir, modeldir,split):
    # parameters
    self.ARCH = ARCH
    self.DATA = DATA
    self.datadir = datadir
    self.logdir = logdir
    self.modeldir = modeldir
    self.split = split

    # get the data
    from dataset.kitti.parser import Parser
    self.parser = Parser(root=self.datadir,
                                      train_sequences=self.DATA["split"]["train"],
                                      valid_sequences=self.DATA["split"]["valid"],
                                      test_sequences=self.DATA["split"]["test"],
                                      labels=self.DATA["labels"],
                                      color_map=self.DATA["color_map"],
                                      learning_map=self.DATA["learning_map"],
                                      learning_map_inv=self.DATA["learning_map_inv"],
                                      sensor=self.ARCH["dataset"]["sensor"],
                                      max_points=self.ARCH["dataset"]["max_points"],
                                      batch_size=1,
                                      workers=self.ARCH["train"]["workers"],
                                      gt=True,
                                      shuffle_train=False)

    # concatenate the encoder and the head
    with torch.no_grad():
        torch.nn.Module.dump_patches = True
        if self.ARCH["train"]["pipeline"] == "hardnet":
            from modules.network.HarDNet import HarDNet
            self.model = HarDNet(self.parser.get_n_classes(), self.ARCH["train"]["aux_loss"])

        if self.ARCH["train"]["pipeline"] == "res":
            from modules.network.ResNet import ResNet_34
            self.model = ResNet_34(self.parser.get_n_classes(), self.ARCH["train"]["aux_loss"])
            self.model = self.model
            # self.get_model_size()
            

            def convert_relu_to_softplus(model, act):
                for child_name, child in model.named_children():
                    if isinstance(child, nn.LeakyReLU):
                        setattr(model, child_name, act)
                    else:
                        convert_relu_to_softplus(child, act)

            if self.ARCH["train"]["act"] == "Hardswish":
                convert_relu_to_softplus(self.model, nn.Hardswish())
            elif self.ARCH["train"]["act"] == "SiLU":
                convert_relu_to_softplus(self.model, nn.SiLU())

        if self.ARCH["train"]["pipeline"] == "fid":
            from modules.network.Fid import ResNet_34
            self.model = ResNet_34(self.parser.get_n_classes(), self.ARCH["train"]["aux_loss"])

            if self.ARCH["train"]["act"] == "Hardswish":
                convert_relu_to_softplus(self.model, nn.Hardswish())
            elif self.ARCH["train"]["act"] == "SiLU":
                convert_relu_to_softplus(self.model, nn.SiLU())

#     print(self.model)
    w_dict = torch.load(modeldir + "/SENet_valid_best",
                        map_location=lambda storage, loc: storage)
    self.model.load_state_dict(w_dict['state_dict'], strict=True)
    # use knn post processing?
    self.post = None
    if self.ARCH["post"]["KNN"]["use"]:
      self.post = KNN(self.ARCH["post"]["KNN"]["params"],
                      self.parser.get_n_classes())
    print(self.parser.get_n_classes())

    # GPU?
    self.gpu = False
    self.model_single = self.model
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Infering in device: ", self.device)
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
      cudnn.benchmark = True
      cudnn.fastest = True
      self.gpu = True
      self.model.cuda()

  def get_model_size(self):
      size_model = 0
      for param in self.model.parameters():
       if param.data.is_floating_point():
        size_model += param.numel() * torch.finfo(param.data.dtype).bits
       else:
        size_model += param.numel() * torch.iinfo(param.data.dtype).bits
      print(f"model size: {size_model} / bit | {size_model / 8e6:.2f} / MB")
      exit()

  def infer(self):
    cnn = []
    knn = []
    dbscan = []
    if self.split == None:

        self.infer_subset(loader=self.parser.get_train_set(),
                          to_orig_fn=self.parser.to_original, cnn=cnn, knn=knn)

        # do valid set
        self.infer_subset(loader=self.parser.get_valid_set(),
                          to_orig_fn=self.parser.to_original, cnn=cnn, knn=knn)
        # do test set
        self.infer_subset(loader=self.parser.get_test_set(),
                          to_orig_fn=self.parser.to_original, cnn=cnn, knn=knn)


    elif self.split == 'valid':
        self.infer_subset(loader=self.parser.get_valid_set(),
                        to_orig_fn=self.parser.to_original, cnn=cnn, knn=knn, dbscan=dbscan)
    elif self.split == 'train':
        self.infer_subset(loader=self.parser.get_train_set(),
                        to_orig_fn=self.parser.to_original, cnn=cnn, knn=knn)
    else:
        self.infer_subset(loader=self.parser.get_test_set(),
                        to_orig_fn=self.parser.to_original, cnn=cnn, knn=knn)
    print("Mean CNN inference time:{}\t std:{}".format(np.mean(cnn), np.std(cnn)))
    print("Mean KNN inference time:{}\t std:{}".format(np.mean(knn), np.std(knn)))
    print("Mean KNN inference time:{}\t std:{}".format(np.mean(dbscan), np.std(dbscan)))
    print("Total Frames:{}".format(len(cnn)))
    print("Finished Infering")

    return

  def infer_subset(self, loader, to_orig_fn,cnn,knn,dbscan=None):
    # switch to evaluate mode

    self.model.eval()
    total_time=0
    total_frames=0
    # empty the cache to infer in high res
    if self.gpu:
      torch.cuda.empty_cache()

    with torch.no_grad():
      for i, (proj_in, proj_mask, _, _, path_seq, path_name, p_x, p_y, proj_range, unproj_range, _, _, _, _, npoints) in enumerate(loader):
        # first cut to rela size (batch size one allows it)
        p_x = p_x[0, :npoints]
        p_y = p_y[0, :npoints]
        proj_range = proj_range[0, :npoints]
        unproj_range = unproj_range[0, :npoints]
        path_seq = path_seq[0]
        path_name = path_name[0]

        if self.gpu:
          proj_in = proj_in.cuda()
          p_x = p_x.cuda()
          p_y = p_y.cuda()
          if self.post:
            proj_range = proj_range.cuda()
            unproj_range = unproj_range.cuda()
        end = time.time()

        if self.ARCH["train"]["aux_loss"]:
            with torch.cuda.amp.autocast(enabled=True):
                [proj_output, x_2, x_3, x_4] = self.model(proj_in)
        else:
            with torch.cuda.amp.autocast(enabled=True):
                proj_output = self.model(proj_in)

        proj_argmax = proj_output[0].argmax(dim=0)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        res = time.time() - end
        print("Network seq", path_seq, "scan", path_name,
              "in", res, "sec")
        end = time.time()
        cnn.append(res)

        if self.post:
            # knn postproc
            unproj_argmax = self.post(proj_range,
                                      unproj_range,
                                      proj_argmax,
                                      p_x,
                                      p_y)
#             # nla postproc
#             proj_unfold_range, proj_unfold_pre = NN_filter(proj_range, proj_argmax)
#             proj_unfold_range=proj_unfold_range.cpu().numpy()
#             proj_unfold_pre=proj_unfold_pre.cpu().numpy()
#             unproj_range = unproj_range.cpu().numpy()
#             #  Check this part. Maybe not correct (Low speed caused by for loop)
#             #  Just simply change from
#             #  https://github.com/placeforyiming/IROS21-FIDNet-SemanticKITTI/blob/7f90b45a765b8bba042b25f642cf12d8fccb5bc2/semantic_inference.py#L177-L202
#             for jj in range(len(p_x)):
#                 py, px = p_y[jj].cpu().numpy(), p_x[jj].cpu().numpy()
#                 if unproj_range[jj] == proj_range[py, px]:
#                     unproj_argmax = proj_argmax[py, px]
#                 else:
#                     potential_label = proj_unfold_pre[0, :, py, px]
#                     potential_range = proj_unfold_range[0, :, py, px]
#                     min_arg = np.argmin(abs(potential_range - unproj_range[jj]))
#                     unproj_argmax = potential_label[min_arg]

        else:
            # put in original pointcloud using indexes
            unproj_argmax = proj_argmax[p_y, p_x]

        # measure elapsed time
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        res = time.time() - end
        print("KNN Infered seq", path_seq, "scan", path_name,
              "in", res, "sec")
        knn.append(res)
        end = time.time()

        # save scan
        # get the first scan in batch and project scan
        pred_np = unproj_argmax.cpu().numpy()
        pred_np = pred_np.reshape((-1)).astype(np.int32)

        # map to original label
        pred_np = to_orig_fn(pred_np)

        preds, time_dbscan = self.dbscan_panoptic(pred_np, proj_argmax, proj_range, p_x, p_y)
        dbscan.append(time_dbscan)

        # save scan
        path = os.path.join(self.logdir, "sequences",
                            path_seq, "predictions", path_name)
        pred_np.tofile(path)


  def dbscan_panoptic(self, pred_np, proj_argmax, proj_range, p_x, p_y):

        # print("points: ", unproj_xyz)
        # dbscan
        import hdbscan
        from sklearn.cluster import DBSCAN

        start_time = time.time()
        # X = unproj_xyz[0][:npoints].cuda()

        cars_idx = proj_argmax == 1
        bicycle_idx = proj_argmax == 2
        motorcycle_idx = proj_argmax == 3
        truch_idx = proj_argmax == 4
        other_vehicle_idx = proj_argmax == 5
        person_idx = proj_argmax == 6
        bicyclist_idx = proj_argmax == 7
        motorcyclist_idx = proj_argmax == 8
        
        # cars_idx = unproj_argmax == 1
        # bicycle_idx = unproj_argmax == 2
        # motorcycle_idx = unproj_argmax == 3
        # truch_idx = unproj_argmax == 4
        # other_vehicle_idx = unproj_argmax == 5
        # person_idx = unproj_argmax == 6
        # bicyclist_idx = unproj_argmax == 7
        # motorcyclist_idx = unproj_argmax == 8

        def cluster(proj_range, idx, k):
           Z = proj_range[idx]
           Y, X = torch.where(idx)
           data = torch.stack((X, Y, Z))
           # data = torch.stack((Z,))
           if torch.numel(data) < 5:
              return torch.zeros(proj_range.shape)
           data = torch.transpose(data, 0, 1)

        #    data = torch.unsqueeze(data, 0)
        # #    print("data: ", data.shape)
        #    dist = torch.cdist(data, data)
        #    if torch.numel(dist) == 1:
        #       return None
        #    print("cdist: ", dist.shape)
        #    cluster = DBSCAN(metric="precomputed").fit(dist[0].detach().cpu().numpy())
        #    cluster = hdbscan.HDBSCAN(metric="precomputed", allow_single_cluster=True).fit(dist[0].detach().cpu().numpy().astype('double'))
           cluster = hdbscan.HDBSCAN(allow_single_cluster=True).fit(data.detach().cpu().numpy())
           labels = cluster.labels_
           labels += 1
           labels[labels > 0] += k
        #    print("labels: ", min(labels), max(labels))
           panoptic_labels = torch.zeros(proj_range.shape)
           panoptic_labels[idx] = torch.Tensor(labels)

        #    print("panoptic labels: ", panoptic_labels, panoptic_labels.min(), panoptic_labels.max())
           return panoptic_labels
        #    return cluster
        # clustering_persons = DBSCAN(eps=3, min_samples=2).fit(X[person_idx])

        cars_cluster = cluster(proj_range, cars_idx, 0)
        bicycle_cluster = cluster(proj_range, bicycle_idx, 1000)
        motorcycle_cluster = cluster(proj_range, motorcycle_idx, 2000)
        truch_cluster = cluster(proj_range, truch_idx, 3000)
        other_vehicle_cluster = cluster(proj_range, other_vehicle_idx, 4000)
        person_cluster = cluster(proj_range, person_idx, 5000)
        bicyclist_cluster = cluster(proj_range, bicyclist_idx, 6000)
        motorcyclist_cluster = cluster(proj_range, motorcyclist_idx, 7000)

        p_x_cpu = p_x.detach().cpu().numpy()
        p_y_cpu = p_y.detach().cpu().numpy()
        
        unproj_panoptic_cars = cars_cluster[p_y_cpu, p_x_cpu]
        unproj_panoptic_bicycle = (bicycle_cluster[p_y_cpu, p_x_cpu])
        unproj_panoptic_motorcycle = motorcycle_cluster[p_y_cpu, p_x_cpu]
        unproj_panoptic_truch = truch_cluster[p_y_cpu, p_x_cpu]
        unproj_panoptic_other_vehicle = other_vehicle_cluster[p_y_cpu, p_x_cpu]
        unproj_panoptic_person = person_cluster[p_y_cpu, p_x_cpu]
        unproj_panoptic_bicyclist = bicyclist_cluster[p_y_cpu, p_x_cpu]
        unproj_panoptic_motorcyclist = motorcyclist_cluster[p_y_cpu, p_x_cpu]
        
        pred_panoptic = (unproj_panoptic_cars + 
                         unproj_panoptic_bicycle + 
                         unproj_panoptic_motorcycle + 
                         unproj_panoptic_truch + 
                         unproj_panoptic_other_vehicle +
                         unproj_panoptic_person +
                         unproj_panoptic_bicyclist + 
                         unproj_panoptic_motorcyclist)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        result_time = time.time() - start_time
        print("DBSCAN", time.time() - start_time, "sec")

        # add panoptic
        pred_np = (pred_panoptic.detach().cpu().numpy().astype(np.int32) << 16) | pred_np


        return pred_np, result_time
         
