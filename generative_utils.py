from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sklearn.covariance
import scipy
import torch.optim as optim

from torch.autograd import Variable

def test_softmax(model, total_data, total_label):
    model.eval()
    
    batch_size = 100
    total, correct_D = 0, 0
    
    for data_index in range(int(np.floor(total_data.size(0)/batch_size))):
        data, target = total_data[total : total + batch_size], total_label[total : total + batch_size]
        data, target = Variable(data, volatile=True), Variable(target)

        total_out = model(data)

        total += batch_size
        pred = total_out.data.max(1)[1]
        equal_flag = pred.eq(target.data).cpu()
        correct_D += equal_flag.sum()
        
    return 100. * correct_D / total

def get_sample_mean_covariance(total_data, total_label, num_classes):
    group_lasso = sklearn.covariance.EmpiricalCovariance(assume_centered=False)
    sample_mean_per_class = torch.Tensor(num_classes, total_data.size(1)).fill_(0).cuda()
    total = 0
    
    for i in range(num_classes):
        index_list = total_label.eq(i)
        temp_feature = total_data[index_list.nonzero(), :]
        temp_feature = temp_feature.view(temp_feature.size(0), -1)
        sample_mean_per_class[i].copy_(torch.mean(temp_feature, 0))
    
    X = 0
    for i in range(total_data.size(0)):
        temp_feature = total_data[i].cuda()
        temp_feature = temp_feature - sample_mean_per_class[total_label[i]]
        temp_feature = temp_feature.view(-1,1)
        if i == 0:
            X = temp_feature.transpose(0,1)
        else:
            X = torch.cat((X,temp_feature.transpose(0,1)),0)
    # find inverse            
    group_lasso.fit(X.cpu().numpy())
    
    inverse_covariance = group_lasso.precision_
    inverse_covariance = torch.from_numpy(inverse_covariance).float().cuda()            

    return sample_mean_per_class, inverse_covariance

# function to extract the features from pre-trained models
def extract_features(model, total_data, total_label, file_root, data_name):
    model.eval()
    
    # compute the number of hidden features
    temp_x = torch.rand(2,3,32,32).cuda()
    temp_x = Variable(temp_x, volatile=True)
    temp_list = model.feature_list(temp_x)[1]
    num_output = len(temp_list)

    # memory for saving the features
    total_final_feature = [0]*num_output
    total = 0
    batch_size = 100
    
    for data_index in range(int(np.floor(total_data.size(0)/batch_size))):
        data = total_data[total : total + batch_size]
        data = Variable(data, volatile=True)
        
        _, out_features = model.feature_list(data)
        for i in range(num_output):
            out_features[i] = out_features[i].view(out_features[i].size(0), out_features[i].size(1), -1)
            out_features[i] = torch.mean(out_features[i].data, 2)
            if total == 0:
                total_final_feature[i] = out_features[i].cpu().clone()
            else:
                total_final_feature[i] = torch.cat((total_final_feature[i], out_features[i].cpu().clone()), 0)
        total += batch_size
        
    total_label = total_label.cpu().numpy()
    file_name_label = '%s/%s_label.npy' % (file_root, data_name)
    np.save(file_name_label, total_label)
    for i in range(num_output):
        file_name_data = '%s/%s_feature_%s.npy' % (file_root, data_name, str(i))
        total_feature = total_final_feature[i].numpy()
        np.save(file_name_data , total_feature)

def random_sample_mean(feature, total_label, num_classes):
    
    group_lasso = sklearn.covariance.EmpiricalCovariance(assume_centered = False)    
    new_feature, fraction_list = [], []
    frac = 0.7
    sample_mean_per_class = torch.Tensor(num_classes, feature.size(1)).fill_(0).cuda()
    total_label = total_label.cuda()

    
    total_selected_list = []
    for i in range(num_classes):
        index_list = total_label.eq(i)
        temp_feature = feature[index_list.nonzero(), :]
        temp_feature = temp_feature.view(temp_feature.size(0), -1)
        shuffler_idx = torch.randperm(temp_feature.size(0))
        index = shuffler_idx[:int(temp_feature.size(0)*frac)]
        fraction_list.append(int(temp_feature.size(0)*frac))
        total_selected_list.append(index_list.nonzero()[index.cuda()])

        selected_feature = torch.index_select(temp_feature, 0, index.cuda())
        new_feature.append(selected_feature)
        sample_mean_per_class[i].copy_(torch.mean(selected_feature, 0))
    
    total_covariance = 0
    for i in range(num_classes):
        flag = 0
        X = 0
        for j in range(fraction_list[i]):
            temp_feature = new_feature[i][j]
            temp_feature = temp_feature - sample_mean_per_class[i]
            temp_feature = temp_feature.view(-1,1)
            if flag  == 0:
                X = temp_feature.transpose(0,1)
                flag = 1
            else:
                X = torch.cat((X,temp_feature.transpose(0,1)),0)
            # find inverse            
        group_lasso.fit(X.cpu().numpy())
        inv_sample_conv = group_lasso.covariance_
        inv_sample_conv = torch.from_numpy(inv_sample_conv).float().cuda()
        if i == 0:
            total_covariance = inv_sample_conv*fraction_list[i]
        else:
            total_covariance += inv_sample_conv*fraction_list[i]
        total_covariance = total_covariance/sum(fraction_list)
    new_precision = scipy.linalg.pinvh(total_covariance.cpu().numpy())
    new_precision = torch.from_numpy(new_precision).float().cuda()
    
    return sample_mean_per_class, new_precision, total_selected_list

def MCD_single(feature, sample_mean, inverse_covariance):
    group_lasso = sklearn.covariance.EmpiricalCovariance(assume_centered=False)
    temp_batch = 100
    total, mahalanobis_score = 0, 0
    frac = 0.7
    for data_index in range(int(np.ceil(feature.size(0)/temp_batch))):
        temp_feature = feature[total : total + temp_batch].cuda()        
        gaussian_score = 0
        batch_sample_mean = sample_mean
        zero_f = temp_feature - batch_sample_mean
        term_gau = -0.5*torch.mm(torch.mm(zero_f, inverse_covariance), zero_f.t()).diag()
        # concat data
        if total == 0:
            mahalanobis_score = term_gau.view(-1,1)
        else:
            mahalanobis_score = torch.cat((mahalanobis_score, term_gau.view(-1,1)), 0)
        total += temp_batch
        
    mahalanobis_score = mahalanobis_score.view(-1)
    feature = feature.view(feature.size(0), -1)
    _, selected_idx = torch.topk(mahalanobis_score, int(feature.size(0)*frac))
    selected_feature = torch.index_select(feature, 0, selected_idx.cuda())
    new_sample_mean = torch.mean(selected_feature, 0)
    
    # compute covariance matrix
    X = 0
    flag = 0
    for j in range(selected_feature.size(0)):
        temp_feature = selected_feature[j]
        temp_feature = temp_feature - new_sample_mean
        temp_feature = temp_feature.view(-1,1)
        if flag  == 0:
            X = temp_feature.transpose(0,1)
            flag = 1
        else:
            X = torch.cat((X, temp_feature.transpose(0,1)),0)
    # find inverse            
    group_lasso.fit(X.cpu().numpy())
    new_sample_cov = group_lasso.covariance_
    
    return new_sample_mean, new_sample_cov, selected_idx

def make_validation(feature, total_label, sample_mean, inverse_covariance, num_classes):
    group_lasso = sklearn.covariance.EmpiricalCovariance(assume_centered=False)
    temp_batch = 100
    total, mahalanobis_score, prediction = 0, 0, 0
    frac = 0.5
    feature = feature.cuda()
    for data_index in range(int(np.floor(feature.size(0)/temp_batch))):
        temp_feature = feature[total : total + temp_batch]
        temp_label = total_label[total : total + temp_batch]
        gaussian_score = 0
        for i in range(num_classes):
            batch_sample_mean = sample_mean[i]
            zero_f = temp_feature - batch_sample_mean
            term_gau = -0.5*torch.mm(torch.mm(zero_f, inverse_covariance), zero_f.t()).diag()
            if i == 0:
                gaussian_score = term_gau.view(-1,1)
            else:
                gaussian_score = torch.cat((gaussian_score, term_gau.view(-1,1)), 1)
        generative_out = torch.index_select(gaussian_score, 1, temp_label.cuda()).diag()
        # concat data
        if total == 0:
            mahalanobis_score = generative_out
        else:
            mahalanobis_score = torch.cat((mahalanobis_score, generative_out), 0)
        total += temp_batch
    
    _, selected_idx = torch.topk(mahalanobis_score, int(total*frac))    
    return selected_idx

def remove_outlier(feature, sample_mean, inverse_covariance, num_classes):
    group_lasso = sklearn.covariance.EmpiricalCovariance(assume_centered=False)
    temp_batch = 100
    total, mahalanobis_score, prediction = 0, 0, 0
    frac = 0.7
    feature = feature.cuda()
    for data_index in range(int(np.floor(feature.size(0)/temp_batch))):
        temp_feature = feature[total : total + temp_batch]       
        gaussian_score = 0
        for i in range(num_classes):
            batch_sample_mean = sample_mean[i]
            zero_f = temp_feature - batch_sample_mean
            term_gau = -0.5*torch.mm(torch.mm(zero_f, inverse_covariance), zero_f.t()).diag()
            if i == 0:
                gaussian_score = term_gau.view(-1,1)
            else:
                gaussian_score = torch.cat((gaussian_score, term_gau.view(-1,1)), 1)
        generative_out, _ = torch.max(gaussian_score, dim=1)
        # concat data
        if total == 0:
            mahalanobis_score = generative_out
            prediction = gaussian_score.max(1)[1]
        else:
            mahalanobis_score = torch.cat((mahalanobis_score, generative_out), 0)
            prediction = torch.cat((prediction, gaussian_score.max(1)[1]), 0)
        total += temp_batch

    new_feature = []
    fraction_list = []
    total_selected_list = []
    sample_mean_per_class = torch.Tensor(num_classes, feature.size(1)).fill_(0).cuda()
    for i in range(num_classes):
        index_list = prediction.eq(i)
        temp_feature = feature[index_list.nonzero(), :]
        temp_feature = temp_feature.view(temp_feature.size(0), -1)
        temp_score = mahalanobis_score[index_list]
        fraction_list.append(int(torch.sum(index_list)*frac))
        _, selected_idx = torch.topk(temp_score, fraction_list[-1])
        total_selected_list.append(index_list.nonzero()[selected_idx])
        selected_feature = torch.index_select(temp_feature, 0, selected_idx.cuda())
        new_feature.append(selected_feature)
        sample_mean_per_class[i].copy_(torch.mean(selected_feature, 0))
    
    flag = 0
    X = 0
    for i in range(num_classes):
        for j in range(fraction_list[i]):
            temp_feature = new_feature[i][j]
            temp_feature = temp_feature - sample_mean_per_class[i]
            temp_feature = temp_feature.view(-1,1)
            if flag  == 0:
                X = temp_feature.transpose(0,1)
                flag = 1
            else:
                X = torch.cat((X,temp_feature.transpose(0,1)),0)

    # find inverse            
    group_lasso.fit(X.cpu().numpy())
    inv_sample_conv = group_lasso.precision_
    inv_sample_conv = torch.from_numpy(inv_sample_conv).float().cuda()
    V, D = torch.eig(inv_sample_conv, eigenvectors = True)
    V = torch.index_select(V, 1, torch.LongTensor([0]).cuda())
    return sample_mean_per_class, inv_sample_conv, total_selected_list

def test_ensemble(G_soft_list, soft_weight, total_val_data, total_val_label):
    
    batch_size = 100
    total, correct_D = 0, 0
    total_num_data = total_val_data[0].size(0)
    num_output = len(G_soft_list)
    
    for data_index in range(int(np.floor(total_num_data/batch_size))):
        target = total_val_label[total : total + batch_size].cuda()
        target = Variable(target)
        total_out = 0

        for i in range(num_output):
            out_features = total_val_data[i][total : total + batch_size].cuda()
            out_features = Variable(out_features, volatile=True)
            feature_dim = out_features.size(1)
            output = F.softmax(G_soft_list[i](out_features), dim=1)
            output = Variable(output.data, volatile=True)
            if i == 0:
                total_out = soft_weight[i]*output
            else:
                total_out += soft_weight[i]*output
                
        total += batch_size
        pred = total_out.data.max(1)[1]
        equal_flag = pred.eq(target.data).cpu()
        correct_D += equal_flag.sum() 
    return 100. * correct_D / total

def train_weights(G_soft_list, total_val_data, total_val_label):
    
    batch_size = 50
    
    # loss function
    nllloss = nn.NLLLoss().cuda()
    
    # parameyer
    num_ensemble = len(G_soft_list)
    train_weights = torch.Tensor(num_ensemble, 1).fill_(1).cuda()
    train_weights = nn.Parameter(train_weights)
    total, correct_D = 0, 0
    optimizer = optim.Adam([train_weights], lr=0.02)
    total_epoch = 20
    total_num_data = total_val_data[0].size(0)

    for data_index in range(int(np.floor(total_num_data/batch_size))):
        target = total_val_label[total : total + batch_size].cuda()
        target = Variable(target)
        soft_weight = F.softmax(train_weights, dim=0)
        total_out = 0

        for i in range(num_ensemble):
            out_features = total_val_data[i][total : total + batch_size].cuda()
            out_features = Variable(out_features, volatile=True)
            feature_dim = out_features.size(1)
            output = F.softmax(G_soft_list[i](out_features) , dim=1)
                
            output = Variable(output.data, volatile=True)
            if i == 0:
                total_out = soft_weight[i]*output
            else:
                total_out += soft_weight[i]*output
                
        total += batch_size
        pred = total_out.data.max(1)[1]
        equal_flag = pred.eq(target.data).cpu()
        correct_D += equal_flag.sum()
            
    for epoch in range(total_epoch):
        total = 0
        shuffler_idx = torch.randperm(total_num_data)

        for data_index in range(int(np.floor(total_num_data/batch_size))):
            index = shuffler_idx[total : total + batch_size]
            target = torch.index_select(total_val_label, 0, index).cuda()
            target = Variable(target)
            total += batch_size

            def closure():
                optimizer.zero_grad()
                soft_weight = F.softmax(train_weights, dim=0)

                total_out = 0
                for i in range(num_ensemble):
                    out_features = torch.index_select(total_val_data[i], 0, index).cuda()
                    out_features = Variable(out_features)
                    feature_dim = out_features.size(1)
                    output = F.softmax(G_soft_list[i](out_features) , dim=1)
                    
                    if i == 0:
                        total_out = soft_weight[i]*output
                    else:
                        total_out += soft_weight[i]*output
                        
                total_out = torch.log(total_out + 10**(-10))
                loss = nllloss(total_out, target)
                loss.backward()
                return loss

            optimizer.step(closure)
        
    correct_D, total = 0, 0
    
    for data_index in range(int(np.floor(total_num_data/batch_size))):
        target = total_val_label[total : total + batch_size].cuda()
        target = Variable(target)
        soft_weight = F.softmax(train_weights, dim=0)
        total_out = 0

        for i in range(num_ensemble):
            out_features = total_val_data[i][total : total + batch_size].cuda()
            out_features = Variable(out_features, volatile=True)
            feature_dim = out_features.size(1)
            output = F.softmax(G_soft_list[i](out_features) , dim=1)
                
            output = Variable(output.data, volatile=True)
            if i == 0:
                total_out = soft_weight[i]*output
            else:
                total_out += soft_weight[i]*output
                
        total += batch_size
        pred = total_out.data.max(1)[1]
        equal_flag = pred.eq(target.data).cpu()
        correct_D += equal_flag.sum()
        
    soft_weight = F.softmax(train_weights, dim=0)
    return soft_weight