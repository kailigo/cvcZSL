from scipy import io
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
# import kNN
from option_trans import Options
from pdb import set_trace as breakpoint
from torch.optim import lr_scheduler

import torch.utils.data as data
from sklearn.metrics import accuracy_score
from utils import ReDirectSTD



args = Options().parse()
if args.log_to_file:
	log_file_name = './logs/'+ args.dataset + '/' + args.log_file
	ReDirectSTD(log_file_name, 'stdout', True)
	# ReDirectSTD(args.stderr_file, 'stderr', False)


weight_gen_model_name = './models/' + args.dataset + '/' + args.wt_model
trans_model_file_name = './models/' + args.dataset + '/' + args.trans_model_name


print(args)


def calc_accuracy(test_visual, test_label, cls_weights, test_id):	
	outpred = [0] * test_visual.shape[0]	
	score=apply_classification_weights(test_visual.unsqueeze(0).cuda(), 
		cls_weights.unsqueeze(0))
	score = score.squeeze(0)
	
	_, pred = score.max(dim=1)
	pred = pred.view(-1)
	test_label = test_label.view(-1)
	# breakpoint()
	outpred = test_id[pred]

	# breakpoint()
	outpred = np.array(outpred, dtype='int')
	test_label = test_label.numpy()
	unique_labels = np.unique(test_label)
	acc = 0
	for l in unique_labels:
		idx = np.nonzero(test_label == l)[0]
		acc += accuracy_score(test_label[idx], outpred[idx])
	acc = acc / unique_labels.shape[0]
	return acc


def calc_accuracy_simple(outpred, temp_test_label):

	outpred = np.array(outpred, dtype='int')
	# temp_test_label = temp_test_label.numpy()
	unique_labels = np.unique(temp_test_label)
	acc = 0
	for l in unique_labels:
		idx = np.nonzero(temp_test_label == l)[0]
		acc += accuracy_score(temp_test_label[idx], outpred[idx])
	acc = acc / unique_labels.shape[0]
	return acc


def compute_accuracy_all(test_att, att_all, test_visual_unseen, test_id_unseen, test_label_unseen,
		test_visual_seen, test_id_all, test_label_seen):

	cls_weights = forward(test_att)		

	cls_weights_norm = F.normalize(cls_weights, p=2, dim=cls_weights.dim()-1, eps=1e-12)                	
	acc_zsl = calc_accuracy(test_visual_unseen, test_label_unseen, cls_weights_norm, test_id_unseen)

	weight_trains = cls_weight_all[:num_train]
	weight_train_all_norm = F.normalize(weight_trains, p=2, dim=weight_trains.dim()-1, eps=1e-12) 

	weight_all = torch.cat((weight_train_all_norm, cls_weights_norm))

	acc_gzsl_unseen = calc_accuracy(test_visual_unseen, test_label_unseen, weight_all, test_id_all)
	acc_gzsl_seen = calc_accuracy(test_visual_seen, test_label_seen, weight_all, test_id_all)	
	H = 2 * acc_gzsl_seen * acc_gzsl_unseen / (acc_gzsl_seen + acc_gzsl_unseen)

	return acc_zsl, acc_gzsl_unseen, acc_gzsl_seen, H


def apply_classification_weights(features, cls_weights):
	cls_weights = F.normalize(cls_weights, p=2, dim=cls_weights.dim()-1, eps=1e-12)
	# bias = torch.FloatTensor(1).fill_(0).cuda()
	cls_scores = scale_cls * torch.baddbmm(1.0, bias.view(1, 1, 1), 1.0, features, cls_weights.transpose(1,2))
	return cls_scores

def forward(att):
	a1 = F.relu(torch.mm(att, w1) + b1)
	a2 = F.relu(torch.mm(a1, w2) + b2)
	return a2


dataroot = './data'
image_embedding = 'res101' 
class_embedding = 'original_att'
dataset = args.dataset+'_data'
matcontent = io.loadmat(dataroot + "/" + dataset + "/" + image_embedding + ".mat")

feature = matcontent['features'].T
label = matcontent['labels'].astype(int).squeeze() - 1
matcontent = io.loadmat(dataroot + "/" + dataset + "/" + class_embedding + "_splits.mat")

trainvalloc = matcontent['trainval_loc'].squeeze() - 1
test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1
test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1


att_name = 'att'
if args.dataset == 'AwA2':
	att_name = 'original_att'
attribute = matcontent[att_name].T 
train_x = feature[trainvalloc] 
train_label = label[trainvalloc].astype(int) 
train_att = attribute[train_label]
train_id, idx = np.unique(train_label, return_inverse=True)
train_att_unique = attribute[train_id]
num_train = len(train_id)
train_label = idx
train_id = np.unique(train_label)


test_x_unseen = feature[test_unseen_loc] 
test_label_unseen = label[test_unseen_loc].astype(int)
test_att_unseen = attribute[test_label_unseen]

test_id, idx = np.unique(test_label_unseen, return_inverse=True)
att_pro = attribute[test_id]
test_label_unseen = idx + num_train
test_id = np.unique(test_label_unseen)

# num_train = len(train_id)
num_test = len(test_id)
num_classes = num_train + num_test

# train_test_att = np.concatenate((train_att_unique, att_pro)) 
train_test_id = np.concatenate((train_id, test_id))

test_x_seen = feature[test_seen_loc] 
test_label_seen = label[test_seen_loc].astype(int)
_, idx = np.unique(test_label_seen, return_inverse=True)
test_label_seen = idx

att_dim = train_att.shape[1]
feat_dim = train_x.shape[1]

att_pro = torch.from_numpy(att_pro).float().cuda()
test_x_seen = torch.from_numpy(test_x_seen).float().cuda()
test_x_seen = F.normalize(test_x_seen, p=2, dim=test_x_seen.dim()-1, eps=1e-12)
test_x_unseen = torch.from_numpy(test_x_unseen).float().cuda()
test_x_unseen = F.normalize(test_x_unseen, p=2, dim=test_x_unseen.dim()-1, eps=1e-12)
test_label_seen = torch.tensor(test_label_seen)
test_label_unseen = torch.tensor(test_label_unseen)



gen_model = torch.load(weight_gen_model_name)
w1 = gen_model['w1'] 
b1 = gen_model['b1'] 
w2 = gen_model['w2'] 
b2 = gen_model['b2'] 
gen_scale_cls = gen_model['scale_cls'] 
gen_bias = gen_model['bias'] 


att_all = torch.from_numpy(train_att_unique).float().cuda()


model = torch.load(trans_model_file_name)
w1 = Variable(model['w1'], requires_grad=False)
b1 = Variable(model['b1'], requires_grad=False)
w2 = Variable(model['w2'], requires_grad=False)
b2 = Variable(model['b2'], requires_grad=False)
scale_cls = Variable(model['scale_cls'], requires_grad=False)
bias = Variable(model['bias'], requires_grad=False)
cls_weight_all = Variable(model['cls_weight_all'], requires_grad=False)
 
acc_zsl, acc_unseen_gzsl, acc_seen_gzsl, H = compute_accuracy_all(att_pro, att_all, test_x_unseen, 
		test_id, test_label_unseen, test_x_seen, train_test_id,  test_label_seen)

print('zsl: %.4f, gzsl: seen=%.4f, unseen=%.4f, h=%.4f' % (acc_zsl, acc_seen_gzsl, acc_unseen_gzsl, H))	