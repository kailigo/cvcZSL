from scipy import io
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
# import kNN
from option import Options
from pdb import set_trace as breakpoint
from torch.optim import lr_scheduler

import torch.utils.data as data
from sklearn.metrics import accuracy_score
from utils import ReDirectSTD

from data_loader import data_loader
# from test_embeded import test_while_training_simple


args = Options().parse()
if args.log_to_file:
	log_file_name = './logs/'+ args.dataset + '/' + args.log_file
	ReDirectSTD(log_file_name, 'stdout', True)
	# ReDirectSTD(args.stderr_file, 'stderr', False)
model_file_name = './models/' + args.dataset + '/' + args.model_file

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
	
	outpred = np.array(outpred, dtype='int')
	test_label = test_label.numpy()
	unique_labels = np.unique(test_label)
	acc = 0
	for l in unique_labels:
		idx = np.nonzero(test_label == l)[0]
		acc += accuracy_score(test_label[idx], outpred[idx])
	acc = acc / unique_labels.shape[0]
	return acc


def compute_accuracy_all(test_att, att_all, test_visual_unseen, test_id_unseen, test_label_unseen,
		test_visual_seen, test_id_all, test_label_seen):

	cls_weights = forward(test_att)		
	cls_weights_norm = F.normalize(cls_weights, p=2, dim=cls_weights.dim()-1, eps=1e-12)                	
	acc_zsl = calc_accuracy(test_visual_unseen, test_label_unseen, cls_weights_norm, test_id_unseen)
	# test_id_all = np.concatenate((test_id_seen, test_id_unseen))	
	# weight_base_norm = F.normalize(weight_base, p=2, dim=cls_weights.dim()-1, eps=1e-12) 

	weight_train_all = forward(att_all)		
	weight_train_all_norm = F.normalize(weight_train_all, p=2, dim=weight_train_all.dim()-1, eps=1e-12) 

	weight_all = torch.cat((weight_train_all_norm, cls_weights_norm))
	
	acc_gzsl_unseen = calc_accuracy(test_visual_unseen, test_label_unseen, weight_all, test_id_all)
	acc_gzsl_seen = calc_accuracy(test_visual_seen, test_label_seen, weight_all, test_id_all)	
	H = 2 * acc_gzsl_seen * acc_gzsl_unseen / (acc_gzsl_seen + acc_gzsl_unseen)

	return acc_zsl, acc_gzsl_unseen, acc_gzsl_seen, H


def apply_classification_weights(features, cls_weights):
	cls_weights = F.normalize(cls_weights, p=2, dim=cls_weights.dim()-1, eps=1e-12)	
	cls_scores = scale_cls * torch.baddbmm(1.0, bias.view(1, 1, 1), 1.0, features, cls_weights.transpose(1,2))
	return cls_scores

def forward(att):
	a1 = F.relu(torch.mm(att, w1) + b1)
	a2 = F.relu(torch.mm(a1, w2) + b2)
	return a2





dataroot = '../data'
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
test_id, idx = np.unique(test_label_unseen, return_inverse=True)
att_pro = attribute[test_id]
test_label_unseen = idx + num_train
test_id = np.unique(test_label_unseen)


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

att_all = torch.from_numpy(train_att_unique).float().cuda()



bias = nn.Parameter(torch.FloatTensor(1).fill_(0).cuda(), requires_grad=True)
scale_cls = nn.Parameter(torch.FloatTensor(1).fill_(10).cuda(), requires_grad=True)
w1 = Variable(torch.FloatTensor(att_dim, args.hidden_dim).cuda(), requires_grad=True)
b1 = Variable(torch.FloatTensor(args.hidden_dim).cuda(), requires_grad=True)
w2 = Variable(torch.FloatTensor(args.hidden_dim, 2048).cuda(), requires_grad=True)
b2 = Variable(torch.FloatTensor(2048).cuda(), requires_grad=True)

w1.data.normal_(0, 0.02)
w2.data.normal_(0, 0.02)
b1.data.fill_(0)
b2.data.fill_(0)

optimizer = torch.optim.Adam([w1, b1, w2, b2, bias, scale_cls], lr=args.lr, weight_decay=args.opt_decay)


# breakpoint()
step_size = args.step_size
gamma = args.gamma
lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
criterion = nn.CrossEntropyLoss()

ways = args.ways
shots = args.shots

dataset = data_loader(train_x, train_att, train_label, ways=ways, shots=shots)

# breakpoint()
best_acc_zsl = 0.0
best_acc_gzsl_seen = 0.0
best_acc_gzsl_unseen = 0.0
best_H = 0.0
best_epoch = 0


for epoch in range(args.num_epochs):	
	epoch_loss = 0
	lr_scheduler.step()

	for i in range(1000):		
		batch_visual, batch_att, batch_label = dataset.__getitem__(i)				
		batch_visual = batch_visual.cuda()				
		batch_visual_norm = F.normalize(batch_visual, p=2, dim=batch_visual.dim()-1, eps=1e-12)				

		# breakpoint()		
		indx = torch.tensor(list(range(0, ways*shots, shots)))	
		unique_batch_att = torch.index_select(batch_att, 0, indx).float().cuda()		
		batch_weights = forward(unique_batch_att)	
		all_cls_weights = batch_weights
		all_cls_weight = F.normalize(all_cls_weights, p=2, dim=all_cls_weights.dim()-1, eps=1e-12)                

		score = apply_classification_weights(batch_visual_norm.unsqueeze(0), all_cls_weight.unsqueeze(0))
		score = score.squeeze(0)		
		loss = criterion(score, Variable(batch_label.cuda()))

		optimizer.zero_grad()
		loss.backward()
		torch.nn.utils.clip_grad_norm_([w1, b1, w2, b2, scale_cls, bias], 1)
		optimizer.step()
		epoch_loss = epoch_loss + loss

	epoch_loss = epoch_loss / 1000.0
	epoch_loss = epoch_loss.data.cpu().numpy()

	acc_zsl, acc_unseen_gzsl, acc_seen_gzsl, H = compute_accuracy_all(att_pro, att_all, test_x_unseen, 
		test_id, test_label_unseen, test_x_seen, train_test_id,  test_label_seen)
	
	H = 2 * acc_seen_gzsl * acc_unseen_gzsl / (acc_seen_gzsl + acc_unseen_gzsl)
		
	if H > best_H:
		best_epoch = epoch
		best_acc_zsl = acc_zsl		
		best_acc_gzsl_seen = acc_seen_gzsl
		best_acc_gzsl_unseen = acc_unseen_gzsl
		best_H = H

		best_w1 = w1.data.clone()
		best_b1 = b1.data.clone()
		best_w2 = w2.data.clone()
		best_b2 = b2.data.clone()
		best_scale_cls = scale_cls.data.clone()
		best_bias = bias.data.clone()

	if epoch % 50 == 0:
		torch.save({'w1': best_w1, 'b1': best_b1, 'w2': best_w2, 'b2': best_b2, 
			'scale_cls': best_scale_cls, 'bias': best_bias}, model_file_name)
		

	for param_group in optimizer.param_groups:
		print('ep: %d,  lr: %lf, loss: %.4f,  zsl: %.4f, gzsl: seen=%.4f, unseen=%.4f, h=%.4f' % 
			(epoch, param_group['lr'],  epoch_loss, acc_zsl, acc_seen_gzsl, acc_unseen_gzsl, H))		
	

print('best_ep: %d, zsl: %.4f, gzsl: seen=%.4f, unseen=%.4f, h=%.4f' % 
	(best_epoch, best_acc_zsl, best_acc_gzsl_seen, best_acc_gzsl_unseen, best_H))	
torch.save({'w1': best_w1, 'b1': best_b1, 'w2': best_w2, 'b2': best_b2, 
	'scale_cls': best_scale_cls, 'bias': best_bias}, model_file_name)

model = torch.load(model_file_name)
w1 = Variable(model['w1'], requires_grad=False)
b1 = Variable(model['b1'], requires_grad=False)
w2 = Variable(model['w2'], requires_grad=False)
b2 = Variable(model['b2'], requires_grad=False)
scale_cls = Variable(model['scale_cls'], requires_grad=False)
bias = Variable(model['bias'], requires_grad=False)

acc_zsl, acc_unseen_gzsl, acc_seen_gzsl, H = compute_accuracy_all(att_pro, att_all, test_x_unseen, 
		test_id, test_label_unseen, test_x_seen, train_test_id,  test_label_seen)
H = 2 * acc_seen_gzsl * acc_unseen_gzsl / (acc_seen_gzsl + acc_unseen_gzsl)

print('zsl: %.4f, gzsl: seen=%.4f, unseen=%.4f, h=%.4f' % (acc_zsl, acc_seen_gzsl, acc_unseen_gzsl, H))	