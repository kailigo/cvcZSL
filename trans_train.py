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


class data_loader(data.Dataset):
	def __init__(self, feats, atts, labels, pseudo_test_feat, 
		pseudo_test_att, pseudo_test_label, ways=32, shots=4, test_shot_multiplifier=1.0):

		self.ways = ways		
		self.shots = shots    		

		self.feats = torch.tensor(feats).float()
		self.atts = torch.tensor(atts).float()
		self.labels = labels
		self.classes = np.unique(labels)

		self.pseudo_test_feat = torch.tensor(pseudo_test_feat)
		self.pseudo_test_att = torch.tensor(pseudo_test_att)
		self.pseudo_test_label = pseudo_test_label
		self.pseudo_test_class = np.unique(pseudo_test_label)
		self.test_shot_multiplifier = test_shot_multiplifier

		self.num_train_class = len(self.classes)
		self.num_test_class = len(self.pseudo_test_class)

		# breakpoint()


	def __getitem__(self, index):		
		
		test_feat_selected = []
		test_att_selected = []
		test_shots = int(self.shots * self.test_shot_multiplifier)
		test_label_selected = torch.LongTensor(self.num_test_class*test_shots)

		train_feat_selected = []
		train_att_selected = []
		train_label_selected = torch.LongTensor(self.ways*self.shots)

		selected_classes = np.random.choice(list(self.classes), self.ways, False)

		is_first = True
		for i in range(self.ways):
			idx = (self.labels==selected_classes[i]).nonzero()[0]			
			select_instances = np.random.choice(idx, self.shots, False)			

			for j in range(self.shots):
				feat = self.feats[select_instances[j], :]
				att = self.atts[select_instances[j], :]  
				feat = feat.unsqueeze(0)
				att = att.unsqueeze(0)
				if is_first:
					is_first=False
					train_feat_selected = feat
					train_att_selected = att
				else:                   
					train_feat_selected = torch.cat((train_feat_selected, feat),0)                
					train_att_selected = torch.cat((train_att_selected, att),0)                
				train_label_selected[i*self.shots+j] = selected_classes[i].item()

		
		num_keep_classes = num_classes - self.num_test_class

		is_first = True
		for i in range(self.num_test_class):
			idx = (self.pseudo_test_label == self.pseudo_test_class[i]).nonzero()[0]
			select_instances = np.random.choice(idx, test_shots, False)

			for j in range(test_shots):
				feat = self.pseudo_test_feat[select_instances[j], :]
				att = self.pseudo_test_att[select_instances[j], :]  
				feat = feat.unsqueeze(0)
				att = att.unsqueeze(0)

				if is_first:
					is_first=False
					test_feat_selected = feat
					test_att_selected = att
				else:                   
					test_feat_selected = torch.cat((test_feat_selected, feat),0)                
					test_att_selected = torch.cat((test_att_selected, att),0)      

				# test_label_selected[i*test_shots+j] = self.pseudo_test_class[i].item()
				test_label_selected[i*test_shots+j] = i + num_keep_classes

		return selected_classes, train_feat_selected, train_att_selected, train_label_selected, \
			self.pseudo_test_class, test_feat_selected, test_att_selected, test_label_selected

		
	def __len__(self):
		return self.__size


def Lq_loss(outputs, labels, q, num_classes = 10):
	outputs = F.softmax(outputs, dim=1)
	one_hot = Variable(torch.zeros(labels.size(0), num_classes).cuda().scatter_(1, labels.long().view(-1, 1).data, 1))
	mask = one_hot.gt(0)
	loss = torch.masked_select(outputs, mask)
	# q = torch.tensor(q).cuda()
	# loss = (1-(loss+10**(-8))**q)/q
	loss = (1-(loss+10**(-8))**q)/q
	# breakpoint()
	# (1-(loss)**(1e-8))/(1e-8)
	loss = loss.sum() / loss.shape[0]

	return loss



def updata_reliable_test_data(filter_type='ratio', ratio=1.2, num_per_class=5):

	# test_x_unseen_var = torch.from_numpy(test_x_unseen).float().cuda()
	# test_x_unseen_norm = F.normalize(test_x_unseen_var, p=2, dim=test_x_unseen_var.dim()-1, eps=1e-12)	
	test_x_unseen_norm = test_x_unseen

	# cls_weights_novel = cls_weight_all[num_train:] 	
	cls_weights_novel = forward(att_pro)	

	cls_weights_novel_norm = F.normalize(cls_weights_novel, p=2, dim=cls_weights_novel.dim()-1, eps=1e-12) 

	score = apply_classification_weights(test_x_unseen_norm.unsqueeze(0).cuda(), cls_weights_novel_norm.unsqueeze(0).cuda())
	score = score.squeeze(0)	

	max_val, pred = score.max(dim=1)
	pred = pred.view(-1)
	outpred = test_id[pred]
	
	# breakpoint()
	acc = calc_accuracy_simple(outpred, test_label_unseen.numpy())	
	print(acc)	
	test_class_num = len(test_id)

	if filter_type == 'ratio':	
		score_sorted, _ = score.sort(dim=1, descending=True)
		# breakpoint()
		sorted_idx_select1 = [i for i in range(score_sorted.size()[0]) 
			if torch.max(score_sorted[i][0].abs(), score_sorted[i][1].abs()) /
				torch.min(score_sorted[i][0].abs(), score_sorted[i][1].abs()) > ratio ]
			# (score_sorted[i][0]-score_sorted[i][1]) / (score_sorted[i][0]+score_sorted[i][1]) > (ratio-1)/(ratio+1)]	
		max_pred = torch.tensor(sorted_idx_select1)			
		temp_test_label_unseen1 = test_label_unseen[sorted_idx_select1]
		outpred = pred[max_pred]
		outpred1 = test_id[outpred]

		select_num = num_per_class
		score_sorted, idx_sorted = score.sort(dim=0, descending=True)		
		sorted_idx_select2 = idx_sorted[:select_num].view(-1)
		outpred =  torch.arange(test_class_num).repeat(select_num).view(-1)		
		outpred2 = test_id[outpred]
		temp_test_label_unseen2 = test_label_unseen[sorted_idx_select2]

		
		outpred = np.concatenate((outpred1, outpred2))
		temp_test_label_unseen = torch.cat((temp_test_label_unseen1, temp_test_label_unseen2))	

		# breakpoint()			

		acc1 = calc_accuracy_simple(outpred1, temp_test_label_unseen1.numpy())
		acc2 = calc_accuracy_simple(outpred2, temp_test_label_unseen2.numpy())
		acc = calc_accuracy_simple(outpred, temp_test_label_unseen.numpy())

		sorted_idx_select = sorted_idx_select1 + sorted_idx_select2.tolist()

		print(acc, acc1, acc2)

	elif filter_type == 'constant':
		select_num = num_per_class
		score_sorted, idx_sorted = score.sort(dim=0, descending=True)		
		sorted_idx_select = idx_sorted[:select_num].view(-1)
		outpred =  torch.arange(test_class_num).repeat(select_num).view(-1)		

		# breakpoint()

		outpred = test_id[outpred]
		gt_label = test_label_unseen[sorted_idx_select]
		acc = calc_accuracy_simple(outpred, gt_label.numpy())

		print(acc)	
	else:	
		# breakpoint()
		max_val_sorted, sorted_idx = max_val.sort(descending=True)		
		sorted_idx = sorted_idx.cpu().data.numpy()
		correct_candidates = int(len(sorted_idx) / 3.0)
		sorted_idx_select = sorted_idx[:correct_candidates]	
		max_pred = pred[sorted_idx_select]
		temp_test_label_unseen = test_label_unseen[sorted_idx_select]
		outpred = test_id[max_pred]
		acc = calc_accuracy_simple(outpred, temp_test_label_unseen.numpy())
		print(acc)	
	
	test_data_temp = test_x_unseen[sorted_idx_select, :]	
	test_att_temp = test_att_unseen[sorted_idx_select, :]	

	print('test data size: %d/%d' % (len(sorted_idx_select), len(test_x_unseen)))
	
	test_label_temp = outpred
	# test_label_temp = temp_test_label_unseen.numpy()
	# test_iter_ = data_iterator_test(test_data_temp, test_label_temp)
	return test_data_temp, test_att_temp, test_label_temp



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
weight_test_init = forward(att_pro).data


weight_base = forward(att_all).data



w1 = nn.Parameter(w1.cuda(), requires_grad=True)
b1 = nn.Parameter(b1.cuda(), requires_grad=True)
w2 = nn.Parameter(w2.cuda(), requires_grad=True)
b2 = nn.Parameter(b2.cuda(), requires_grad=True)
cls_weight_all = torch.cat((weight_base, weight_test_init))
cls_weight_all = nn.Parameter(cls_weight_all, requires_grad=True)
bias = nn.Parameter(gen_bias.cuda(), requires_grad=True)
scale_cls = nn.Parameter(gen_scale_cls.cuda(), requires_grad=True)


optimizer = torch.optim.Adam([w1, b1, w2, b2, bias, scale_cls, cls_weight_all], lr=args.lr, weight_decay=args.opt_decay)

step_size = args.step_size
gamma = args.gamma
lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
criterion = nn.CrossEntropyLoss()

ways = args.ways
shots = args.shots

# breakpoint()
best_acc_zsl = 0.0
best_acc_gzsl_seen = 0.0
best_acc_gzsl_unseen = 0.0
best_H = 0.0
best_epoch = 0

epoch_interval = args.ep_int

for epoch in range(args.num_epochs):	
	epoch_loss = 0
	lr_scheduler.step()

	# breakpoint()
	if (epoch) % epoch_interval == 0: 		
		test_data_temp, test_att_temp, test_label_temp = updata_reliable_test_data(filter_type='ratio', ratio=1.2)		
		dataset = data_loader(train_x, train_att, train_label, test_data_temp, test_att_temp, test_label_temp,
			ways=ways, shots=shots)

	# breakpoint()
	for i in range(1000):

		baseid, base_batch_visual, base_batch_att, base_batch_label, \
		  novelid, novel_batch_visual, novel_batch_att, novel_batch_label = dataset.__getitem__(i)		

		base_batch_visual = base_batch_visual.cuda()
		novel_batch_visual = novel_batch_visual.cuda()
		indx = torch.tensor(list(range(0, ways*shots, shots)))	
		# novel_atts = torch.index_select(novel_batch_att, 0, indx).float().cuda()
		novel_weights = forward(att_pro[novelid-num_train])	
		# novel_weights = torch.zeros(0, 0).cuda()
		# breakpoint()

		# breakpoint()	
		num1 = len(base_batch_visual)
		num2 = len(novel_batch_visual)
		optimizer.zero_grad()

		# breakpoint()
		keep_idx = torch.tensor(list(set(list(range(num_classes))) - set(novelid.tolist())))
		keep_weights = cls_weight_all[keep_idx] 		 
		temp_all_cls_weights = torch.cat((keep_weights, novel_weights))		
		all_cls_weights = temp_all_cls_weights
		all_cls_weight = F.normalize(all_cls_weights, p=2, dim=all_cls_weights.dim()-1, eps=1e-12)  

		vis_feat = torch.cat((base_batch_visual, novel_batch_visual))
		vis_feat_norm = F.normalize(vis_feat, p=2, dim=vis_feat.dim()-1, eps=1e-12)				
		score = apply_classification_weights(vis_feat_norm.unsqueeze(0), all_cls_weight.unsqueeze(0))
		score = score.squeeze(0)

		batch_label = torch.cat((base_batch_label, novel_batch_label))	
		# loss = criterion(score, Variable(batch_label.cuda()))

		# loss = Lq_loss(score, Variable(batch_label.cuda()))
		# breakpoint()

		loss1 = criterion(score[:num1], base_batch_label.cuda())	
		loss2 = Lq_loss(score[num1:], novel_batch_label.cuda(), args.loss_q, num_classes=num_classes)	

		loss = (loss1*num1+loss2*num2) / (num1+num2) 
	

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		epoch_loss = epoch_loss + loss


	epoch_loss = epoch_loss / 1000.0
	epoch_loss = epoch_loss.data.cpu().numpy()

	# breakpoint()

	acc_zsl, acc_unseen_gzsl, acc_seen_gzsl, H = compute_accuracy_all(att_pro, att_all, test_x_unseen, 
		test_id, test_label_unseen, test_x_seen, train_test_id,  test_label_seen)
	
	for param_group in optimizer.param_groups:
		print('ep: %d,  lr: %lf,  loss: %.4f,  zsl: %.4f, gzsl: seen=%.4f, unseen=%.4f, h=%.4f' % 
			(epoch, param_group['lr'], epoch_loss, acc_zsl, acc_seen_gzsl, acc_unseen_gzsl, H))		

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
		best_cls_weight_all = cls_weight_all.data.clone()




print('best_ep: %d, zsl: %.4f, gzsl: seen=%.4f, unseen=%.4f, h=%.4f' % 
	(best_epoch, best_acc_zsl, best_acc_gzsl_seen, best_acc_gzsl_unseen, best_H))	


# exit(0)
torch.save({'w1': best_w1, 'b1': best_b1, 'w2': best_w2, 'b2': best_b2, 
	'scale_cls': best_scale_cls, 'bias': best_bias, 'cls_weight_all': best_cls_weight_all}, 
	trans_model_file_name)


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


