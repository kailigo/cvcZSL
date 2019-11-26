from scipy import io
import numpy as np
import torch
from pdb import set_trace as breakpoint
import torch.utils.data as data




def data_iterator(train_x, train_att):
	""" A simple data iterator """
	batch_idx = 0
	while True:
		# shuffle labels and features
		idxs = np.arange(0, len(train_x))
		np.random.shuffle(idxs)
		shuf_visual = train_x[idxs]
		shuf_att = train_att[idxs]
		batch_size = 100
		# breakpoint()

		for batch_idx in range(0, len(train_x), batch_size):
			visual_batch = shuf_visual[batch_idx:batch_idx + batch_size]
			visual_batch = visual_batch.astype("float32")
			att_batch = shuf_att[batch_idx:batch_idx + batch_size]

			att_batch = Variable(torch.from_numpy(att_batch).float().cuda())
			visual_batch = Variable(torch.from_numpy(visual_batch).float().cuda())
			yield att_batch, visual_batch



class data_loader(data.Dataset):
	def __init__(self, feats, atts, labels,  ways=32, shots=4):
		self.ways = ways		
		self.shots = shots    

		self.feats = torch.tensor(feats).float()
		self.atts = torch.tensor(atts).float()
		self.labels = labels
		self.classes = np.unique(labels)
		
	def __getitem__(self, index):
		is_first = True
		select_feats = []
		select_atts = []
		select_labels = []        
		select_labels = torch.LongTensor(self.ways*self.shots)
		selected_classes = np.random.choice(list(self.classes), self.ways, False)

		for i in range(self.ways):
			idx = (self.labels==selected_classes[i]).nonzero()[0]
			select_instances = np.random.choice(idx, self.shots, False)
			for j in range(self.shots):
				feat = self.feats[select_instances[j], :]
				att = self.atts[select_instances[j], :]  

				feat = feat.unsqueeze(0)
				att = att.unsqueeze(0)
				# print(feat.size())
				# print(att.size())
				if is_first:
					is_first=False
					select_feats = feat
					select_atts = att
				else:                   
					select_feats = torch.cat((select_feats, feat),0)                
					select_atts = torch.cat((select_atts, att),0)                
				select_labels[i*self.shots+j] = i

		return select_feats, select_atts, select_labels
		
	def __len__(self):
		return self.__size



class data_loader_wt_att(data.Dataset):
	def __init__(self, feats, labels,  ways=32, shots=4):
		self.ways = ways		
		self.shots = shots    

		self.feats = torch.tensor(feats).float()
		self.labels = labels
		self.classes = np.unique(labels)
		
	def __getitem__(self, index):
		is_first = True
		select_feats = []
		select_labels = []        
		select_labels = torch.LongTensor(self.ways*self.shots)
		selected_classes = np.random.choice(list(self.classes), self.ways, False)

		for i in range(self.ways):
			idx = (self.labels==selected_classes[i]).nonzero()[0]
			select_instances = np.random.choice(idx, self.shots, False)
			for j in range(self.shots):
				feat = self.feats[select_instances[j], :]		
				feat = feat.unsqueeze(0)

				if is_first:
					is_first=False
					select_feats = feat					
				else:                   
					select_feats = torch.cat((select_feats, feat),0)                					
				select_labels[i*self.shots+j] = selected_classes[i].item()

		return select_feats, select_labels
		
	def __len__(self):
		return self.__size