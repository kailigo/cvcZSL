
import argparse
import os

class Options():
    def __init__(self):
        # Training settings

        parser = argparse.ArgumentParser(description='Tank Shot')

        parser.add_argument('--dataset', default='CUB1', type=str,
                            help='dataset to be processed')
        
        # parser.add_argument('--batchSize', default=25,type=int,
        #                     help='Batch Size')
        # parser.add_argument('--lr', default=1e-3, type=float,
        #                     help='learning rate')
        parser.add_argument('--step_size', default=200, type=int,
                            help='decay step')
        parser.add_argument('--gamma', default=0.5, type=float,
                            help='decay rate')
        parser.add_argument('--num_epochs', default=500, type=int,
                            help='epoch number')                
        parser.add_argument('--nthreads', default=8,type=int,
                            help='threads num to load data')

        parser.add_argument('--ways', default=32,type=int,
                            help='number of class for one test')
        parser.add_argument('--shots', default=4,type=int,
                            help='number of pictures of each class to support')

        parser.add_argument('--lr', default=1e-5,type=float,
                            help='learning rate')
            
        parser.add_argument('--weight_model', default='weightnet',type=str,
                            help='weight model name after finetuning')

        parser.add_argument('--hidden_dim', default=1600,type=int,
                            help='hidden dimension')

        parser.add_argument('--opt_decay', default=1e-3,type=float,
                            help='decay rate for optimizer')
        
        parser.add_argument("--log_to_file", type=bool, default=True)
        parser.add_argument("--log_file", type=str, default='temp.log')
    
        parser.add_argument("--trans_model_name", type=str, default='trans_model.pt')            
        parser.add_argument("--loss_q", type=float, default=0.5)  

        parser.add_argument("--ep_int", type=int, default=10)  

        parser.add_argument("--wt_model", type=str, default='weight_gen_model.pt')            


        parser.add_argument("--test_shots", default=4, type=int, help='test shots')        
        parser.add_argument("--test_ways", default=32, type=int, help='test ways')                

        # parser.add_argument("--wt_model", type=int, default=10)  
        

        self.parser = parser

    def parse(self):
        return self.parser.parse_args()
