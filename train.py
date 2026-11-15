import argparse
from federated import Config, run

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--out',required=True)
    p.add_argument('--backend',choices=['serial','ray'],default='serial')
    p.add_argument('--rounds',type=int,default=100);p.add_argument('--fraction',type=float,default=.1)
    p.add_argument('--epochs',type=int,default=1);p.add_argument('--lr',type=float,default=.05)
    p.add_argument('--seed',type=int,default=17);p.add_argument('--noise',type=float,default=0.)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu');p.add_argument('--cpus',type=int,default=4)
    p.add_argument('--gpu-per-actor',type=float,default=0.)
    a=vars(p.parse_args());data=a.pop('data');out=a.pop('out');run(data,out,Config(**a))
