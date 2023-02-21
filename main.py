import torch
from torch import optim
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
from config import Config
from data_loader import Dataset
from model import AutoEncoder
import arguments
from trainer import train
from testing import test
from utils import save_best_record

if __name__ == '__main__':
    args = arguments.parser.parse_args()
    config = Config(args)

    train_nloader = DataLoader(Dataset(args, test_mode=False, is_normal=True),
                               batch_size=args.batch_size, shuffle=True,
                               num_workers=0, pin_memory=False, drop_last=True)
    train_aloader = DataLoader(Dataset(args, test_mode=False, is_normal=False),
                               batch_size=args.batch_size, shuffle=True,
                               num_workers=0, pin_memory=False, drop_last=True)
    test_loader = DataLoader(Dataset(args, test_mode=True),
                              batch_size=1, shuffle=False,
                              num_workers=0, pin_memory=False)
    model = AutoEncoder(args.feature_size, args.batch_size)

    for name, value in model.named_parameters():
        print(name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    if args.continue_training is not None:
        model.load_state_dict(torch.load(args.continue_training))

    if args.test == 1:
        model.load_state_dict(torch.load(args.modelpath))
        auc = test(test_loader, model, args, None, device)

    else:
      if not os.path.exists('./ckpt'):
          os.makedirs('./ckpt')

      optimizer = optim.Adam(model.parameters(),
                              lr=config.lr[0], weight_decay=0.005)

      test_info = {"epoch": [], "test_AUC": []}
      best_AUC = -1
      output_path = 'results'   # put your own path here
      if not os.path.exists('./results'):
          os.makedirs('./results')
      #auc = test(test_loader, model, args, None, device)
      auc_history = []
      old_model = None
      for step in tqdm(
              range(1, args.max_epoch + 1),
              total=args.max_epoch,
              dynamic_ncols=True
      ):
          if step > 1 and config.lr[step - 1] != config.lr[step - 2]:
              for param_group in optimizer.param_groups:
                  param_group["lr"] = config.lr[step - 1]

          if (step - 1) % len(train_nloader) == 0:
              loadern_iter = iter(train_nloader)

          if (step - 1) % len(train_aloader) == 0:
              loadera_iter = iter(train_aloader)

          train(loadern_iter, loadera_iter, model, args.batch_size, optimizer, None, device)
          
          if step % 1 == 0 and step >= args.save_step:

              auc = test(test_loader, model, args, None, device)
              auc_history.append(auc)
              test_info["epoch"].append(step)
              test_info["test_AUC"].append(auc)

              if test_info["test_AUC"][-1] > best_AUC and test_info["test_AUC"][-1]>0.8:
                  if old_model is not None:
                    os.remove(old_model)
                  model_path = './ckpt/' + args.model_name + '{}-i3d.pkl'.format(step)
                  best_AUC = test_info["test_AUC"][-1]
                  torch.save(model.state_dict(), model_path)
                  save_best_record(test_info, os.path.join(output_path, '{}-{}-step-AUC.txt'.format(args.model_name,step)))
                  old_model = model_path
              with open('auc_history.txt', 'w') as f:
                  f.write(str(auc_history)) 
        
      torch.save(model.state_dict(), './ckpt/' + args.model_name + 'final.pkl')