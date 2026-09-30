import argparse
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from pathlib import Path
from utils.utils import *
from utils.models import *
from tqdm import tqdm


def parse_arguments():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--content_dir', type = str, default ='', help = 'Location of content dataset')
    
    parser.add_argument('--style_dir', type = str. default = '', help = 'Loacation of style dataset' )
    
    parser.add_argument('--vgg', type = str, default ='', help ='Location of pre defined VGG')
    
    parser.add_Argument('--experiment', type = str, default = 'experiment1', help = 'Name of experiment')
    
    parser.add_argument('--final_size', type = int, default = 256, help = 'Size of final image')
    
    parser.add_argument('--content_size', type = int, default = 512, help = 'Size of content image')
    
    parser.add_argument('--style_size', type = int, default = 512, help = 'Size of style image')
    
    parser.add_argument('--crop', action = 'store_true', default = True, help = 'Crop image')
    
    parser.add_argument('--batch_size', type = int, default = 4, help = 'Batch size')
    
    parser.add_argument('--lr', type = float, default = le-4, help = 'Learning Rate')
    
    parser.add_argument('--lr_decay', type = float, default = 5e-5, help = 'Learning rate decay')
    
    parser.add_argument('--epochs', type = int, default = 2, help = 'Number of epochs')
    
    return parser.parse_args()



def main():
    args = parse_arguments()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    save_dir = Path('experiment') / args.experiment
    
    save_dir.mkdir(exist_ok = True, parents = True)
    
    #Save argument values
    with open(save_dir / 'args.txt', 'w') as args file:
        for key, value in vars(args).items():
            args_file.write(f'{key}: {value}\n')
            
    content_transform = get_transform(args.content_size, args.crop, args.final_size)
    style_transform = get_transform(args.style_size, args.crop, args.final_size)
           
    content_dataset = ImageFolderDataset(args.content_dir, content_transform)
    style_dataset = ImageFolderDataset(args.style_dir, style_transform)
    
    content_dataloader = DataLoader(content_dataset, batch_size = args.batch_size, shuffle = True, pin_memory = True, drop_last = True)
    style_dataloader = DataLoader(style_dataset, batch_size = args.batch_size, shuffle = True, pin_memory = True, drop_last = True)
    
    print('Number of batches in content dataset:', len(content_dataloader))
    print('Number of batches in style dataset:', len(style_dataloader))
    
    encoder = VGGEncoder(args.vgg).to(device)
    decoder = Decoder().to(device)
    
    optimizer = optim.Adam(decoder.parameters(), lr = args.lr)
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: 1.0 / (1.0 + args.lr_decay * epoch)
    )
    
    mse_loss = torch.nn.MSELoss()
    
    encoder.eval()
    
    running_loss = None
    running_class = None
    running_sloss = None
    
    
    for epoch in range(args.epochs):
        progress_bar = tqdm(zip(content_dataloader, style_dataloader),
                            total = min(len(content_dataloader), len(style_dataloader)))
        
        running_loss = 0
        running_closs = 0
        running_sloss = 0
        
        for content_batch, style_batch in progress_bar:
            
            content_batch = content_batch.to(device)
            style_batch = style_batch.to(device)
            
            c_feats = encoder(content_batch)
            s_feats = encoder(style_batch)
            
             
            t = adaptive_instance_normalization(c_feats[-1], s_feats[-1])
            
            g = decoder(t)
            
            g_feats = encoder(g)
            
            loss_c = mse_loss(g_feats[-1], t) * args.content_weight
            
            loss_s = 0
            for g_f, s_f in zip(g_feats, s_feats):
                g_mean, g_std = calc_mean_std(g_f)
                s_mean, s_std = calc_mean_std(s_f)
                loss_s += mse_loss(g_mean, s_mean) + mse_loss(g_std, s_std)
                
            loss_s = loss_s * args.style_weight
            
            loss = loss_c + loss_s
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            running_closs += loss_c.item()
            running_sloss += loss_s.item()
        
        scheduler.step()
        
        running_loss /= len(content_dataloader)
        running_closs /= len(content_dataloader)
        running_sloss /= len(content_dataloader)
        
        if (epoch+1) % args.log_interval == 0:
            tqdm.write(f'Iter {epoch + 1} : Loss')
            
            
            
            
    

if __name__ == '__main__':
    main()