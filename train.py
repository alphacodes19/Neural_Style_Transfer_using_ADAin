import argparse
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from pathlib import Path
from utils.utils import *
from utils.models import *
from utils.data_prep import load_split_manifests
from tqdm import tqdm
from torchvision.utils import save_image


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--manifest_dir', type=str, default=None,
                        help='Directory with content_train/val.txt and style_train/val.txt '
                             '(from prepare_data.py). Preferred over --content_dir/--style_dir.')
    parser.add_argument('--content_dir', type=str, default=None,
                        help='Content image folder (searched recursively); used if no --manifest_dir')
    parser.add_argument('--style_dir', type=str, default=None,
                        help='Style image folder (searched recursively); used if no --manifest_dir')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='DataLoader worker processes')
    parser.add_argument('--vgg', type=str, default=str(Path('weights') / 'vgg_normalised.pth'),
                        help='Location of pre-trained VGG (default: weights/vgg_normalised.pth)')
    parser.add_argument('--experiment', type=str, default='experiment1',
                        help='Name of experiment')
    
    parser.add_argument('--final_size', type=int, default=256,
                        help='Size of final image')
    parser.add_argument('--content_size', type=int, default=512,
                        help='Size of content image')
    parser.add_argument('--style_size', type=int, default=512,
                        help='Size of style image')
    parser.add_argument('--crop', action='store_true', default=True,
                        help='Crop image')
    
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--lr_decay', type=float, default=5e-5,
                        help='Learning rate decay')
    
    parser.add_argument('--epochs', type=int, default=1,
                        help='Number of epochs')
    
    parser.add_argument('--content_weight', type=float, default=1.0,
                        help='Content weight')
    parser.add_argument('--style_weight', type=float, default=5,
                        help='Style weight')
    
    parser.add_argument('--log_interval', type=int, default=1,
                        help='Log interval')
    
    parser.add_argument('--save_interval', type=int, default=2,
                        help='Save interval')
    
    parser.add_argument('--resume', action='store_true', default=False,
                        help='Resume training')
    
    parser.add_argument('--decoder_path', type=str, default=None,
                        help='Path to decoder checkpoint')
    
    parser.add_argument('--optimizer_path', type=str, default=None,
                        help='Path to optimizer checkpoint')
    

    args = parser.parse_args()
    if not args.manifest_dir and not (args.content_dir and args.style_dir):
        parser.error('provide --manifest_dir, or both --content_dir and --style_dir')
    return args


def main():
    args = parse_arguments()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path('experiment') / args.experiment
    save_dir.mkdir(exist_ok=True, parents=True)

    #Save argument values
    with open(save_dir / 'args.txt', 'w') as args_file:
        for key, value in vars(args).items():
            args_file.write(f'{key}: {value}\n')
    
    content_transform = get_transform(args.content_size, args.crop, args.final_size)
    style_transform = get_transform(args.style_size, args.crop, args.final_size)
    
    if args.manifest_dir:
        splits = load_split_manifests(args.manifest_dir)
        content_dataset = ImageFolderDataset(splits['content_train'], content_transform)
        style_dateset = ImageFolderDataset(splits['style_train'], style_transform)
        counts = {k: len(v) for k, v in splits.items()}
        print(f'Manifests: {args.manifest_dir}')
    else:
        content_dataset = ImageFolderDataset(args.content_dir, content_transform)
        style_dateset = ImageFolderDataset(args.style_dir, style_transform)
        counts = {'content_train': len(content_dataset), 'content_val': 'n/a (no manifest)',
                  'style_train': len(style_dateset), 'style_val': 'n/a (no manifest)'}
    print(f"Content train: {counts['content_train']}")
    print(f"Content val:   {counts['content_val']}")
    print(f"Style train:   {counts['style_train']}")
    print(f"Style val:     {counts['style_val']}")

    loader_kwargs = dict(batch_size=args.batch_size, shuffle=True, pin_memory=True,
                         drop_last=True, num_workers=args.num_workers,
                         persistent_workers=args.num_workers > 0)
    content_dataloader = DataLoader(content_dataset, **loader_kwargs)
    style_dataloader = DataLoader(style_dateset, **loader_kwargs)
    
    print('Number of batches in content dataset: ', len(content_dataloader))
    print('Number of batches in style dataset: ', len(style_dataloader))
    
    if not Path(args.vgg).is_file():
        raise FileNotFoundError(
            f"VGG weights not found: {args.vgg}\n"
            "Pass --vgg pointing at vgg_normalised.pth, or place it at "
            "weights/vgg_normalised.pth."
        )
    encoder = VGGEncoder(args.vgg, map_location=device).to(device)
    decoder = Decoder().to(device)

    optimizer = optim.Adam(decoder.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda = lambda epoch:  1.0 / (1.0 + args.lr_decay * epoch)
    )

    if args.resume:
        if not args.decoder_path or not Path(args.decoder_path).is_file():
            raise FileNotFoundError(
                f"--resume was given but decoder checkpoint was not found: "
                f"{args.decoder_path!r}. Pass --decoder_path pointing at an "
                "existing decoder_latest.pth / decoder_final.pth."
            )
        if not args.optimizer_path or not Path(args.optimizer_path).is_file():
            raise FileNotFoundError(
                f"--resume was given but optimizer checkpoint was not found: "
                f"{args.optimizer_path!r}. Pass --optimizer_path pointing at an "
                "existing checkpoint_latest.pth."
            )
        decoder.load_state_dict(torch.load(args.decoder_path, map_location=device))
        optimizer.load_state_dict(torch.load(args.optimizer_path, map_location=device))

    print('Training...')

    mse_loss = torch.nn.MSELoss()

    encoder.eval()

    running_loss = None
    running_closs = None
    running_sloss = None

    for epoch in range(args.epochs):
        progress_bar = tqdm(zip(content_dataloader, style_dataloader),
                            total=min(len(content_dataloader), len(style_dataloader)))

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

            progress_bar.set_description(f'Loss:{loss.item():4f}, Content Loss: {loss_c.item():4f}, Style Loss: {loss_s.item():4f}')

            running_loss += loss.item()
            running_closs += loss_c.item()
            running_sloss += loss_s.item()
        
        scheduler.step()

        running_loss /= len(content_dataloader)
        running_closs /= len(content_dataloader)
        running_sloss /= len(content_dataloader)

        if (epoch+1) % args.log_interval == 0:
            tqdm.write(f'Iter {epoch+1}: Loss:{running_loss:4f}, Content Loss: {running_closs:4f}, Style Loss: {running_sloss:4f}')

        if (epoch+1) % args.save_interval == 0:
            torch.save(decoder.state_dict(), save_dir / 'decoder_latest.pth')
            torch.save(optimizer.state_dict(), save_dir / 'checkpoint_latest.pth')

            with torch.no_grad():
                output = torch.cat([content_batch, style_batch, g], dim=0)
                save_image(output, save_dir / f'output_{epoch+1}.png', nrow=args.batch_size)

    # Phase 1 checkpoint contract: a final decoder checkpoint must always exist
    # after training, even if epochs is small or save_interval was never hit.
    torch.save(decoder.state_dict(), save_dir / 'decoder_final.pth')
    torch.save(optimizer.state_dict(), save_dir / 'checkpoint_final.pth')
    print(f'Saved final decoder checkpoint to {save_dir / "decoder_final.pth"}')


if __name__ == '__main__':
    main()