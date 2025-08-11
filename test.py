import torch
import os
import numpy as np
from datasets.crowd import Crowd
from models.vgg import vgg19
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Test ')
    parser.add_argument('--data-dir', default='/home/teddy/UCF-Train-Val-Test', help='training data directory')
    parser.add_argument('--save-dir', default='/home/teddy/vgg', help='model directory')
    parser.add_argument('--device', default='0', help='assign device')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize dataset and dataloader
    datasets = Crowd(os.path.join(args.data_dir, 'test'), 512, 8, is_gray=False, method='val')
    dataloader = torch.utils.data.DataLoader(
        datasets,
        batch_size=1,
        shuffle=False,
        num_workers=0,  # Best practice for Windows and simple setups
        pin_memory=torch.cuda.is_available()
    )

    # Load model
    model = vgg19().to(device)
    model.load_state_dict(torch.load(os.path.join(args.save_dir, 'best_model.pth'), map_location=device))
    model.eval()

    epoch_minus = []
    with torch.no_grad():
        for inputs, count, name in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            temp_minu = count[0].item() - torch.sum(outputs).item()
            print(f"{name}, Diff: {temp_minu:.2f}, Ground Truth: {count[0].item():.2f}, Predicted: {torch.sum(outputs).item():.2f}")
            epoch_minus.append(temp_minu)

    epoch_minus = np.array(epoch_minus)
    mae = np.mean(np.abs(epoch_minus))
    mse = np.sqrt(np.mean(np.square(epoch_minus)))
    log_str = f'Final Test: mae {mae:.2f}, mse {mse:.2f}'
    print(log_str)