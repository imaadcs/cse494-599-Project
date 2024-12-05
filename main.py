import argparse
import os
import sys
import csv
import time
from collections import deque
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from data_loader import define_dataloader, load_precomputed_embeddings
from utils import str2bool, timeSince, get_performance_batchiter, print_performance, write_blackbox_output_batchiter

# Constants
PRINT_EVERY_EPOCH = 1

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for x_pep, x_tcr, y in train_loader:
        x_pep, x_tcr, y = x_pep.to(device), x_tcr.to(device), y.to(device)
        optimizer.zero_grad()
        yhat = model(x_pep, x_tcr).squeeze(-1)  # Ensure output is [batch_size]
        loss = F.binary_cross_entropy(yhat, y.float())  # Ensure y is float for BCE
        loss.backward()
        optimizer.step()

    if epoch % PRINT_EVERY_EPOCH == 1:
        print(f'[TRAIN] Epoch {epoch} Loss {loss.item():.4f}')

def test(model, device, test_loader, output_file):
    """
    Evaluate the model on the test set and save performance metrics to a file.
    """
    model.eval()
    print("Evaluating the model on the test set...")
    performance = get_performance_batchiter(test_loader['loader'], model, device)

    # Save performance metrics to a file
    print(f"Saving performance metrics to {output_file}...")
    with open(output_file, 'w', newline='') as wf:
        writer = csv.writer(wf)
        writer.writerow(['Metric', 'Value'])
        for metric, value in performance.items():
            writer.writerow([metric, value])
    print(f"Performance metrics saved to {output_file}")
    
    return performance

def save_predictions(model, device, test_loader, output_file):
    """
    Save predictions to a CSV file.
    """
    print(f"Saving predictions to {output_file}...")
    with open(output_file, 'w', newline='') as wf:
        write_blackbox_output_batchiter(test_loader, model, wf, device, ifscore=True)
    print(f"Predictions saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Prediction of TCR binding to peptide-MHC complexes')
    parser.add_argument('--infile', type=str, help='Input file for training (precomputed embeddings)')
    parser.add_argument('--indepfile', type=str, default=None, help='Independent test data file')
    parser.add_argument('--batch_size', type=int, default=32, help='Training batch size')
    parser.add_argument('--model_name', type=str, default='catelmo.ckpt', help='Model name to save or load')
    parser.add_argument('--epoch', type=int, default=200, help='Maximum number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--cuda', type=str2bool, default=True, help='Enable CUDA')
    parser.add_argument('--seed', type=int, default=1039, help='Random seed')
    parser.add_argument('--mode', type=str, default='train', help='Mode: train or test')
    parser.add_argument('--save_model', type=str2bool, default=True, help='Whether to save the model after training')
    parser.add_argument('--drop_rate', type=float, default=0.25, help='Dropout rate for dense layers')
    parser.add_argument('--lin_size', type=int, default=1024, help='Size of the linear transformation layers')
    parser.add_argument('--heads', type=int, default=8, help='Number of attention heads for MultiheadAttention')
    args = parser.parse_args()

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')

    # Set random seed
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # Training mode
    if args.mode == 'train':
        assert args.infile is not None, "Training mode requires --infile"
        x_pep, x_tcr, y = load_precomputed_embeddings(args.infile)
        train_loader = define_dataloader(x_pep, x_tcr, y, batch_size=args.batch_size, device=device)

        # Initialize model
        from attention import Net
        model = Net(args).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        # Train model
        t0 = time.time()
        for epoch in range(1, args.epoch + 1):
            train(model, device, train_loader['loader'], optimizer, epoch)
            if epoch % PRINT_EVERY_EPOCH == 0:
                print(f'Epoch {epoch} completed.')

        # Save model
        if args.save_model:
            model_name = f'./models/{args.model_name}'
            torch.save(model.state_dict(), model_name)
            print(f'Model saved to {model_name}')
        print(f'Training time: {timeSince(t0)}')

    # Testing mode
    elif args.mode == 'test':
        assert args.indepfile is not None, "Testing mode requires --indepfile"
        model_name = f'./models/{args.model_name}'
        assert os.path.exists(model_name), f"Model file {model_name} not found!"

        # Load model
        from attention import Net
        model = Net(args).to(device)
        model.load_state_dict(torch.load(model_name, map_location=device))
        print(f"Loaded model from {model_name}")

        # Load independent test data
        x_test_pep, x_test_tcr, y_test = load_precomputed_embeddings(args.indepfile)
        test_loader = define_dataloader(x_test_pep, x_test_tcr, y_test, batch_size=args.batch_size, device=device)

        # Evaluate and save predictions and performance
        # Evaluate and save predictions
        test_performance = test(
            model, 
            device, 
            test_loader, 
            output_file=f'./result/perf_{os.path.splitext(args.model_name)[0]}.csv'
        )
        predictions_file = f'./result/pred_{os.path.splitext(args.model_name)[0]}.csv'
        save_predictions(model, device, test_loader, predictions_file)

    else:
        raise ValueError('Invalid mode. Use --mode train or --mode test.')

if __name__ == '__main__':
    main()
