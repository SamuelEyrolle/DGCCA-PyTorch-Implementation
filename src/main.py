import argparse
import torch
import torch.nn.init as init
import os
import pandas as pd
import json
from model import DeepGCCA, generalised_gcca_loss
from utils import set_seed, load_views 

def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        init.xavier_normal_(m.weight)
        if m.bias is not None:
            init.constant_(m.bias, 0)

def main():
    parser = argparse.ArgumentParser(description='Deep GCCA Training Pipeline')
    
    # Paths
    parser.add_argument('--input_dir', type=str, required=True, help='Directory with view1.csv, view2.csv...')
    parser.add_argument('--output_dir', type=str, default='./results', help='Directory to save results')
    
    # Reproducibility & Device
    parser.add_argument('--seed', type=int, default=42)
    
    # Architecture
    parser.add_argument('--hidden_layers', type=int, nargs='+', default=[32, 64], help='List of hidden layer sizes')
    parser.add_argument('--latent_dims', type=int, default=8, help='Output dimension of each encoder')
    
    # Training Hyperparameters
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    
    # DGCCA Loss Parameters
    parser.add_argument('--top_k', type=int, default=8, help='Number of components for GCCA correlation')
    parser.add_argument('--eps', type=float, default=1e-8, help='Stability constant for SVD scaling')
    
    # Optimizer Scheduler
    parser.add_argument('--step_size', type=int, default=100)
    parser.add_argument('--gamma', type=float, default=0.1)
    
    args = parser.parse_args()

    # Ensure output directory exists and save config receipt
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load and move data to device
    views = load_views(args.input_dir) 
    views = [v.to(device) for v in views]

    # Build configurations for each encoder dynamically
    view_configs = []
    for v in views:
        view_configs.append({
            'input_size': v.shape[1],
            'layer_sizes': args.hidden_layers + [args.latent_dims]
        })

    # Initialise model and weights
    model = DeepGCCA(view_configs, device)
    model.apply(init_weights)

    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=args.lr, 
        betas=(args.beta1, args.beta2), 
        weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    model.train()
    train_loss_history = []

    print(f"Starting training on {device}...")
    for epoch in range(args.epochs):
        optimizer.zero_grad()
        outputs = model(views)
        loss, G = generalised_gcca_loss(outputs, top_k=args.top_k, eps=args.eps)
        
        loss.backward()
        optimizer.step()
        scheduler.step()

        train_loss_history.append(loss.item())

        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch+1}/{args.epochs}] - Loss: {loss.item():.4f}")

    # --- FINAL DATA CAPTURE ---
    # Convert tensors back to numpy for saving
    final_G_np = G.detach().cpu().numpy()
    final_outputs_np = [out.detach().cpu().numpy() for out in outputs]

    # --- SAVING ARTIFACTS ---
    # 1. Shared Representation G
    pd.DataFrame(final_G_np).to_csv(os.path.join(args.output_dir, "shared_G.csv"), index=False, header=False)
    
    # 2. Individual Encoded Views H_n
    for i, out_np in enumerate(final_outputs_np):
        pd.DataFrame(out_np).to_csv(os.path.join(args.output_dir, f"encoded_view_{i+1}.csv"), index=False, header=False)
    
    # 3. Training Loss History
    loss_data = {
        'epoch': range(1, len(train_loss_history) + 1),
        'loss': train_loss_history
    }
    pd.DataFrame(loss_data).to_csv(os.path.join(args.output_dir, "loss_history.csv"), index=False)
    
    # 4. Trained Model State
    torch.save(model.state_dict(), os.path.join(args.output_dir, "model_weights.pth"))
    
    print(f"✅ Training complete. Results saved in: {args.output_dir}")

if __name__ == "__main__":
    main()