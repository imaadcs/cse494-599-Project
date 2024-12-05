import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()

        self.attn_tcr = nn.MultiheadAttention(embed_dim=1024, num_heads=args.heads)
        self.attn_pep = nn.MultiheadAttention(embed_dim=1024, num_heads=args.heads)

        # Dense Layer
        self.size_hidden1_dense = 2 * args.lin_size
        self.size_hidden2_dense = 1 * args.lin_size
        self.net = nn.Sequential(
            nn.Linear(2048, self.size_hidden1_dense),  # Adjusted input size
            nn.LayerNorm(self.size_hidden1_dense),  # Use LayerNorm
            nn.Dropout(args.drop_rate * 2),
            nn.SiLU(),
            nn.Linear(self.size_hidden1_dense, self.size_hidden2_dense),
            nn.LayerNorm(self.size_hidden2_dense),  # Use LayerNorm
            nn.Dropout(args.drop_rate),
            nn.SiLU(),
            nn.Linear(self.size_hidden2_dense, 1),
            nn.Sigmoid(),
        )

    def forward(self, pep, tcr):
        # Ensure correct dimensions for attention
        if pep.dim() == 2:
            pep = pep.unsqueeze(1)  # [batch_size, 1, embed_dim]
        if tcr.dim() == 2:
            tcr = tcr.unsqueeze(1)  # [batch_size, 1, embed_dim]

        # Apply attention
        pep = pep.transpose(0, 1)  # [seq_len, batch_size, embed_dim]
        tcr = tcr.transpose(0, 1)  # [seq_len, batch_size, embed_dim]
        pep, _ = self.attn_pep(pep, pep, pep)
        tcr, _ = self.attn_tcr(tcr, tcr, tcr)

        # Reshape and concatenate
        pep = pep.mean(dim=0)  # [batch_size, embed_dim]
        tcr = tcr.mean(dim=0)  # [batch_size, embed_dim]
        peptcr = torch.cat((pep, tcr), dim=-1)  # [batch_size, embed_dim * 2]

        # Pass through the dense layers
        peptcr = self.net(peptcr)  # [batch_size, 1]

        return peptcr
