import torch.nn as nn

class Projector(nn.Module):
    def __init__(self, classi_emb_dim, clip_emb_dim, proj_model='linear', proj_n_layers=1, proj_activ='relu', use_relu=True):
        super().__init__()
        self.proj_model = proj_model
        self.proj_n_layers = proj_n_layers
        self.proj_activ = proj_activ
        print(f'proj_model: {proj_model}, proj_n_layers: {proj_n_layers}, proj_activ: {proj_activ}, use_relu: {use_relu}')

        if self.proj_model == 'linear':
            self.proj = nn.Linear(classi_emb_dim, clip_emb_dim)
            if use_relu:
                self.inv_proj = nn.Sequential(
                    nn.Linear(clip_emb_dim, classi_emb_dim),
                    nn.ReLU()
                )
            else:
                self.inv_proj = nn.Sequential(
                    nn.Linear(clip_emb_dim, classi_emb_dim),
                )
        else:
            hidden_dim = int((classi_emb_dim * clip_emb_dim) ** 0.5)
            n_hidden = self.proj_n_layers - 2
    
            self.proj = nn.Sequential(
                nn.Linear(classi_emb_dim, hidden_dim),
                nn.ReLU(),
                *[nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU()) for _ in range(n_hidden)],
                nn.Linear(hidden_dim, clip_emb_dim)
            )
            self.inv_proj = nn.Sequential(
                nn.Linear(clip_emb_dim, hidden_dim),
                nn.ReLU(),
                *[nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU()) for _ in range(n_hidden)],
                nn.Linear(hidden_dim, classi_emb_dim),
                nn.ReLU()
            )
    
    def forward(self, classi_emb, clip_emb):
        return self.proj(classi_emb), self.inv_proj(clip_emb), None

            
