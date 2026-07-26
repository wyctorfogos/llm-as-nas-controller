import torch
import torch.nn as nn
import torch.nn.functional as F

class Controller(nn.Module):
    def __init__(self, search_space, hidden_size=64):
        super().__init__()
        self.search_space = search_space
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)
        
        self.param_heads = nn.ModuleDict({})
        self.embeddings = nn.ModuleDict({}) 

        for name, choices in search_space.items():
            self.param_heads[name] = nn.Linear(hidden_size, len(choices))
            self.embeddings[name] = nn.Embedding(num_embeddings=len(choices), embedding_dim=hidden_size)
            
        self.start_token = nn.Parameter(torch.randn(1, hidden_size)) 

    def sample_config(self):
        device = next(self.parameters()).device
        
        h = torch.zeros(1, 1, self.hidden_size, device=device)
        c = torch.zeros(1, 1, self.hidden_size, device=device)
        
        lstm_input_start = self.start_token.unsqueeze(0)        
        out, (h, c) = self.lstm(lstm_input_start, (h, c))
        
        config = {}
        log_probs = []

        for name, head in self.param_heads.items():
            logits = head(out.squeeze(1)) 
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs=probs)
            idx = dist.sample() # idx agora tem shape torch.Size([1])
                        
            config[name] = self.search_space[name][idx.item()]
            log_probs.append(dist.log_prob(idx))
            
            # A camada Embedding espera um tensor de índices.
            current_choice_embedding = self.embeddings[name](idx) 
            
            # ele se torna (1, 1, hidden_size), que é o formato correto para o LSTM (batch, seq_len, features).
            lstm_input_choice = current_choice_embedding.unsqueeze(0) 
            
            out, (h, c) = self.lstm(lstm_input_choice, (h, c))

        return config, torch.stack(log_probs).sum()