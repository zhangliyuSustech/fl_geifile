import torch
import torch.nn as nn
    

class GlobalPredictor(nn.Module):
    
    def __init__(self, num_locs, loc_embedding_dim, num_times, time_embedding_dim, hidden_dim, latent_dim, n_layers=1):
        
        super(GlobalPredictor, self).__init__()
        
        self.loc_embedding = nn.Embedding(num_locs, loc_embedding_dim)
        self.time_embedding = nn.Embedding(num_times, time_embedding_dim)
        self.gru = nn.GRU(loc_embedding_dim + time_embedding_dim, hidden_dim, batch_first=True)
        self.out_layer = nn.Sequential(nn.Linear(hidden_dim, latent_dim),
                             nn.ReLU(),
                             nn.Linear(latent_dim, num_locs))
        
    def predict(self, x_loc, x_t):
        loc_embed = self.loc_embedding(x_loc)
        time_embed = self.time_embedding(x_t)
        out, _ = self.gru(torch.cat([loc_embed, time_embed], dim=-1), None)
        return self.out_layer(out[:, -1])
    
    def predict_prob(self, x_loc, x_t):
        return nn.functional.softmax(self.predict(x_loc, x_t), dim=-1)
    
    def forward(self, x_loc, x_t, y):
        return nn.functional.cross_entropy(self.predict(x_loc, x_t), y, reduction='sum')
    
    
class LocalPredictor(nn.Module):
    
    
    def __init__(self, num_locs, loc_embedding_dim, num_times, time_embedding_dim, hidden_dim, latent_dim, n_layers=1):
        
        super(LocalPredictor, self).__init__()
        
        self.loc_embedding = nn.Embedding(num_locs, loc_embedding_dim)
        self.time_embedding = nn.Embedding(num_times, time_embedding_dim)
        
        self.gru_qry = nn.GRU(loc_embedding_dim + time_embedding_dim, hidden_dim, batch_first=True)
        self.qry_key = nn.Sequential(nn.Linear(hidden_dim, latent_dim),
                                     nn.ReLU())
        self.qry_val = nn.Sequential(nn.Linear(hidden_dim, latent_dim),
                                     nn.ReLU())
        
        self.gru_doc = nn.GRU(loc_embedding_dim + time_embedding_dim, hidden_dim, batch_first=True)
        self.doc_key = nn.Sequential(nn.Linear(hidden_dim, latent_dim),
                                     nn.ReLU())
        self.doc_val = nn.Sequential(nn.Linear(hidden_dim, latent_dim),
                                     nn.ReLU())
        
        self.out_layer = nn.Linear(latent_dim * 2, num_locs)
        self.out_layer_incomplete = nn.Linear(latent_dim, num_locs)
        
        
    def predict(self, x_loc_qry, x_t_qry, x_loc_doc, x_t_doc):
        
        if (x_loc_doc is None) or (x_t_doc is None):
            loc_embed_qry = self.loc_embedding(x_loc_qry)
            time_embed_qry = self.time_embedding(x_t_qry)

            out_qry, _ = self.gru_qry(torch.cat([loc_embed_qry, time_embed_qry], dim=-1), None)
            out_qry_val = self.qry_val(out_qry[:, -1]) #1, L
            
            return self.out_layer_incomplete(out_qry_val)

        else:

            loc_embed_qry = self.loc_embedding(x_loc_qry)
            time_embed_qry = self.time_embedding(x_t_qry)
            loc_embed_doc = self.loc_embedding(x_loc_doc)
            time_embed_doc = self.time_embedding(x_t_doc)

            out_qry, _ = self.gru_qry(torch.cat([loc_embed_qry, time_embed_qry], dim=-1), None)
            out_qry_key = self.qry_key(out_qry[:, -1]) #1, L
            out_qry_val = self.qry_val(out_qry[:, -1]) #1, L
            
            out_doc, _ = self.gru_doc(torch.cat([loc_embed_doc, time_embed_doc], dim=-1), None)
            out_doc_key = self.doc_key(out_doc[:, 3]) #D, L
            out_doc_val = self.doc_val(out_doc[:, -1]) #D, L

            atten = torch.mm(out_qry_key, out_doc_key.transpose(0, 1))
            atten = nn.functional.softmax(atten, dim=-1) #1, D
            doc_val_weighted = torch.mm(atten, out_doc_val) #1, L
            fused_val = torch.cat([out_qry_val, doc_val_weighted], dim=-1)

            return self.out_layer(fused_val)
        
    
    def predict_prob(self, x_loc_qry, x_t_qry, x_loc_doc, x_t_doc):
        return nn.functional.softmax(self.predict(x_loc_qry, x_t_qry, x_loc_doc, x_t_doc), dim=-1)
    
    
    def forward(self, x_loc_qry, x_t_qry, x_loc_doc, x_t_doc, y):
        return nn.functional.cross_entropy(self.predict(x_loc_qry, x_t_qry, x_loc_doc, x_t_doc), y, reduction='sum')
    
    
class LocalPredictorFullSearch(nn.Module):
    
    
    def __init__(self, num_locs, loc_embedding_dim, num_times, time_embedding_dim, hidden_dim, latent_dim, n_layers=1):
        
        super(LocalPredictorFullSearch, self).__init__()
        
        self.latent_dim = latent_dim
        
        self.loc_embedding = nn.Embedding(num_locs, loc_embedding_dim)
        self.time_embedding = nn.Embedding(num_times, time_embedding_dim)
        
        self.gru_qry = nn.GRU(loc_embedding_dim + time_embedding_dim, hidden_dim, batch_first=True)
        self.qry_key = nn.Sequential(nn.Linear(hidden_dim, latent_dim),
                                     nn.ReLU())
        self.qry_val = nn.Sequential(nn.Linear(hidden_dim, latent_dim),
                                     nn.ReLU())
        
        self.gru_doc = nn.GRU(loc_embedding_dim + time_embedding_dim, hidden_dim, batch_first=True)
        self.doc_key = nn.Sequential(nn.Linear(hidden_dim, latent_dim),
                                     nn.ReLU())
        self.doc_val = nn.Sequential(nn.Linear(hidden_dim, latent_dim),
                                     nn.ReLU())
        
        self.out_layer = nn.Linear(latent_dim * 2, num_locs)
        self.out_layer_incomplete = nn.Linear(latent_dim, num_locs)
        
        
    def predict(self, x_loc_qry, x_t_qry, x_loc_doc, x_t_doc):
        
        if (x_loc_doc is None) or (x_t_doc is None):
            loc_embed_qry = self.loc_embedding(x_loc_qry)
            time_embed_qry = self.time_embedding(x_t_qry)

            out_qry, _ = self.gru_qry(torch.cat([loc_embed_qry, time_embed_qry], dim=-1), None)
            out_qry_val = self.qry_val(out_qry[:, -1]) #1, L
            
            return self.out_layer_incomplete(out_qry_val)

        else:

            loc_embed_qry = self.loc_embedding(x_loc_qry)
            time_embed_qry = self.time_embedding(x_t_qry)
            loc_embed_doc = self.loc_embedding(x_loc_doc)
            time_embed_doc = self.time_embedding(x_t_doc)

            out_qry, _ = self.gru_qry(torch.cat([loc_embed_qry, time_embed_qry], dim=-1), None)
            out_qry_key = self.qry_key(out_qry[:, -1]) #1, L
            out_qry_val = self.qry_val(out_qry[:, -1]) #1, L
            
            out_doc, _ = self.gru_doc(torch.cat([loc_embed_doc, time_embed_doc], dim=-1), None)
            out_doc_key = self.doc_key(out_doc[:, 3:-3]).view(-1, self.latent_dim) #D, L
            out_doc_val = self.doc_val(out_doc[:, 6:]).view(-1, self.latent_dim) #D, L

            atten = torch.mm(out_qry_key, out_doc_key.transpose(0, 1))
            atten = nn.functional.softmax(atten, dim=-1) #1, D
            doc_val_weighted = torch.mm(atten, out_doc_val) #1, L
            fused_val = torch.cat([out_qry_val, doc_val_weighted], dim=-1)

            return self.out_layer(fused_val)
        
    
    def predict_prob(self, x_loc_qry, x_t_qry, x_loc_doc, x_t_doc):
        return nn.functional.softmax(self.predict(x_loc_qry, x_t_qry, x_loc_doc, x_t_doc), dim=-1)
    
    
    def forward(self, x_loc_qry, x_t_qry, x_loc_doc, x_t_doc, y):
        return nn.functional.cross_entropy(self.predict(x_loc_qry, x_t_qry, x_loc_doc, x_t_doc), y, reduction='sum')