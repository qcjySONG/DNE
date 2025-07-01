import torch  
import torch.nn as nn  
from torch.utils.data import Dataset, DataLoader  
import pickle  
import numpy as np  

# --------------------------  
# Set device  
# --------------------------  
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  

# --------------------------  
# Load data and apply log transformation  
# --------------------------  
with open('/home/amax/qcjySONG/tarindata.pkl', 'rb') as f:  
    data_np = pickle.load(f)  # numpy array, shape = (988, 23)  

# Apply log1p transformation to the original data to handle zero values  
data_np_log = np.log1p(data_np)  

# Convert to tensor  
data_log = torch.tensor(data_np_log, dtype=torch.float32)  

# --------------------------  
# Z-score normalization (in log space)  
# --------------------------  
mean = data_log.mean(dim=0)    # (23,)  
std = data_log.std(dim=0)      # (23,)  
data_normalized = (data_log - mean) / std  

# --------------------------  
# Hyperparameters  
# --------------------------  
input_window = 52    # Past 52 weeks  
output_window = 156  # Future 156 weeks  
batch_size = 4  
d_model = 128  
nhead = 8  
num_layers = 3  
dim_feedforward = 256  
dropout = 0.1  
learning_rate = 0.0001  
num_epochs = 100  
num_regions = 23  

# --------------------------  
# Custom Dataset  
# --------------------------  
class DiseaseDataset(Dataset):  
    def __init__(self, data, input_window, output_window):  
        self.data = data  
        self.input_window = input_window  
        self.output_window = output_window  
        
    def __len__(self):  
        return len(self.data) - self.input_window - self.output_window + 1  

    def __getitem__(self, idx):  
        x = self.data[idx : idx + self.input_window]  
        y = self.data[idx + self.input_window : idx + self.input_window + self.output_window]  
        return x, y  

# --------------------------  
# Positional Encoding  
# --------------------------  
class PositionalEncoding(nn.Module):  
    def __init__(self, d_model, max_len=5000):  
        super(PositionalEncoding, self).__init__()  
        
        pe = torch.zeros(max_len, d_model)  
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))  
        
        pe[:, 0::2] = torch.sin(position * div_term)  
        pe[:, 1::2] = torch.cos(position * div_term)  
        
        pe = pe.unsqueeze(0)  
        self.register_buffer('pe', pe)  

    def forward(self, x):  
        x = x + self.pe[:, :x.size(1)]  
        return x  

# --------------------------  
# Transformer Model  
# --------------------------  
class DiseaseTransformer(nn.Module):  
    def __init__(self, input_size, d_model, nhead, num_layers, dim_feedforward, output_size, dropout=0.1):  
        super(DiseaseTransformer, self).__init__()  
        self.input_proj = nn.Linear(input_size, d_model)  
        self.pos_encoder = PositionalEncoding(d_model)  
        
        encoder_layer = nn.TransformerEncoderLayer(  
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)  
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)  
        
        self.decoder = nn.Linear(d_model, output_size)  

    def forward(self, src):  
        src = self.input_proj(src)  
        src = self.pos_encoder(src)  
        memory = self.transformer_encoder(src)  
        # Use the output of the last time step of the sequence for prediction  
        out = memory[:, -1, :]  
        out = self.decoder(out)  
        return out  

# --------------------------  
# Prepare data loading  
# --------------------------  
dataset = DiseaseDataset(data_normalized, input_window, output_window)  
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)  

# --------------------------  
# Instantiate the model  
# --------------------------  
model = DiseaseTransformer(  
    input_size=num_regions,  
    d_model=d_model,  
    nhead=nhead,  
    num_layers=num_layers,  
    dim_feedforward=dim_feedforward,  
    output_size=num_regions * output_window,  
    dropout=dropout  
).to(device)  

# Loss function and optimizer  
criterion = nn.MSELoss()  
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

# --------------------------  
# Train the model  
# --------------------------  
for epoch in range(num_epochs):  
    model.train()  
    total_loss = 0  
    for x_batch, y_batch in dataloader:  
        x_batch = x_batch.to(device)  # (batch, input_window, num_regions)  
        y_batch = y_batch.to(device)  # (batch, output_window, num_regions)  

        optimizer.zero_grad()  
        output = model(x_batch)  # (batch, num_regions * output_window)  
        output = output.view(-1, output_window, num_regions)  # (batch, output_window, num_regions)  
        
        loss = criterion(output, y_batch)  
        loss.backward()  
        optimizer.step()  
        
        total_loss += loss.item()  
    
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader):.6f}')  

# --------------------------  
# Predict the next 156 weeks  
# --------------------------  
model.eval()  
with torch.no_grad():  
    recent_data = data_normalized[-input_window:].unsqueeze(0).to(device)  # (1, 52, 23)  
    pred = model(recent_data)  # (1, 23*156)  
    pred = pred.view(output_window, num_regions)  # (156, 23)  
    pred = pred.cpu()  

# --------------------------  
# Inverse normalization and inverse log to restore the actual counts  
# --------------------------  
pred_log = pred * std + mean       # Inverse normalization (in log space)  
pred_real = torch.expm1(pred_log)  # exp(x) - 1, restore to original positive space  
# Prevent negative values (should not occur in theory, set to 0 if they do)  
pred_real = torch.clamp(pred_real, min=0)  
# Set values less than 1 to 0, and take the ceiling for values greater than or equal to 1  
pred_real = torch.where(pred_real < 1, torch.tensor(0., device=pred_real.device), torch.ceil(pred_real))  
pred_real_np = pred_real.numpy()  

# --------------------------  
# Save prediction results to a pkl file  
# --------------------------  
with open('/home/amax/qcjySONG/future_prediction_52.pkl', 'wb') as f:  
    pickle.dump(pred_real_np, f)  
