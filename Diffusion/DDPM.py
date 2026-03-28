import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import time
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# UTKFace 데이터셋
class UTKFaceDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.image_paths = []
        self.transform = transform
        
        for filename in os.listdir(data_dir):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                parts = filename.split('_')
                if len(parts) >= 3:
                    try:
                        age = int(parts[0])
                        gender = int(parts[1])
                        race = int(parts[2])
                        if 0 <= age <= 116 and gender in [0, 1] and 0 <= race <= 4:
                            self.image_paths.append(os.path.join(data_dir, filename))
                    except ValueError:
                        continue
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

# U-Net 기반 노이즈 예측 네트워크
class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.linear1 = nn.Linear(dim, dim * 4)
        self.linear2 = nn.Linear(dim * 4, dim * 4)
    
    def forward(self, t):
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        emb = self.linear1(emb)
        emb = nn.functional.silu(emb)
        emb = self.linear2(emb)
        return emb

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Linear(time_dim, out_ch)
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
    
    def forward(self, x, t):
        h = self.conv1(nn.functional.silu(x))
        h = h + self.time_mlp(nn.functional.silu(t))[:, :, None, None]
        h = self.norm1(h)
        h = self.conv2(nn.functional.silu(h))
        h = self.norm2(h)
        return h + self.shortcut(x)

class UNet(nn.Module):
    def __init__(self, img_channels=3, time_dim=256):
        super().__init__()
        self.time_embed = TimeEmbedding(time_dim)
        
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(img_channels, 64, 3, padding=1),
            ResBlock(64, 64, time_dim * 4),
            ResBlock(64, 64, time_dim * 4)
        )
        self.down1 = nn.Conv2d(64, 64, 4, 2, 1)
        
        self.enc2 = nn.Sequential(
            ResBlock(64, 128, time_dim * 4),
            ResBlock(128, 128, time_dim * 4)
        )
        self.down2 = nn.Conv2d(128, 128, 4, 2, 1)
        
        self.enc3 = nn.Sequential(
            ResBlock(128, 256, time_dim * 4),
            ResBlock(256, 256, time_dim * 4)
        )
        self.down3 = nn.Conv2d(256, 256, 4, 2, 1)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            ResBlock(256, 512, time_dim * 4),
            ResBlock(512, 512, time_dim * 4)
        )
        
        # Decoder
        self.up3 = nn.ConvTranspose2d(512, 256, 4, 2, 1)
        self.dec3 = nn.Sequential(
            ResBlock(512, 256, time_dim * 4),
            ResBlock(256, 256, time_dim * 4)
        )
        
        self.up2 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.dec2 = nn.Sequential(
            ResBlock(256, 128, time_dim * 4),
            ResBlock(128, 128, time_dim * 4)
        )
        
        self.up1 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.dec1 = nn.Sequential(
            ResBlock(128, 64, time_dim * 4),
            ResBlock(64, 64, time_dim * 4)
        )
        
        self.final = nn.Conv2d(64, img_channels, 3, padding=1)
    
    def forward(self, x, t):
        t = self.time_embed(t)
        
        # Encoder
        e1 = self.enc1[0](x)
        for block in self.enc1[1:]:
            e1 = block(e1, t)
        
        e2 = self.down1(e1)
        for block in self.enc2:
            e2 = block(e2, t)
        
        e3 = self.down2(e2)
        for block in self.enc3:
            e3 = block(e3, t)
        
        # Bottleneck
        b = self.down3(e3)
        for block in self.bottleneck:
            b = block(b, t)
        
        # Decoder
        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        for block in self.dec3:
            d3 = block(d3, t)
        
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        for block in self.dec2:
            d2 = block(d2, t)
        
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        for block in self.dec1:
            d1 = block(d1, t)
        
        return self.final(d1)

# DDPM 클래스
class DDPM:
    def __init__(self, model, timesteps=1000, beta_start=0.0001, beta_end=0.02):
        self.model = model
        self.timesteps = timesteps
        
        # 베타 스케줄
        self.betas = torch.linspace(beta_start, beta_end, timesteps).to(device)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]).to(device), self.alphas_cumprod[:-1]])
        
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)
        
        self.posterior_variance = self.betas * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)
    
    def q_sample(self, x0, t, noise=None):
        """Forward diffusion process"""
        if noise is None:
            noise = torch.randn_like(x0)
        
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        
        return sqrt_alpha_prod * x0 + sqrt_one_minus_alpha_prod * noise
    
    @torch.no_grad()
    def p_sample(self, x, t):
        """Reverse diffusion single step"""
        betas_t = self.betas[t][:, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        sqrt_recip_alphas_t = torch.sqrt(1.0 / self.alphas[t])[:, None, None, None]
        
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * self.model(x, t) / sqrt_one_minus_alphas_cumprod_t
        )
        
        if t[0] == 0:
            return model_mean
        else:
            noise = torch.randn_like(x)
            posterior_variance_t = self.posterior_variance[t][:, None, None, None]
            return model_mean + torch.sqrt(posterior_variance_t) * noise
    
    @torch.no_grad()
    def sample(self, batch_size=16, img_size=64, channels=3):
        """생성 샘플링"""
        self.model.eval()
        x = torch.randn(batch_size, channels, img_size, img_size).to(device)
        
        for t in tqdm(reversed(range(self.timesteps)), desc='Sampling'):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            x = self.p_sample(x, t_batch)
        
        return x

# 학습 함수
def train_ddpm(ddpm, dataloader, epochs=100, lr=2e-4):
    optimizer = optim.Adam(ddpm.model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        ddpm.model.train()
        total_loss = 0
        
        for batch in tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}'):
            batch = batch.to(device)
            batch_size = batch.shape[0]
            
            # 랜덤 타임스텝
            t = torch.randint(0, ddpm.timesteps, (batch_size,), device=device).long()
            
            # 노이즈 추가
            noise = torch.randn_like(batch)
            x_noisy = ddpm.q_sample(batch, t, noise)
            
            # 노이즈 예측
            noise_pred = ddpm.model(x_noisy, t)
            
            # 손실 계산
            loss = criterion(noise_pred, noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}')
        
        # 모델 저장
        if (epoch + 1) % 10 == 0:
            torch.save(ddpm.model.state_dict(), f'ddpm_utkface_epoch_{epoch+1}.pth')
            
            # 샘플 이미지 생성
            samples = ddpm.sample(batch_size=16, img_size=64, channels=3)
            samples = (samples + 1) / 2  # [-1, 1] -> [0, 1]
            samples = samples.clamp(0, 1)
            
            # 샘플 저장
            from torchvision.utils import save_image
            save_image(samples, f'samples_epoch_{epoch+1}.png', nrow=4)
            print(f'Saved samples to samples_epoch_{epoch+1}.png')

# 메인 실행
if __name__ == "__main__":
    # 데이터 경로
    DATA_DIR = "./UTKFace"
    IMG_SIZE = 64
    BATCH_SIZE = 32
    EPOCHS = 100
    
    # 데이터 변환
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # [-1, 1]
    ])
    
    # 데이터셋 및 데이터로더
    print("Loading dataset...")
    dataset = UTKFaceDataset(DATA_DIR, transform=transform)
    print(f"Total images: {len(dataset)}")
    
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    
    # 모델 및 DDPM 생성
    model = UNet(img_channels=3, time_dim=256).to(device)
    ddpm = DDPM(model, timesteps=1000)
    
    # 학습
    print("\nStarting training...")
    start_time = time.time()
    train_ddpm(ddpm, dataloader, epochs=EPOCHS)
    end_time = time.time()
    
    print(f"\nTraining completed in {(end_time - start_time) / 3600:.2f} hours")
    
    # 최종 샘플 생성
    print("\nGenerating final samples...")
    model.load_state_dict(torch.load(f'ddpm_utkface_epoch_{EPOCHS}.pth'))
    samples = ddpm.sample(batch_size=64, img_size=IMG_SIZE, channels=3)
    samples = (samples + 1) / 2
    samples = samples.clamp(0, 1)
    
    from torchvision.utils import save_image
    save_image(samples, 'final_samples.png', nrow=8)
    print("Final samples saved to final_samples.png")