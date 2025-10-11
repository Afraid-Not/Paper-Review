import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# VAE 네트워크
class VAE(nn.Module):
    def __init__(self, latent_dim=128):
        super(VAE, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            # 3 x 32 x 32
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # 32 x 16 x 16
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # 64 x 8 x 8
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # 128 x 4 x 4
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # 256 x 2 x 2
        )
        
        self.fc_mu = nn.Linear(256 * 2 * 2, latent_dim)
        self.fc_logvar = nn.Linear(256 * 2 * 2, latent_dim)
        
        # Decoder
        self.fc_decode = nn.Linear(latent_dim, 256 * 2 * 2)
        
        self.decoder = nn.Sequential(
            # 256 x 2 x 2
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # 128 x 4 x 4
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # 64 x 8 x 8
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # 32 x 16 x 16
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Tanh()
            # 3 x 32 x 32
        )
    
    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h = self.fc_decode(z)
        h = h.view(h.size(0), 256, 2, 2)
        return self.decoder(h)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

# VAE 손실 함수
def vae_loss(recon_x, x, mu, logvar):
    # Reconstruction loss (Binary Cross Entropy)
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    
    # KL Divergence loss
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + kld, recon_loss, kld

# 학습 함수
def train_vae():
    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 하이퍼파라미터
    batch_size = 128
    lr = 0.001
    latent_dim = 128
    num_epochs = 50

    # CIFAR-10 데이터셋 로드
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0  # Windows에서는 0으로 설정
    )

    # 모델 초기화
    model = VAE(latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 원본 이미지 샘플 저장
    real_batch = next(iter(dataloader))
    real_images = real_batch[0][:64].to(device)

    # 학습
    print("Starting Training...")
    train_losses = []
    recon_losses = []
    kld_losses = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        epoch_recon = 0
        epoch_kld = 0
        
        for i, data in enumerate(dataloader):
            images = data[0].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            recon_images, mu, logvar = model(images)
            loss, recon_loss, kld = vae_loss(recon_images, images, mu, logvar)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_recon += recon_loss.item()
            epoch_kld += kld.item()
            
            # 통계 출력
            if i % 100 == 0:
                print(f'[{epoch}/{num_epochs}][{i}/{len(dataloader)}] '
                      f'Loss: {loss.item()/len(images):.4f} '
                      f'Recon: {recon_loss.item()/len(images):.4f} '
                      f'KLD: {kld.item()/len(images):.4f}')
        
        avg_loss = epoch_loss / len(train_dataset)
        avg_recon = epoch_recon / len(train_dataset)
        avg_kld = epoch_kld / len(train_dataset)
        
        train_losses.append(avg_loss)
        recon_losses.append(avg_recon)
        kld_losses.append(avg_kld)
        
        print(f'Epoch [{epoch}/{num_epochs}] Average Loss: {avg_loss:.4f}')

    print("Training finished!")
    return model, real_images, train_losses, recon_losses, kld_losses, device, latent_dim

# 결과 시각화 함수
def visualize_results(model, real_images, train_losses, recon_losses, kld_losses, device, latent_dim):
    def denorm(x):
        out = (x + 1) / 2
        return out.clamp(0, 1)

    model.eval()
    
    # 원본 이미지 재구성
    with torch.no_grad():
        recon_images, _, _ = model(real_images)
    
    # 랜덤 샘플링으로 새로운 이미지 생성
    with torch.no_grad():
        z = torch.randn(64, latent_dim).to(device)
        generated_images = model.decode(z)

    # 시각화
    plt.figure(figsize=(15, 10))
    
    # 1. 원본 이미지
    plt.subplot(2, 3, 1)
    plt.axis("off")
    plt.title("Original Images (CIFAR-10)")
    plt.imshow(np.transpose(torchvision.utils.make_grid(
        denorm(real_images.cpu())[:64], padding=2, normalize=False).cpu(), (1, 2, 0)))

    # 2. 재구성된 이미지
    plt.subplot(2, 3, 2)
    plt.axis("off")
    plt.title("Reconstructed Images (VAE)")
    plt.imshow(np.transpose(torchvision.utils.make_grid(
        denorm(recon_images.cpu())[:64], padding=2, normalize=False), (1, 2, 0)))

    # 3. 생성된 이미지 (랜덤 샘플링)
    plt.subplot(2, 3, 3)
    plt.axis("off")
    plt.title("Generated Images (Random Sampling)")
    plt.imshow(np.transpose(torchvision.utils.make_grid(
        denorm(generated_images.cpu())[:64], padding=2, normalize=False), (1, 2, 0)))

    # 4. 손실 그래프
    plt.subplot(2, 3, 4)
    plt.title("Training Loss")
    plt.plot(train_losses, label="Total Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # 5. Reconstruction Loss vs KLD
    plt.subplot(2, 3, 5)
    plt.title("Reconstruction Loss vs KLD")
    plt.plot(recon_losses, label="Recon Loss", alpha=0.7)
    plt.plot(kld_losses, label="KLD", alpha=0.7)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # 6. 원본 vs 재구성 비교
    plt.subplot(2, 3, 6)
    comparison = torch.cat([
        denorm(real_images.cpu())[:8],
        denorm(recon_images.cpu())[:8]
    ])
    plt.axis("off")
    plt.title("Comparison (Top: Original, Bottom: Reconstructed)")
    plt.imshow(np.transpose(torchvision.utils.make_grid(
        comparison, padding=2, nrow=8, normalize=False), (1, 2, 0)))

    plt.tight_layout()
    plt.savefig('vae_cifar10_results.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f"\n결과가 'vae_cifar10_results.png'로 저장되었습니다.")
    print(f"CUDA 사용 여부: {torch.cuda.is_available()}")


if __name__ == '__main__':
    model, real_images, train_losses, recon_losses, kld_losses, device, latent_dim = train_vae()
    visualize_results(model, real_images, train_losses, recon_losses, kld_losses, device, latent_dim)