import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# Generator 네트워크
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # 입력: latent_dim x 1 x 1
            nn.ConvTranspose2d(100, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # 512 x 4 x 4
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # 256 x 8 x 8
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # 128 x 16 x 16
            nn.ConvTranspose2d(128, 3, 4, 2, 1, bias=False),
            nn.Tanh()
            # 3 x 32 x 32
        )

    def forward(self, x):
        return self.main(x)

# Discriminator 네트워크
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # 입력: 3 x 32 x 32
            nn.Conv2d(3, 128, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 128 x 16 x 16
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # 256 x 8 x 8
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # 512 x 4 x 4
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x).view(-1, 1).squeeze(1)

# 가중치 초기화 함수
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# 학습 함수
def train_gan():
    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 하이퍼파라미터
    batch_size = 128
    lr = 0.0002
    beta1 = 0.5
    latent_dim = 100
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
    netG = Generator().to(device)
    netD = Discriminator().to(device)

    # 가중치 초기화
    netG.apply(weights_init)
    netD.apply(weights_init)

    # 손실 함수와 옵티마이저
    criterion = nn.BCELoss()
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    # 고정된 노이즈 (생성 결과 시각화용)
    fixed_noise = torch.randn(64, latent_dim, 1, 1, device=device)

    # 실제 이미지 샘플 저장
    real_batch = next(iter(dataloader))
    real_images = real_batch[0][:64].to(device)

    # 학습
    print("Starting Training...")
    G_losses = []
    D_losses = []

    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader):
            # (1) Discriminator 업데이트
            netD.zero_grad()
            real_images_batch = data[0].to(device)
            b_size = real_images_batch.size(0)
            label = torch.full((b_size,), 1., dtype=torch.float, device=device)
            
            # 실제 이미지로 학습
            output = netD(real_images_batch)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()
            
            # 가짜 이미지로 학습
            noise = torch.randn(b_size, latent_dim, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(0.)
            output = netD(fake.detach())
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()
            
            # (2) Generator 업데이트
            netG.zero_grad()
            label.fill_(1.)
            output = netD(fake)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()
            
            # 통계 출력
            if i % 100 == 0:
                print(f'[{epoch}/{num_epochs}][{i}/{len(dataloader)}] '
                      f'Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f} '
                      f'D(x): {D_x:.4f} D(G(z)): {D_G_z1:.4f}/{D_G_z2:.4f}')
            
            G_losses.append(errG.item())
            D_losses.append(errD.item())

    print("Training finished!")
    return netG, real_images, fixed_noise, G_losses, D_losses, device

# 결과 시각화 함수
def visualize_results(netG, real_images, fixed_noise, G_losses, D_losses, device):
    def denorm(x):
        out = (x + 1) / 2
        return out.clamp(0, 1)

    # 원본 이미지 시각화
    plt.figure(figsize=(15, 8))
    plt.subplot(2, 2, 1)
    plt.axis("off")
    plt.title("Real Images (CIFAR-10)")
    plt.imshow(np.transpose(torchvision.utils.make_grid(
        denorm(real_images.cpu())[:64], padding=2, normalize=False).cpu(), (1, 2, 0)))

    # GAN 생성 이미지
    with torch.no_grad():
        fake = netG(fixed_noise).detach().cpu()
    plt.subplot(2, 2, 2)
    plt.axis("off")
    plt.title("Generated Images (GAN Output)")
    plt.imshow(np.transpose(torchvision.utils.make_grid(
        denorm(fake)[:64], padding=2, normalize=False), (1, 2, 0)))

    # 손실 그래프
    plt.subplot(2, 2, 3)
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G", alpha=0.7)
    plt.plot(D_losses, label="D", alpha=0.7)
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()

    # 비교 (개별 이미지)
    plt.subplot(2, 2, 4)
    comparison = torch.cat([
        denorm(real_images.cpu())[:8],
        denorm(fake)[:8]
    ])
    plt.axis("off")
    plt.title("Comparison (Top: Real, Bottom: Generated)")
    plt.imshow(np.transpose(torchvision.utils.make_grid(
        comparison, padding=2, nrow=8, normalize=False), (1, 2, 0)))

    plt.tight_layout()
    plt.savefig('gan_cifar10_results.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f"\n결과가 'gan_cifar10_results.png'로 저장되었습니다.")
    print(f"CUDA 사용 여부: {torch.cuda.is_available()}")


if __name__ == '__main__':
    netG, real_images, fixed_noise, G_losses, D_losses, device = train_gan()
    visualize_results(netG, real_images, fixed_noise, G_losses, D_losses, device)