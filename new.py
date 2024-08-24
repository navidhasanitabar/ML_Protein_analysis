import torch
import torch.nn as nn
import torch.optim as optim
from kludo import KludoModel
from GraphSite_predict import GraphSiteModel

# مرحله 1: اجرای Kludo و گرفتن خروجی خوشه‌بندی گره‌ها
kludo_model = KludoModel()
protein_graph_data = load_protein_data("path_to_protein_data")
kludo_output = kludo_model.run(protein_graph_data)

# مرحله 2: اجرای GraphSite و گرفتن پیش‌بینی سایت‌های اتصال
graphsite_model = GraphSiteModel()
structure_data = load_structure_data("path_to_structure_data")
graphsite_output = graphsite_model.predict(structure_data)

# مرحله 3: آماده‌سازی ورودی برای GAN
combined_input = torch.cat((kludo_output, graphsite_output), dim=1)

# مرحله 4: تعریف شبکه GAN
class Generator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# پارامترهای شبکه
input_dim = combined_input.shape[1]
hidden_dim = 128
output_dim = input_dim

generator = Generator(input_dim, hidden_dim, output_dim)
discriminator = Discriminator(input_dim, hidden_dim)

# مرحله 5: آموزش GAN
criterion = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002)
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002)

num_epochs = 1000

for epoch in range(num_epochs):
    # آموزش Discriminator
    real_labels = torch.ones(batch_size, 1)
    fake_labels = torch.zeros(batch_size, 1)

    outputs = discriminator(combined_input)
    d_loss_real = criterion(outputs, real_labels)
    real_score = outputs

    z = torch.randn(batch_size, input_dim)
    fake_inputs = generator(z)
    outputs = discriminator(fake_inputs)
    d_loss_fake = criterion(outputs, fake_labels)
    fake_score = outputs

    d_loss = d_loss_real + d_loss_fake
    optimizer_D.zero_grad()
    d_loss.backward()
    optimizer_D.step()

    # آموزش Generator
    z = torch.randn(batch_size, input_dim)
    fake_inputs = generator(z)
    outputs = discriminator(fake_inputs)
    g_loss = criterion(outputs, real_labels)

    optimizer_G.zero_grad()
    g_loss.backward()
    optimizer_G.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], d_loss: {d_loss.item()}, g_loss: {g_loss.item()}')

print("Training complete.")



