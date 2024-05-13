"""

I think there will be an Error because
of Dimensions

"""






import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# Load your small dataset
# from Models.data_helper_for_models import


data = pd.read_csv(r'Dataset\smalldata89modeified.csv')
print(data.head())

# Take copy of the data

Data_copied = data



# Preprocess your data (normalize, handle missing values, etc.)
# Assuming you've already done this step


# Convert pandas DataFrame to PyTorch tensors
target = torch.tensor(Data_copied['health_status'].values, dtype=torch.float32)
features = torch.tensor(Data_copied.drop(columns=['health_status']).values, dtype=torch.float32)



# Decode the dataset first
def decode() :
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()

    # overriding
    Data_copied['health_status'] = label_encoder.fit_transform(Data_copied['health_status'])




if True :
    decode()
    print('Dataset Decoded Successfully')



print(Data_copied.head())



# Define the generator architecture
class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
            nn.Tanh()
        )


    def forward(self, x):
        return self.model(x)



# Define the discriminator architecture
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)



# Define GAN
class GAN(nn.Module):
    def __init__(self, generator, discriminator):
        super(GAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator

    def forward(self, x):
        return self.generator(x)




# Define GAN hyperparameters
latent_dim = 100
input_dim = features.shape[1]
output_dim = input_dim




# Create instances of generator, discriminator, and GAN
generator = Generator(latent_dim, output_dim)
discriminator = Discriminator(input_dim)
gan = GAN(generator, discriminator)



# Define loss function and optimizer
criterion = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(),     lr=0.002)
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.002)



# Training loop
# u can used 2 dictionaries and nested loop on them to try
# some combination


num_epochs = 100
batch_size = 32

for epoch in range(num_epochs):
    for i in range(0, features.size(0), batch_size):
        # Train discriminator
        discriminator.zero_grad()
        real_data = features[i:i+batch_size]
        real_target = target[i:i+batch_size]
        real_output = discriminator(real_data)
        real_loss = criterion(real_output, real_target.view(-1, 1))

        noise = torch.randn(batch_size, latent_dim)
        fake_data = generator(noise)
        fake_target = torch.zeros(batch_size, 1)
        fake_output = discriminator(fake_data.detach())
        fake_loss = criterion(fake_output, fake_target)

        d_loss = real_loss + fake_loss
        d_loss.backward()
        optimizer_D.step()

        # Train generator
        generator.zero_grad()
        noise = torch.randn(batch_size, latent_dim)
        generated_data = generator(noise)
        generated_output = discriminator(generated_data)
        g_target = torch.ones_like(generated_output)  # Target label for generator is 1 (real)
        g_loss = criterion(generated_output, g_target)

        g_loss.backward()
        optimizer_G.step()



    # Print progress
    if epoch % 10 == 0:
        print(f"Epoch [{epoch}/{num_epochs}], Discriminator Loss: {d_loss.item()}, Generator Loss: {g_loss.item()}")








def Generate_new_synthetic_data() :
    # Calculate the number of synthetic examples needed for each class
    num_samples_0 = 357  # Number of examples for class 0 in the original dataset
    num_samples_1 = 212  # Number of examples for class 1 in the original dataset
    total_samples = 500  # Total number of synthetic examples needed

    frac_samples_0 = num_samples_0 / (num_samples_0 + num_samples_1)
    frac_samples_1 = num_samples_1 / (num_samples_0 + num_samples_1)

    # Generate synthetic data for each class
    num_samples_0_synthetic = int(total_samples * frac_samples_0)
    num_samples_1_synthetic = total_samples - num_samples_0_synthetic

    # Generate synthetic data for class 0
    noise_0 = torch.randn(num_samples_0_synthetic, latent_dim)
    synthetic_data_0 = generator(noise_0).detach().numpy()
    synthetic_target_0 = np.zeros((num_samples_0_synthetic, 1))

    # Generate synthetic data for class 1
    noise_1 = torch.randn(num_samples_1_synthetic, latent_dim)
    synthetic_data_1 = generator(noise_1).detach().numpy()
    synthetic_target_1 = np.ones((num_samples_1_synthetic, 1))

    # Combine synthetic data and target labels
    synthetic_data = np.concatenate([synthetic_data_0, synthetic_data_1])
    synthetic_target = np.concatenate([synthetic_target_0, synthetic_target_1])

    # Shuffle synthetic data
    synthetic_indices = np.arange(total_samples)
    np.random.shuffle(synthetic_indices)
    synthetic_data = synthetic_data[synthetic_indices]
    synthetic_target = synthetic_target[synthetic_indices]

    # Combine synthetic data with target labels
    synthetic_data_with_labels = np.concatenate([synthetic_target, synthetic_data], axis=1)

    # Convert to DataFrame
    synthetic_df = pd.DataFrame(synthetic_data_with_labels,
                                  columns=['health_status'] + Data_copied.columns[1:].tolist())

    # # Save synthetic data
    # synthetic_df.to_csv("synthetic_data_GANs_version.csv", index=False)


    return synthetic_df
