import os

import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader

import torchvision
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from torchvision.utils import save_image

from torch.utils.data import random_split

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from PIL import Image

import time

import glob


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        os.path.join(dirname, filename)

# Print out used versions
print("Used versions:")
print(f"Torch:\t\t{torch.__version__}")
print(f"Torchvision:\t{torchvision.__version__}")
print()


def get_device():
    """
    Check if GPU is available, and if so, picks the GPU, else picks the CPU
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

    
# Set the device to cuda is available
device = get_device()
print(f"Used device is {device}.")


def get_num_correct(preds, labels):
    """
    Returns the number of the correctly predicted images
        Parameters:
            preds (tensor): the predicted labels
            labels (tensor): the true labels (targets)
        Returns:
            num_correct (int): the number of correctly predicted images
    """
    num_correct = preds.argmax(dim=1).eq(labels).sum().item()
    return num_correct


def accuracy(preds, labels):
    """
    Returns the accuracy of the predictions
        Parameters:
            preds (tensor): the predicted labels
            labels (tensor): the true labels (targets)
        Returns:
            acc (float): the accuracy of the predictions (correctly predicted labels / all predictions)
    """
    acc = get_num_correct(preds, labels) / len(labels)
    return acc


def predict(network, image):
    """
    Returns the prediction of an image
        Parameters:
            network (model): the model to use
            image (tensor): the image to predict the label for
        Returns:
            pred (int): the predicted label of the image
    """
    network = network.to(device)
    image = image.to(device)
    output = network(image.unsqueeze(0))
    pred = output.argmax(dim=1).item()
    return pred


train_ds_normal_path = glob.glob("../input/chest-xray-pneumonia/chest_xray/train/NORMAL/*")
train_ds_pneumonia_path = glob.glob("../input/chest-xray-pneumonia/chest_xray/train/PNEUMONIA/*")

test_ds_normal_path = glob.glob("../input/chest-xray-pneumonia/chest_xray/test/NORMAL/*")
test_ds_pneumonia_path = glob.glob("../input/chest-xray-pneumonia/chest_xray/test/PNEUMONIA/*")

val_ds_normal_path = glob.glob("../input/chest-xray-pneumonia/chest_xray/val/NORMAL/*")
val_ds_pneumonia_path = glob.glob("../input/chest-xray-pneumonia/chest_xray/val/PNEUMONIA/*")


print(f"'Normal' images in training set: \t{len(train_ds_normal_path):6,}")
print(f"'Pneumonia' images in training set: \t{len(train_ds_pneumonia_path):6,}")
print()

print(f"'Normal' images in validation set: \t{len(val_ds_normal_path):6,}")
print(f"'Pneumonia' images in validation set: \t{len(val_ds_pneumonia_path):6,}")
print()

print(f"'Normal' images in test set: \t\t{len(test_ds_normal_path):6,}")
print(f"'Pneumonia' images in test set: \t{len(test_ds_pneumonia_path):6,}")
print()


# Since the validation set is basically empty, and after a lot of trying and readings on forums, the test set is likely to be labelled incorrectly, 
# the validation and test sets will be separated from the training set, which will be the only folder in use

# 300-300 of normal and pneumonia images is separated from training set -> test set
# 200-200 of normal and pneumonia images is separated from training set -> test set


normal_test_size = 300
normal_val_size = 200
normal_train_size = len(train_ds_normal_path) - normal_test_size - normal_val_size

train_ds_normal_path, test_ds_normal_path, val_ds_normal_path = random_split(train_ds_normal_path, [normal_train_size, normal_test_size, normal_val_size])

pneumonia_test_size = 300
pneumonia_val_size = 200
pneumonia_train_size = len(train_ds_pneumonia_path) - pneumonia_test_size - pneumonia_val_size

train_ds_pneumonia_path, test_ds_pneumonia_path, val_ds_pneumonia_path = random_split(train_ds_pneumonia_path, 
                                                                                      [pneumonia_train_size, pneumonia_test_size, pneumonia_val_size])

# In the training set there are about 3 times as many pneumonia images as normal images, which could cause nonoptimal learning
# With data augmentation the number of normal images can be increased


print()
print(f"Images in datasets:")
print(f"TRAINING set size: \t{normal_train_size:5,} + {pneumonia_train_size:5,}")
print(f"VALIDATION set size: \t{normal_val_size:5,} + {pneumonia_val_size:5,}")
print(f"TEST set size: \t\t{normal_test_size:5,} + {pneumonia_test_size:5,}")


class CustomImages(torch.utils.data.Dataset):
    """
    Opens images from their paths and returns transformed images
          Parameters:
              paths (list): the list of image paths
              image_transformations (transforms): the transformations we want to apply on the images
          Returns:
              image (image): the current modified image
    """

    def __init__(self, path, image_transformations=None):
        self.path = path
        self.image_transformations = image_transformations

    def __getitem__(self, idx):
        # dataset[idx]
        path = self.path[idx]
        image = Image.open(path).convert("RGB")
        if self.image_transformations:
            image = self.image_transformations(image)
            
        return image

    def __len__(self):
        # len(dataset)
        return len(self.path)


# Transformations for the augmented normal images
augmentation_transformations = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(5),
    transforms.ColorJitter(brightness=.05),
    transforms.RandomPerspective(distortion_scale=0.1, p=0.1),
    transforms.ToTensor()
])


dataset = CustomImages(train_ds_normal_path, augmentation_transformations)


# Creating a new folder for the augmented images
target_dir = r"../working/augmented"
if not os.path.exists(target_dir):
    os.mkdir("../working/augmented")


# Create augmented images (for the range of 2, so the normal images' number would increase)
counter = 0
for i in range(2):
    for image in dataset:
        
        filepath = os.path.join(target_dir, f"normal_augmented_{counter:04n}.jpg")
        
        save_image(image, filepath)
        counter += 1


train_ds_normal_augmented_path = glob.glob("../working/augmented/*")


normal_train_size += len(train_ds_normal_augmented_path)


print()
print(f"Images in datasets after augmenting:")
print(f"TRAINING set size: \t{normal_train_size:5,} + {pneumonia_train_size:5,}")
print(f"VALIDATION set size: \t{normal_val_size:5,} + {pneumonia_val_size:5,}")
print(f"TEST set size: \t\t{normal_test_size:5,} + {pneumonia_test_size:5,}")


# Create the labels for the images (0-normal, 1-pneumonia)
train_ds_normal_labels = torch.zeros(len(train_ds_normal_path) + len(train_ds_normal_augmented_path))
test_ds_normal_labels = torch.zeros(len(test_ds_normal_path))
val_ds_normal_labels = torch.zeros(len(val_ds_normal_path))

train_ds_pneumonia_labels = torch.ones(len(train_ds_pneumonia_path))
test_ds_pneumonia_labels = torch.ones(len(test_ds_pneumonia_path))
val_ds_pneumonia_labels = torch.ones(len(val_ds_pneumonia_path))


train_labels = torch.cat((train_ds_normal_labels, train_ds_pneumonia_labels), dim=0)
test_labels = torch.cat((test_ds_normal_labels, test_ds_pneumonia_labels), dim=0)
val_labels = torch.cat((val_ds_normal_labels, val_ds_pneumonia_labels), dim=0)


train_path = train_ds_normal_path + train_ds_normal_augmented_path + train_ds_pneumonia_path
test_path = test_ds_normal_path + test_ds_pneumonia_path
val_path = val_ds_normal_path + val_ds_pneumonia_path


class MergeDataset(torch.utils.data.Dataset):
    """
    Returns merged dataset with image path - label pairs
          Parameters:
              paths (list): the list of image paths
              labels (list): the list of labels
          Returns:
              image, label (tuple): the current modified image-label tuple
    """

    def __init__(self, paths, labels):
        self.paths = paths
        self.labels = labels

    def __getitem__(self, idx):
        # dataset[idx]
        path = self.paths[idx]
        image = Image.open(path).convert("RGB")
        label = self.labels[idx]

        return image, label

    def __len__(self):
        # len(dataset)
        return len(self.paths)


train_dataset = MergeDataset(train_path, train_labels)
val_dataset = MergeDataset(val_path, val_labels)
test_dataset = MergeDataset(test_path, test_labels)


class PneumoniaDataset(torch.utils.data.Dataset):
    """
    Opens images from their paths and returns transformed dataset
          Parameters:
              paths (list): the list of image paths
              labels (list): the list of labels
              image_transformations (transforms): the transformations we want to apply on the images
              label_transformations (transforms): the transformations we want to apply on the labels
          Returns:
              image, label (tuple): the current modified image-label tuple
    """

    def __init__(self, dataset, image_transformations=None, label_transformations=None):
        self.dataset = dataset
        self.image_transformations = image_transformations
        self.label_transformations = label_transformations

    def __getitem__(self, idx):
        # dataset[idx]
        image, label = self.dataset[idx]
        if self.image_transformations:
            image = self.image_transformations(image)
        if self.label_transformations:
            label = self.label_transformations(label)

        return image, label

    def __len__(self):
        # len(dataset)
        return len(self.dataset)


# Create a dataloader for sample images to show
show_dataset = PneumoniaDataset(train_dataset, image_transformations=transforms.Compose([transforms.Resize((300, 300)), transforms.ToTensor()]))
show_dataloader = DataLoader(show_dataset, batch_size=16, shuffle=True)


classes = {
            0 : "Normal",
            1 : "Pneumonia"
          }

for images, labels in show_dataloader:
    plt.figure(figsize=(30,30))
    plt.axis("off")
    plt.imshow(make_grid(images, nrow=4, padding=0).permute(1,2,0))
    for index, label in enumerate(labels):
        if index < 4:
            plt.text(index*300+6, 18, classes[label.item()], bbox={'facecolor': 'white', 'pad': 10}, fontsize=20)
        elif index < 8:
            plt.text((index-4)*300+6, 318, classes[label.item()], bbox={'facecolor': 'white', 'pad': 10}, fontsize=20)
        elif index < 12:
            plt.text((index-8)*300+6, 618, classes[label.item()], bbox={'facecolor': 'white', 'pad': 10}, fontsize=20)
        else:
            plt.text((index-12)*300+6, 918, classes[label.item()], bbox={'facecolor': 'white', 'pad': 10}, fontsize=20)
    break


# Set image size for transformations
image_size = (256, 256)

# Set training|validation|test transformations
training_transformations = transforms.Compose([
    transforms.Resize(image_size),
    transforms.RandomRotation(degrees=10),
    transforms.RandomPerspective(distortion_scale=0.1, p=0.1),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(size=image_size, padding=4),
    transforms.ToTensor()
])

validation_transformations = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor()
])

test_transformations = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor()
])


# Apply transformations
train_ds = PneumoniaDataset(train_dataset, training_transformations)
train_dataset_size = len(train_ds)

val_ds = PneumoniaDataset(val_dataset, validation_transformations)
val_dataset_size = len(val_ds)

test_ds = PneumoniaDataset(test_dataset, test_transformations)
test_dataset_size = len(test_ds)


"""
MODEL DEFINITION 1
"""

class model1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.batchnorm_conv_1 = nn.BatchNorm2d(32)
        self.maxpool2d_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.20)
        
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.batchnorm_conv_2 = nn.BatchNorm2d(32)
        self.maxpool2d_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.batchnorm_conv_3 = nn.BatchNorm2d(64)
        self.maxpool2d_3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(0.20)
        
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.maxpool2d_4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc1 = nn.Linear(in_features=64 * int(image_size[0]/16) * int(image_size[1]/16), out_features=1024)
        self.dropout3 = nn.Dropout(0.30)
        
        self.fc2 = nn.Linear(in_features=1024, out_features=256)
        
        self.out = nn.Linear(in_features=256, out_features=2)
        
    def forward(self, t):
        t = F.relu(self.conv1(t))
        t = self.dropout1(self.maxpool2d_1(F.relu(self.batchnorm_conv_1(self.conv2(t)))))
        t = self.maxpool2d_2(F.relu(self.batchnorm_conv_2(self.conv3(t))))
        t = self.dropout2(self.maxpool2d_3(F.relu(self.batchnorm_conv_3(self.conv4(t)))))
        t = self.maxpool2d_4(F.relu(self.conv5(t)))
        
        t = torch.flatten(t, start_dim=1)
        #t = t.reshape(-1, 64 * int(image_size[0]/16) * int(image_size[1]/16))
        t = self.dropout3(F.relu(self.fc1(t)))
        t = F.relu(self.fc2(t))
        t = F.relu(self.out(t))
        
        return t


"""
MODEL DEFINITION 2
"""

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=(5, 5), stride=(1, 1),
                               padding=(2, 2))
        self.conv2 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=(5, 5), stride=(1, 1),
                               padding=(2, 2))

    def forward(self, t):
        res_t = t
        t = self.conv1(t)
        t = F.relu(t)
        t = self.conv2(t)
        t += res_t
        t = F.relu(t)

        return t
    
    
model2 = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
    nn.ReLU(),
    nn.BatchNorm2d(16),
    nn.Conv2d(in_channels=16, out_channels=64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
    nn.ReLU(),
    nn.BatchNorm2d(64),
    nn.MaxPool2d(kernel_size=2, stride=2),
    ResidualBlock(channels=64),
    ResidualBlock(channels=64),
    ResidualBlock(channels=64),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
    nn.ReLU(),
    nn.BatchNorm2d(128),
    nn.MaxPool2d(kernel_size=2, stride=2),
    ResidualBlock(channels=128),
    ResidualBlock(channels=128),
    ResidualBlock(channels=128),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
    nn.ReLU(),
    nn.BatchNorm2d(128),
    nn.MaxPool2d(kernel_size=2, stride=2),
    ResidualBlock(channels=128),
    ResidualBlock(channels=128),
    ResidualBlock(channels=128),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
    nn.ReLU(),
    nn.BatchNorm2d(256),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Flatten(start_dim=1),
    nn.Linear(in_features=256 * int(image_size[0]/128) * int(image_size[1]/128), out_features=2048),
    nn.ReLU(),
    nn.BatchNorm1d(2048),
    nn.Linear(in_features=2048, out_features=1024),
    nn.ReLU(),
    nn.BatchNorm1d(1024),
    nn.Linear(in_features=1024, out_features=256),
    nn.ReLU(),
    nn.BatchNorm1d(256),
    nn.Linear(in_features=256, out_features=64),
    nn.ReLU(),
    nn.BatchNorm1d(64),
    nn.Linear(in_features=64, out_features=2)
)


"""
HYPERPARAMETERS
"""

network = model1()

batch_size = 32
loss_fn = F.cross_entropy
learning_rate = 0.0005
num_epoch = 25

network.to(device)

print(network)
print()
pytorch_total_params = sum(p.numel() for p in network.parameters())
print(f"Total parameters: \t\t{pytorch_total_params:,}")
pytorch_total_learnable_params = sum(p.numel() for p in network.parameters() if p.requires_grad)
print(f"Total trainable parameters: \t{pytorch_total_learnable_params:,}")
print()
print(f"Training dataset size: \t\t{train_dataset_size:,}")
print(f"Validation dataset size: \t{val_dataset_size:,}")
print(f"Test dataset size: \t\t{test_dataset_size:,}")
print()
print(f"Batch size    \t {batch_size}")
print(f"Learning rate \t {learning_rate}")
print(f"Loss function \t {loss_fn}")
print(f"No. epochs    \t {num_epoch}")


train_loader = DataLoader(train_ds, batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size * 2)
test_loader = DataLoader(test_ds, batch_size * 2)


optimizer = optim.Adam(network.parameters(), lr=learning_rate)

sched = optim.lr_scheduler.OneCycleLR(optimizer, learning_rate, epochs=num_epoch, steps_per_epoch=len(train_loader))


"""
BEFORE TRAINING METRICS
"""

history_train_loss = []
history_train_acc = []

history_val_loss = []
history_val_acc = []

total_loss = 0
total_correct = 0

total_val_loss = 0
total_val_correct = 0


network.eval()
for batch in train_loader:
    images, labels = batch

    images = images.to(device)
    labels = labels.to(device)
    labels = labels.to(torch.int64)

    preds = network(images)
    loss = loss_fn(preds, labels)

    total_loss += loss.item() * len(batch[0])
    total_correct += get_num_correct(preds, labels)

for batch in val_loader:
    images, labels = batch

    images = images.to(device)
    labels = labels.to(device)
    labels = labels.to(torch.int64)

    preds = network(images)
    loss = loss_fn(preds, labels)

    total_val_loss += loss.item() * len(batch[0])
    total_val_correct += get_num_correct(preds, labels)

print()
print(f"BEFORE TRAINING: train accuracy {100 * total_correct / train_dataset_size:3.2f}%, "
      f"loss = {total_loss / train_dataset_size:4.4f}")
print(f"BEFORE TRAINING: val accuracy   {100 * total_val_correct / val_dataset_size:3.2f}%, "
      f"loss = {total_val_loss / val_dataset_size:4.4f}")
print()
print()

history_train_acc.append(100 * total_correct / train_dataset_size)
history_val_acc.append(100 * total_val_correct / val_dataset_size)

history_train_loss.append(total_loss / train_dataset_size)
history_val_loss.append(total_val_loss / val_dataset_size)



"""
TRAINING
"""

since = time.time()

torch.cuda.empty_cache()

for epoch in range(num_epoch):

    # Training phase

    total_loss = 0
    total_correct = 0

    network.train()

    for batch in train_loader:
        images, labels = batch

        images = images.to(device)
        labels = labels.to(device)
        labels = labels.to(torch.int64)

        preds = network(images)
        loss = loss_fn(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        sched.step()

        total_loss += loss.item() * len(batch[0])
        total_correct += get_num_correct(preds, labels)

    # Validation phase

    total_val_loss = 0
    total_val_correct = 0

    network.eval()

    for batch in val_loader:
        images, labels = batch

        images = images.to(device)
        labels = labels.to(device)
        labels = labels.to(torch.int64)

        with torch.no_grad():
            preds = network(images)
            loss = loss_fn(preds, labels)

        total_val_loss += loss.item() * len(batch[0])
        total_val_correct += get_num_correct(preds, labels)

    # Print out metrics

    training_percentage = 100 * total_correct / train_dataset_size
    validation_percentage = 100 * total_val_correct / val_dataset_size

    print(f"Epoch: {epoch + 1}")

    history_train_acc.append(training_percentage)
    history_val_acc.append(validation_percentage)

    history_train_loss.append(total_loss / train_dataset_size)
    history_val_loss.append(total_val_loss / val_dataset_size)

    print(f"TRAINING     "
          f"train accuracy {training_percentage:3.2f}%, "
          f"train loss: {total_loss / train_dataset_size:2.4f}")
    print(f"VALIDATION   "
          f"val accuracy   {validation_percentage:3.2f}%, "
          f"val loss:   {total_val_loss / val_dataset_size:2.4f}")
    print()
    
print()
print()
passed = time.time()-since

passed_hrs = passed // 3600
passed -= passed_hrs * 3600

passed_mins = passed // 60
passed -= passed_mins * 60

passed_secs = int(passed)

print(f"TIME of training (with validation phases) {passed_hrs}h {passed_mins}m {passed_secs}s.")

"""
END OF TRAINING
"""

# Training and validation accuracy plot
plt.figure(figsize=(12,8))
plt.plot(history_val_acc)
plt.plot(history_train_acc)
plt.ylabel('Accuracy (%)')
plt.xlabel('No. epochs')
plt.title('Accuracy through epochs')
plt.legend(["Validation", "Training"])
plt.show()


"""
TESTING THE MODEL
"""

predictions = []
true_labels = []

test_correct = 0

network.eval()

for batch in test_loader:
    images, labels = batch
    true_labels.extend(labels)

    images = images.to(device)
    labels = labels.to(device)
    labels = labels.to(torch.int64)

    with torch.no_grad():
        preds = network(images)
        predictions.extend((preds.argmax(dim=1)).tolist())
        test_correct += get_num_correct(preds, labels)

print()
print(f"Test set accuracy:\t{100 * test_correct / test_dataset_size}%")
print(test_correct)
print(test_dataset_size)


"""
CONFUSION MATRIX on the test set
"""

cm = confusion_matrix(true_labels, predictions)

plt.figure(figsize=(12, 10))
f = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
f.set_xticklabels(['Normal', 'Pneumonia'], rotation=40, size=16)
f.set_yticklabels(['Normal', 'Pneumonia'], rotation=40, size=16)
f.set_xlabel('True', size=24)
f.set_ylabel('Predicted', size=24)

f.set_title("Confusion Matrix", size=32)
