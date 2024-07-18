# from google.colab import drive
# drive.mount('/content/drive')

# pip install transformers

# !unzip "/content/drive/MyDrive/Colab Notebooks/Students_DL基礎2024（公開）/4_最終課題/VQA2/train.zip"
# !unzip "/content/drive/MyDrive/Colab Notebooks/Students_DL基礎2024（公開）/4_最終課題/VQA2/valid.zip"

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision.transforms as transforms
from transformers import BertTokenizer, BertModel
import pandas as pd
from PIL import Image
import random
import numpy as np
import re
import time
from statistics import mode

# シードの設定


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# テキストの前処理


def process_text(text):
    text = text.lower()

    num_word_to_digit = {
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
        'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
        'ten': '10'
    }
    for word, digit in num_word_to_digit.items():
        text = text.replace(word, digit)

    text = re.sub(r'(?<!\d)\.(?!\d)', '', text)
    text = re.sub(r'\b(a|an|the)\b', '', text)

    contractions = {
        "dont": "don't", "isnt": "isn't", "arent": "aren't", "wont": "won't",
        "cant": "can't", "wouldnt": "wouldn't", "couldnt": "couldn't"
    }
    for contraction, correct in contractions.items():
        text = text.replace(contraction, correct)

    text = re.sub(r"[^\w\s':]", ' ', text)
    text = re.sub(r'\s+,', ',', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# データセットクラス


class VQADataset(torch.utils.data.Dataset):
    def __init__(self, df_path, image_dir, tokenizer, transform=None, answer=True):
        self.transform = transform
        self.image_dir = image_dir
        self.df = pd.read_json(df_path)
        self.answer = answer
        self.tokenizer = tokenizer  # 変更点

        self.question2idx = {}
        self.answer2idx = {}
        self.idx2question = {}
        self.idx2answer = {}

        for question in self.df["question"]:
            question = process_text(question)
            words = question.split(" ")
            for word in words:
                if word not in self.question2idx:
                    self.question2idx[word] = len(self.question2idx)
        self.idx2question = {v: k for k, v in self.question2idx.items()}

        if self.answer:
            for answers in self.df["answers"]:
                for answer in answers:
                    word = process_text(answer["answer"])
                    if word not in self.answer2idx:
                        self.answer2idx[word] = len(self.answer2idx)
            self.idx2answer = {v: k for k, v in self.answer2idx.items()}

    def update_dict(self, dataset):
        self.question2idx = dataset.question2idx
        self.answer2idx = dataset.answer2idx
        self.idx2question = dataset.idx2question
        self.idx2answer = dataset.idx2answer

    def __getitem__(self, idx):
        image = Image.open(f"{self.image_dir}/{self.df['image'][idx]}")
        image = self.transform(image)

        question = self.df["question"][idx]
        question_inputs = self.tokenizer(
            question, return_tensors="pt", padding="max_length", truncation=True, max_length=20)
        question_inputs = {k: v.squeeze(
            0) for k, v in question_inputs.items()}  # バッチ次元を削除

        if self.answer:
            answers = [self.answer2idx[process_text(
                answer["answer"])] for answer in self.df["answers"][idx]]
            mode_answer_idx = mode(answers)
            return image, question_inputs, torch.Tensor(answers), int(mode_answer_idx)
        else:
            return image, question_inputs

    def __len__(self):
        return len(self.df)

# 評価指標


def VQA_criterion(batch_pred, batch_answers):
    total_acc = 0.
    for pred, answers in zip(batch_pred, batch_answers):
        acc = 0.
        for i in range(len(answers)):
            num_match = 0
            for j in range(len(answers)):
                if i == j:
                    continue
                if pred == answers[j]:
                    num_match += 1
            acc += min(num_match / 3, 1)
        total_acc += acc / 10
    return total_acc / len(batch_pred)

# ResNetを利用できるようにしておく


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out += self.shortcut(residual)
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers):
        super().__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, layers[0], 64)
        self.layer2 = self._make_layer(block, layers[1], 128, stride=2)
        self.layer3 = self._make_layer(block, layers[2], 256, stride=2)
        self.layer4 = self._make_layer(block, layers[3], 512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, 512)

    def _make_layer(self, block, blocks, out_channels, stride=1):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

# モデル


class VQAModel(nn.Module):
    def __init__(self, n_answer):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')  # 変更点
        self.resnet = ResNet18()
        self.fc = nn.Sequential(
            nn.Linear(768 + 512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, n_answer)
        )  # 変更点

    def forward(self, image, question_inputs):
        image_feature = self.resnet(image)
        question_feature = self.bert(**question_inputs).pooler_output
        x = torch.cat([image_feature, question_feature], dim=1)
        x = self.fc(x)
        return x

# トレーニングと評価


def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    total_acc = 0
    simple_acc = 0
    start = time.time()
    for image, question_inputs, answers, mode_answer in dataloader:
        image = image.to(device)
        question_inputs = {k: v.to(device) for k, v in question_inputs.items()}
        answers = answers.to(device)
        mode_answer = mode_answer.to(device)

        pred = model(image, question_inputs)
        loss = criterion(pred, mode_answer)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += VQA_criterion(pred.argmax(1), answers)
        simple_acc += (pred.argmax(1) == mode_answer).float().mean().item()
    return total_loss / len(dataloader), total_acc / len(dataloader), simple_acc / len(dataloader), time.time() - start


def eval(model, dataloader, device):
    model.eval()
    predictions = []
    start = time.time()
    with torch.no_grad():
        for image, question_inputs in dataloader:
            image = image.to(device)
            question_inputs = {k: v.to(device)
                               for k, v in question_inputs.items()}
            pred = model(image, question_inputs)
            predictions.extend(pred.argmax(1).cpu().tolist())
    return predictions, time.time() - start

# メイン関数


def main():
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')  # 変更点

    train_dataset = VQADataset(df_path="/content/drive/MyDrive/Colab Notebooks/Students_DL基礎2024（公開）/4_最終課題/VQA2/train.json",
                               image_dir="/content/train", tokenizer=tokenizer, transform=transform)
    test_dataset = VQADataset(df_path="/content/drive/MyDrive/Colab Notebooks/Students_DL基礎2024（公開）/4_最終課題/VQA2/valid.json",
                              image_dir="/content/valid", tokenizer=tokenizer, transform=transform, answer=False)

    test_dataset.update_dict(train_dataset)

    train_loader = DataLoader(train_dataset, batch_size=16,
                              num_workers=2, pin_memory=True, shuffle=True)  # 変更点
    test_loader = DataLoader(test_dataset, batch_size=16,
                             num_workers=2, pin_memory=True)  # 変更点

    model = VQAModel(n_answer=len(train_dataset.answer2idx))  # 変更点
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(15):
        train_loss, train_acc, simple_train_acc, train_time = train(
            model, train_loader, optimizer, criterion, device)
        predictions, eval_time = eval(model, test_loader, device)

        print(f"Epoch: {epoch + 1}, Train Loss: {train_loss:.3f}, "
              f"Train Acc: {train_acc:.3f}, Simple Train Acc: {simple_train_acc:.3f}, "
              f"Train Time: {train_time:.3f}, Eval Time: {eval_time:.3f}")

    print("Training Complete")

    # 提出用ファイル作成部分
    predictions, _ = eval(model, test_loader, device)
    submission = [train_dataset.idx2answer[id] for id in predictions]
    submission = np.array(submission)
    np.save("submission15.npy", submission)


if __name__ == "__main__":
    main()
