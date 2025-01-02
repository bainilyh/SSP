import torch
from torch.utils.data import Dataset, DataLoader


class TextSequenceDataset(Dataset):
    def __init__(self, file_path):
        self.file_path = file_path
        self.sequences = self._load_data()

    def _load_data(self):
        with open(self.file_path, 'r', encoding='utf-8') as f:
            sequences = [line.strip() for line in f]
        return sequences

    def _process_sequence(self, sequence):
        tokens = sequence.split(' ')
        return torch.tensor(list(map(int, tokens)), dtype=torch.long)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        return self._process_sequence(sequence)

if __name__ == '__main__':
    # 创建数据集
    dataset = TextSequenceDataset(
        file_path='C:\\Users\\huang\\Downloads\\SSP\\sequences.txt'
    )

    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=32,      # 批量大小
        shuffle=True,       # 是否打乱数据
        num_workers=1       # 使用多少个子进程加载数据
    )

    # 使用示例
    for batch in dataloader:
        # batch 的形状是 [batch_size, seq_size]
        print(batch.shape)  # 输出: torch.Size([32, 50])
        break
        # 这里可以将 batch 直接输入模型
        # model(batch)
        