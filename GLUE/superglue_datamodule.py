
import datasets
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import (
    AdamW,
    AutoModelForSequenceClassification,
    AutoConfig,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    glue_compute_metrics,
)


class SuperGLUEDataModule(pl.LightningDataModule):

    task_text_field_map = {
        'boolq': ['question', 'passage'],
        'cb': ['premise', 'hypothesis'],
        'copa': ['premise', 'choice1', 'choice2', 'question'],
        'multirc': ['paragraph', 'question', 'answer'],
        'axb': ['sentence1', 'sentence2'],
        'axg': ['premise', 'hypothesis'],
        'rte': ['premise', 'hypothesis'],
    }

    glue_task_num_labels = {
        'boolq': 2,
        'cb': 3,
        'copa': 2,
        'multirc': 2,
        'axb': 2,
        'axg': 2,
        'rte': 2,
    }

    loader_columns = [
        'datasets_idx',
        'input_ids',
        'token_type_ids',
        'attention_mask',
        'start_positions',
        'end_positions',
        'labels'
    ]

    def __init__(
        self,
        model_name_or_path: str,
        task_name: str = 'copa',
        max_seq_length: int = 128,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        **kwargs
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.task_name = task_name
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size

        self.text_fields = self.task_text_field_map[task_name]
        self.num_labels = self.glue_task_num_labels[task_name]
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path, use_fast=True)

    def setup(self, stage):
        self.dataset = datasets.load_dataset('super_glue', self.task_name)

        for split in self.dataset.keys():
            self.dataset[split] = self.dataset[split].map(
                self.convert_to_features,
                batched=True,
                remove_columns=['label'],
            )
            self.columns = [
                c for c in self.dataset[split].column_names if c in self.loader_columns]
            self.dataset[split].set_format(type="torch", columns=self.columns)

        self.eval_splits = [
            x for x in self.dataset.keys() if 'validation' in x]

    def prepare_data(self):
        datasets.load_dataset('super_glue', self.task_name)
        AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)

    def train_dataloader(self):
        return DataLoader(self.dataset['train'], batch_size=self.train_batch_size)

    def val_dataloader(self):
        if len(self.eval_splits) == 1:
            return DataLoader(self.dataset['validation'], batch_size=self.eval_batch_size)
        elif len(self.eval_splits) > 1:
            return [DataLoader(self.dataset[x], batch_size=self.eval_batch_size) for x in self.eval_splits]

    def test_dataloader(self):
        if len(self.eval_splits) == 1:
            return DataLoader(self.dataset['test'], batch_size=self.eval_batch_size)
        elif len(self.eval_splits) > 1:
            return [DataLoader(self.dataset[x], batch_size=self.eval_batch_size) for x in self.eval_splits]

    def convert_to_features(self, example_batch, indices=None):

        # Either encode single sentence or sentence pairs
        if len(self.text_fields) > 1:
            texts_or_text_pairs = list(
                zip(example_batch[self.text_fields[0]], example_batch[self.text_fields[1]]))
        else:
            texts_or_text_pairs = example_batch[self.text_fields[0]]

        # Tokenize the text/text pairs
        features = self.tokenizer.batch_encode_plus(
            texts_or_text_pairs,
            max_length=self.max_seq_length,
            pad_to_max_length=True,
            truncation=True
        )

        # Rename label to labels to make it easier to pass to model forward
        features['labels'] = example_batch['label']

        return features


if __name__ == "__main__":
    dm = SuperGLUEDataModule('distilbert-base-uncased')
    dm.prepare_data()
    dm.setup('fit')
    next(iter(dm.train_dataloader()))
