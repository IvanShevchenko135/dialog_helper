from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.nn import functional as F
from torch.utils.data import DataLoader

from dataset import CommonVoiceDataset, collate_fn_padd
from model import SpeechRecognition

HYPER_PARAMETERS = {
    "num_classes": 29,
    "n_feats": 81,
    "dropout": 0.1,
    "hidden_size": 1024,
    "num_layers": 1
}


class ASRNuralNetwork(LightningModule):

    def __init__(self, model, args):
        super(ASRNuralNetwork, self).__init__()
        self.model = model
        self.criterion = nn.CTCLoss(blank=28, zero_infinity=True)
        self.args = args

    def forward(self, x, hidden):
        return self.model(x, hidden)

    def configure_optimizers(self):
        self.optimizer = optim.AdamW(self.model.parameters(), self.args.learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min',
            factor=0.50, patience=6)
        return {"optimizer": self.optimizer, "lr_scheduler": self.scheduler, "monitor": "loss"}

    def step(self, batch):
        spectrograms, labels, input_lengths, label_lengths = batch
        bs = spectrograms.shape[0]
        hidden = self.model._init_hidden_state(bs)
        hn, c0 = hidden[0].to(self.device), hidden[1].to(self.device)
        output, _ = self(spectrograms, (hn, c0))
        output = F.log_softmax(output, dim=2)
        loss = self.criterion(output, labels, input_lengths, label_lengths)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch)

        self.log('loss', loss)
        self.log('lr', self.optimizer.param_groups[0]['lr'])

        return loss

    def train_dataloader(self):
        train_dataset = CommonVoiceDataset(json_path='./dataset/train.json', sample_rate=8000)
        return DataLoader(dataset=train_dataset,
                          batch_size=self.args.batch_size,
                          num_workers=12,
                          pin_memory=True,
                          collate_fn=collate_fn_padd)

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch)
        return {'valid_loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['valid_loss'] for x in outputs]).mean()
        self.scheduler.step(avg_loss)
        self.log('valid_loss', avg_loss)

    def val_dataloader(self):
        test_dataset = CommonVoiceDataset(json_path='./dataset/train.json', sample_rate=8000, valid=True)
        return DataLoader(dataset=test_dataset,
                          batch_size=self.args.batch_size,
                          num_workers=12,
                          collate_fn=collate_fn_padd,
                          pin_memory=True)


def main(args):
    model_path = 'best_models/asr_model_epoch=40.ckpt'
    model = SpeechRecognition(**HYPER_PARAMETERS)

    if model_path:
        speech_module = ASRNuralNetwork.load_from_checkpoint(model_path, model=model, args=args)
    else:
        speech_module = ASRNuralNetwork(model, args)

    logger = TensorBoardLogger('model_logs', name='automatic_speech_recognition')

    checkpoint_callback = ModelCheckpoint(
        dirpath='./best_models',
        filename='asr_model_{epoch}',
        save_top_k=3,
        monitor='valid_loss',
        mode='min',
        verbose=True,
    )

    trainer = Trainer(
        max_epochs=3,
        gpus=0,
        callbacks=[checkpoint_callback],
        logger=logger,
        gradient_clip_val=1.0,
        val_check_interval=1.0,
        fast_dev_run=False,
    )

    trainer.fit(speech_module)


if __name__ == "__main__":
    parser = ArgumentParser()

    # general
    parser.add_argument('--epochs', default=10, type=int, help='number of total epochs to run')
    parser.add_argument('--batch_size', default=5, type=int, help='size of batch')
    parser.add_argument('--learning_rate', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--pct_start', default=0.3, type=float, help='percentage of growth phase in one cycle')
    parser.add_argument('--div_factor', default=100, type=int, help='div factor for one cycle')

    args = parser.parse_args()

    main(args)
