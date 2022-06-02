import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.nn import functional as F
from torch.utils.data import DataLoader

from dataset import CommonVoiceDataset, collate_fn
from model import SpeechRecognition

# hyper parameters for model
model_hyper_parameters = {
    'num_classes': 29,
    'n_feats': 81,
    'dropout': 0.1,
    'hidden_size': 1024,
    'num_layers': 1,
}

# hyper parameters for training
max_epochs = 50
batch_size = 1
lr = 1e-3
lr_reduce_factor = 0.5
scheduler_patience = 5
pct_start = 0.3
div_factor = 100
sample_rate = 8000


class ASRNuralNetwork(pl.LightningModule):

    def __init__(self, model):
        super(ASRNuralNetwork, self).__init__()

        self.model = model

        self.criterion = nn.CTCLoss(blank=28, zero_infinity=True)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            factor=lr_reduce_factor,
            patience=scheduler_patience,
            mode='min'
        )

    def forward(self, x, hidden):
        return self.model(x, hidden)

    def configure_optimizers(self):
        return {'optimizer': self.optimizer, 'lr_scheduler': {'scheduler': self.scheduler, 'monitor': 'valid_loss'}}

    def step(self, batch):
        spectrograms, target, input_lengths, target_lengths = batch
        current_batch_size = spectrograms.shape[0]

        hidden = self.model.init_hidden(current_batch_size)
        hidden_state = hidden[0].to(self.device)
        cell_state = hidden[1].to(self.device)

        output, _ = self(spectrograms, (hidden_state, cell_state))
        output = F.log_softmax(output, dim=2)

        loss = self.criterion(output, target, input_lengths, target_lengths)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch)

        self.log('loss', loss)
        self.log('lr', self.optimizer.param_groups[0]['lr'])

        return loss

    def train_dataloader(self):
        train_dataset = CommonVoiceDataset(sample_rate=sample_rate, valid=False)
        return DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_fn
        )

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log('valid_loss', loss)

        return loss

    def val_dataloader(self):
        validation_dataset = CommonVoiceDataset(sample_rate=sample_rate, valid=True)
        return DataLoader(
            dataset=validation_dataset,
            batch_size=batch_size,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_fn,
        )

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack(outputs).mean()

        self.log('valid_loss', avg_loss)


if __name__ == '__main__':
    model_path = 'best_models/asr_model-epoch=50.ckpt'
    model = SpeechRecognition(**model_hyper_parameters)

    if model_path:
        speech_nn = ASRNuralNetwork.load_from_checkpoint(model_path, model=model)
    else:
        speech_nn = ASRNuralNetwork(model)

    logger = TensorBoardLogger('model_logs', name='automatic_speech_recognition')

    checkpoint_callback = ModelCheckpoint(
        dirpath='./best_models',
        filename='asr_model-{epoch}',
        monitor='valid_loss',
        mode='min',
        verbose=True,
    )

    trainer = pl.Trainer(
        max_epochs=3,
        gpus=0,
        callbacks=[checkpoint_callback],
        logger=logger,
        gradient_clip_val=1.0,
        val_check_interval=1.0,
        fast_dev_run=False,
    )

    trainer.fit(speech_nn)
