import numpy as np
import torch
from torch import optim, nn, Tensor
from torchmetrics import Accuracy
import pytorch_lightning as pl
import mjx
from mjx import Agent, Action

class _MLP(pl.LightningModule):

    def __init__(self, obs_size=107*34, n_actions=181, hidden_size=107*34):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
        )
        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = Accuracy(task="multiclass", num_classes=n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self.forward(x)
        loss = self.criterion(preds, y)
        acc = self.accuracy(preds, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        preds = self.forward(x)
        loss = self.criterion(preds, y)
        acc = self.accuracy(preds, y)
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self.forward(x)
        loss = self.criterion(preds, y)
        acc = self.accuracy(preds, y)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def configure_optimizers(self):
        return self.optimizer

class MLPAgent(Agent):
    def __init__(self) -> None:
        super().__init__()

        model = _MLP()
        
        state = torch.load('./reinforce_model.pth')
        # model.load_state_dict(state)
        model.load_state_dict(state['model_state_dict'])
        self.model = model

    def act(self, observation) -> Action:
        legal_actions = observation.legal_actions()
        if len(legal_actions) == 1:
            return legal_actions[0]
        
        feature_small = observation.to_features(feature_name="mjx-small-v0")
        feature_han22 = observation.to_features(feature_name="han22-v0")
        feature_han22 = np.delete(feature_han22, [0, 3], 0) # duplicate features
        feature = np.concatenate([feature_small, feature_han22], axis=0)
        
        with torch.no_grad():
            action_logit = self.model(Tensor(feature.ravel()))
        action_proba = torch.sigmoid(action_logit).numpy()
        
        mask = observation.action_mask()
        action_idx = (mask * action_proba).argmax()
        return mjx.Action.select_from(action_idx, legal_actions)
