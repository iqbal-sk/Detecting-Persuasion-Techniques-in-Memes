import torch
import torch.nn as nn

from .base_model import BaseModel
from config import config
from config.logger import get_logger

logger = get_logger(__name__)


class mBERT(BaseModel):
    def __init__(self):
        super(mBERT, self).__init__()
        hyperparameters = config.training.hyperparameters

        self.dropout_rate = hyperparameters.dropout_rate
        logger.info(f"Initializing mBERT model with dropout rate: {self.dropout_rate}")

        self.linear_level1 = nn.Sequential(
            nn.Linear(768, 1024),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(512, 128),
            nn.ReLU()
        )
        logger.debug("Initialized linear_level1 layers.")

        self.linear_level2 = nn.Sequential(
            nn.Linear(768, 1024),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(512, 128),
            nn.ReLU()
        )
        logger.debug("Initialized linear_level2 layers.")

        self.linear_level3 = nn.Sequential(
            nn.Linear(768, 1024),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(512, 128),
            nn.ReLU()
        )
        logger.debug("Initialized linear_level3 layers.")

        self.linear_level4 = nn.Sequential(
            nn.Linear(768, 1024),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(512, 128),
            nn.ReLU()
        )
        logger.debug("Initialized linear_level4 layers.")

        self.linear_level5 = nn.Sequential(
            nn.Linear(768, 1024),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(512, 128),
            nn.ReLU()
        )
        logger.debug("Initialized linear_level5 layers.")

        self.sigmoid_reg1 = nn.Sequential(
            nn.Linear(128, 2)
        )
        logger.debug("Initialized sigmoid_reg1 layers.")

        self.sigmoid_reg2 = nn.Sequential(
            nn.Linear(128 * 2, 4)
        )
        logger.debug("Initialized sigmoid_reg2 layers.")

        self.sigmoid_reg3 = nn.Sequential(
            nn.Linear(128 * 3, 13),
            nn.Sigmoid()
        )
        logger.debug("Initialized sigmoid_reg3 layers.")

        self.sigmoid_reg4 = nn.Sequential(
            nn.Linear(128 * 4, 13),
            nn.Sigmoid()
        )
        logger.debug("Initialized sigmoid_reg4 layers.")

        self.sigmoid_reg5 = nn.Sequential(
            nn.Linear(128 * 5, 7),
            nn.Sigmoid()
        )
        logger.debug("Initialized sigmoid_reg5 layers.")

        logger.info("mBERT model initialization complete.")

    def forward(self, embeddings):
        logger.debug("Starting forward pass.")

        lvl1_rep = self.linear_level1(embeddings)
        lvl2_rep = self.linear_level2(embeddings)
        lvl3_rep = self.linear_level3(embeddings)
        lvl4_rep = self.linear_level4(embeddings)
        lvl5_rep = self.linear_level5(embeddings)
        logger.debug("Computed representations for all levels.")

        lvl1_pred = self.sigmoid_reg1(lvl1_rep)
        lvl2_pred = self.sigmoid_reg2(torch.cat((lvl1_rep, lvl2_rep), dim=1))
        lvl3_pred = self.sigmoid_reg3(torch.cat((lvl1_rep, lvl2_rep, lvl3_rep), dim=1))
        lvl4_pred = self.sigmoid_reg4(torch.cat((lvl1_rep, lvl2_rep, lvl3_rep, lvl4_rep), dim=1))
        lvl5_pred = self.sigmoid_reg5(torch.cat((lvl1_rep, lvl2_rep, lvl3_rep, lvl4_rep, lvl5_rep), dim=1))
        logger.debug("Generated predictions for all levels.")

        logger.debug("Forward pass completed.")
        return lvl1_pred, lvl2_pred, lvl3_pred, lvl4_pred, lvl5_pred
