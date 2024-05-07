import torch
import torch.nn as nn


class OpenAiSmall(torch.nn.Module):
    def __init__(self):
        super(OpenAiSmall, self).__init__()

        # self.embeddings = BertModel.from_pretrained("bert-base-multilingual-cased")

        # for param in self.embeddings.parameters():
        #     param.requires_grad = False

        dropout_rate = 0.15

        self.linear_level1 = nn.Sequential(nn.Linear(1536, 1024), nn.ReLU(), nn.Dropout(dropout_rate),
                                           nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(dropout_rate),
                                           nn.Linear(512, 128), nn.ReLU())

        self.linear_level2 = nn.Sequential(nn.Linear(1536, 1024), nn.ReLU(), nn.Dropout(dropout_rate),
                                           nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(dropout_rate),
                                           nn.Linear(512, 128), nn.ReLU())
        self.linear_level3 = nn.Sequential(nn.Linear(1536, 1024), nn.ReLU(), nn.Dropout(dropout_rate),
                                           nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(dropout_rate),
                                           nn.Linear(512, 128), nn.ReLU())
        self.linear_level4 = nn.Sequential(nn.Linear(1536, 1024), nn.ReLU(), nn.Dropout(dropout_rate),
                                           nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(dropout_rate),
                                           nn.Linear(512, 128), nn.ReLU())
        self.linear_level5 = nn.Sequential(nn.Linear(1536, 1024), nn.ReLU(), nn.Dropout(dropout_rate),
                                           nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(dropout_rate),
                                           nn.Linear(512, 128), nn.ReLU())

        self.sigmoid_reg1 = nn.Sequential(nn.Linear(128, 2))
        self.sigmoid_reg2 = nn.Sequential(nn.Linear(128 * 2, 4))
        self.sigmoid_reg3 = nn.Sequential(nn.Linear(128 * 3, 13), nn.Sigmoid())
        self.sigmoid_reg4 = nn.Sequential(nn.Linear(128 * 4, 13), nn.Sigmoid())
        self.sigmoid_reg5 = nn.Sequential(nn.Linear(128 * 5, 7), nn.Sigmoid())

    def forward(self, embeddings):
        # embeddings = self.embeddings(input_ids, attention_mask, token_type_ids)

        # embeddings = embeddings.last_hidden_state[:, 0, :]

        lvl1_rep = self.linear_level1(embeddings)
        lvl2_rep = self.linear_level2(embeddings)
        lvl3_rep = self.linear_level3(embeddings)
        lvl4_rep = self.linear_level4(embeddings)
        lvl5_rep = self.linear_level5(embeddings)

        lvl1_pred = self.sigmoid_reg1(lvl1_rep)
        lvl2_pred = self.sigmoid_reg2(torch.cat((lvl1_rep, lvl2_rep), dim=1))
        lvl3_pred = self.sigmoid_reg3(torch.cat((lvl1_rep, lvl2_rep, lvl3_rep), dim=1))
        lvl4_pred = self.sigmoid_reg4(torch.cat((lvl1_rep, lvl2_rep, lvl3_rep, lvl4_rep), dim=1))
        lvl5_pred = self.sigmoid_reg5(torch.cat((lvl1_rep, lvl2_rep, lvl3_rep, lvl4_rep, lvl5_rep), dim=1))

        return lvl1_pred, lvl2_pred, lvl3_pred, lvl4_pred, lvl5_pred
