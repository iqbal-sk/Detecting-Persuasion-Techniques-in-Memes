import torch
from .base_model import BaseModel


class MultiModal(BaseModel):
    def __init__(self, img_feature_size, text_feature_size):
        super(MultiModal, self).__init__()
        # self.resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.image_features_size = img_feature_size
        self.text_features_size = text_feature_size
        infeatures = self.image_features_size + self.text_features_size

        # self.resnet50 = torch.nn.Sequential(*(list(self.resnet50.children())[:-1]))
        # Freeze the parameters

        # for param in self.resnet50.parameters():
        #     param.requires_grad = False

        self.linear_level1 = nn.Sequential(nn.Linear(infeatures, 4096), nn.ReLU(), nn.Linear(4096, 1024), nn.ReLU(),
                                           nn.Linear(1024, 256), nn.ReLU())
        self.linear_level2 = nn.Sequential(nn.Linear(infeatures, 4096), nn.ReLU(), nn.Linear(4096, 1024), nn.ReLU(),
                                           nn.Linear(1024, 256), nn.ReLU())
        self.linear_level3 = nn.Sequential(nn.Linear(infeatures, 4096), nn.ReLU(), nn.Linear(4096, 1024), nn.ReLU(),
                                           nn.Linear(1024, 256), nn.ReLU())
        self.linear_level4 = nn.Sequential(nn.Linear(infeatures, 4096), nn.ReLU(), nn.Linear(4096, 1024), nn.ReLU(),
                                           nn.Linear(1024, 256), nn.ReLU())
        self.linear_level5 = nn.Sequential(nn.Linear(infeatures, 4096), nn.ReLU(), nn.Linear(4096, 1024), nn.ReLU(),
                                           nn.Linear(1024, 256), nn.ReLU())

        self.sigmoid_reg1 = nn.Sequential(nn.Linear(256, 2))
        self.sigmoid_reg2 = nn.Sequential(nn.Linear(256 * 2, 4))
        self.sigmoid_reg3 = nn.Sequential(nn.Linear(256 * 3, 15), nn.Sigmoid())
        self.sigmoid_reg4 = nn.Sequential(nn.Linear(256 * 4, 13), nn.Sigmoid())
        self.sigmoid_reg5 = nn.Sequential(nn.Linear(256 * 5, 7), nn.Sigmoid())

    def forward(self, text_features, image_features):
        # _, embeddings = self.embeddings(input_ids, attention_mask, token_type_ids, return_dict=False)
        # image_features = self.resnet50(images).squeeze().squeeze()

        # print(image_features.shape)
        # print(embeddings.shape)
        multimodal_embeddings = torch.cat((text_features, image_features), dim=1)

        lvl1_rep = self.linear_level1(multimodal_embeddings)
        lvl2_rep = self.linear_level2(multimodal_embeddings)
        lvl3_rep = self.linear_level3(multimodal_embeddings)
        lvl4_rep = self.linear_level4(multimodal_embeddings)
        lvl5_rep = self.linear_level5(multimodal_embeddings)

        lvl1_pred = self.sigmoid_reg1(lvl1_rep)
        lvl2_pred = self.sigmoid_reg2(torch.cat((lvl1_rep, lvl2_rep), dim=1))
        lvl3_pred = self.sigmoid_reg3(torch.cat((lvl1_rep, lvl2_rep, lvl3_rep), dim=1))
        lvl4_pred = self.sigmoid_reg4(torch.cat((lvl1_rep, lvl2_rep, lvl3_rep, lvl4_rep), dim=1))
        lvl5_pred = self.sigmoid_reg5(torch.cat((lvl1_rep, lvl2_rep, lvl3_rep, lvl4_rep, lvl5_rep), dim=1))

        return lvl1_pred, lvl2_pred, lvl3_pred, lvl4_pred, lvl5_pred


import torch
import torch.nn as nn


class MultiModalNER(BaseModel):
    def __init__(self, img_feature_size, text_feature_size, ner_features_size):
        super(MultiModalNER, self).__init__()
        # self.resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.image_features_size = img_feature_size
        self.text_features_size = text_feature_size
        infeatures = self.image_features_size + self.text_features_size + ner_features_size

        # self.resnet50 = torch.nn.Sequential(*(list(self.resnet50.children())[:-1]))
        # Freeze the parameters

        # for param in self.resnet50.parameters():
        #     param.requires_grad = False

        self.linear_level1 = nn.Sequential(nn.Linear(infeatures, 4096), nn.ReLU(), nn.Linear(4096, 1024), nn.ReLU(),
                                           nn.Linear(1024, 256), nn.ReLU())
        self.linear_level2 = nn.Sequential(nn.Linear(infeatures, 4096), nn.ReLU(), nn.Linear(4096, 1024), nn.ReLU(),
                                           nn.Linear(1024, 256), nn.ReLU())
        self.linear_level3 = nn.Sequential(nn.Linear(infeatures, 4096), nn.ReLU(), nn.Linear(4096, 1024), nn.ReLU(),
                                           nn.Linear(1024, 256), nn.ReLU())
        self.linear_level4 = nn.Sequential(nn.Linear(infeatures, 4096), nn.ReLU(), nn.Linear(4096, 1024), nn.ReLU(),
                                           nn.Linear(1024, 256), nn.ReLU())
        self.linear_level5 = nn.Sequential(nn.Linear(infeatures, 4096), nn.ReLU(), nn.Linear(4096, 1024), nn.ReLU(),
                                           nn.Linear(1024, 256), nn.ReLU())

        self.sigmoid_reg1 = nn.Sequential(nn.Linear(256, 2))
        self.sigmoid_reg2 = nn.Sequential(nn.Linear(256 * 2, 4))
        self.sigmoid_reg3 = nn.Sequential(nn.Linear(256 * 3, 15), nn.Sigmoid())
        self.sigmoid_reg4 = nn.Sequential(nn.Linear(256 * 4, 13), nn.Sigmoid())
        self.sigmoid_reg5 = nn.Sequential(nn.Linear(256 * 5, 7), nn.Sigmoid())

    def forward(self, text_features, image_features, ner_features):
        # _, embeddings = self.embeddings(input_ids, attention_mask, token_type_ids, return_dict=False)
        # image_features = self.resnet50(images).squeeze().squeeze()

        # print(image_features.shape)
        # print(embeddings.shape)
        multimodal_embeddings = torch.cat((text_features, image_features, ner_features), dim=1)

        lvl1_rep = self.linear_level1(multimodal_embeddings)
        lvl2_rep = self.linear_level2(multimodal_embeddings)
        lvl3_rep = self.linear_level3(multimodal_embeddings)
        lvl4_rep = self.linear_level4(multimodal_embeddings)
        lvl5_rep = self.linear_level5(multimodal_embeddings)

        lvl1_pred = self.sigmoid_reg1(lvl1_rep)
        lvl2_pred = self.sigmoid_reg2(torch.cat((lvl1_rep, lvl2_rep), dim=1))
        lvl3_pred = self.sigmoid_reg3(torch.cat((lvl1_rep, lvl2_rep, lvl3_rep), dim=1))
        lvl4_pred = self.sigmoid_reg4(torch.cat((lvl1_rep, lvl2_rep, lvl3_rep, lvl4_rep), dim=1))
        lvl5_pred = self.sigmoid_reg5(torch.cat((lvl1_rep, lvl2_rep, lvl3_rep, lvl4_rep, lvl5_rep), dim=1))

        return lvl1_pred, lvl2_pred, lvl3_pred, lvl4_pred, lvl5_pred