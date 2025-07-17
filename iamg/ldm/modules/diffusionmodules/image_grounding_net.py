import torch
import torch.nn as nn
from ldm.modules.attention import BasicTransformerBlock
from ldm.modules.diffusionmodules.util import checkpoint, FourierEmbedder
import torch.nn.functional as F

from ldm.modules.x_transformer import AbsolutePositionalEmbedding, FixedPositionalEmbedding


class HOIPositionNetV5(nn.Module):
    """
    Transform interaction information into interaction condition tokens
    """
    def __init__(self, in_dim, out_dim, fourier_freqs=96, max_interactions=30):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.fourier_embedder = FourierEmbedder(num_freqs=fourier_freqs)
        self.interaction_embedding = AbsolutePositionalEmbedding(dim=out_dim, max_seq_len=max_interactions)
        self.position_embedding = AbsolutePositionalEmbedding(dim=out_dim, max_seq_len=3)
        self.position_dim = fourier_freqs * 2 * 4  # 2 is sin&cos, 4 is xyxy

        self.linears = nn.Sequential(
            nn.Linear(258, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
        )
        
        self.linears_2 = nn.Sequential(
            nn.Linear(768, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, 768),
        )
        
        # self.linear_action = nn.Sequential(
        #     nn.Linear(self.in_dim + self.position_dim, 512),
        #     nn.SiLU(),
        #     nn.Linear(512, 512),
        #     nn.SiLU(),
        #     nn.Linear(512, out_dim),
        # )

        self.null_fore_feature = torch.nn.Parameter(torch.zeros([257*768]))
        #self.null_action_feature = torch.nn.Parameter(torch.zeros([self.in_dim]))
        self.null_position_feature = torch.nn.Parameter(torch.zeros([768]))

    def get_between_box(self, bbox1, bbox2):
        """ Between Set Operation
        Operation of Box A between Box B from Prof. Jiang idea
        """
        all_x = torch.cat([bbox1[:, :, 0::2], bbox2[:, :, 0::2]],dim=-1)
        all_y = torch.cat([bbox1[:, :, 1::2], bbox2[:, :, 1::2]],dim=-1)
        all_x, _ = all_x.sort()
        all_y, _ = all_y.sort()
        return torch.stack([all_x[:,:,1], all_y[:,:,1], all_x[:,:,2], all_y[:,:,2]],2)

    def forward(self, subject_boxes, object_boxes, masks,
                subject_fore_embeddings, object_fore_embeddings):
        B, N, _ = subject_boxes.shape
        fore_masks = masks.unsqueeze(-1).unsqueeze(-1).repeat(1,1,257,768)
        xyxy_masks = masks.unsqueeze(-1).unsqueeze(-1).repeat(1,1,1,768)
        # embedding position (it may include padding as placeholder)
        #action_boxes = self.get_between_box(subject_boxes, object_boxes)
        subject_xyxy_embedding = self.fourier_embedder(subject_boxes).unsqueeze(-1).view(B,N,1,-1)  # B*N*4 --> B*N*1024, align with fore_feature which is B*N*257*1024,then concat as B*N*258*1024
        object_xyxy_embedding = self.fourier_embedder(object_boxes).unsqueeze(-1).view(B,N,1,-1)  # B*N*4 --> B*N*C
        #action_xyxy_embedding = self.fourier_embedder(action_boxes)  # B*N*4 --> B*N*C

        # learnable null embedding
        fore_null = self.null_fore_feature.view(1, 1, 257, 768)
        xyxy_null = self.null_position_feature.view(1, 1, 1, 768)
        #action_null = self.null_action_feature.view(1, 1, -1)

        # replace padding with learnable null embedding
        subject_fore_embeddings = subject_fore_embeddings * fore_masks + (1 - fore_masks) * fore_null
        object_fore_embeddings = object_fore_embeddings * fore_masks + (1 - fore_masks) * fore_null

        subject_xyxy_embedding = subject_xyxy_embedding * xyxy_masks + (1 - xyxy_masks) * xyxy_null
        object_xyxy_embedding = object_xyxy_embedding * xyxy_masks + (1 - xyxy_masks) * xyxy_null
        #action_xyxy_embedding = action_xyxy_embedding * masks + (1 - masks) * xyxy_null

        #action_positive_embeddings = action_positive_embeddings * masks + (1 - masks) * action_null

        objs_subject = self.linears(torch.cat([subject_fore_embeddings, subject_xyxy_embedding], dim=2).permute(0,1,3,2)).squeeze(3)  #input dim : B*N*258*1024 ->B*N*1024*258 -> B*N*1024
        objs_object = self.linears(torch.cat([object_fore_embeddings, object_xyxy_embedding], dim=2).permute(0,1,3,2)).squeeze(3) #output dim :   
        
        objs_subject = self.linears_2(objs_subject)
        objs_object = self.linears_2(objs_object)
        #objs_action = self.linear_action(torch.cat([action_positive_embeddings, action_xyxy_embedding], dim=-1))

        # objs_subject = objs_subject + self.interaction_embedding(objs_subject)
        # objs_object = objs_object + self.interaction_embedding(objs_object)
        # objs_action = objs_action + self.interaction_embedding(objs_action)

        # objs_subject = objs_subject + self.position_embedding.emb(torch.tensor(0).to(objs_subject.device))
        # objs_object = objs_object + self.position_embedding.emb(torch.tensor(1).to(objs_object.device))
        # objs_action = objs_action + self.position_embedding.emb(torch.tensor(2).to(objs_action.device))

        objs = torch.cat([objs_subject, objs_object], dim=1)

        assert objs.shape == torch.Size([B, N*2, self.out_dim])
        return objs
    
    
    
class HOIPositionNetV5_only_dino(nn.Module):
    """
    Transform interaction information into interaction condition tokens
    """
    def __init__(self, in_dim, out_dim, fourier_freqs=96, max_interactions=30):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        #self.fourier_embedder = FourierEmbedder(num_freqs=fourier_freqs)
        #self.interaction_embedding = AbsolutePositionalEmbedding(dim=out_dim, max_seq_len=max_interactions)
        #self.position_embedding = AbsolutePositionalEmbedding(dim=out_dim, max_seq_len=3)
        #self.position_dim = fourier_freqs * 2 * 4  # 2 is sin&cos, 4 is xyxy

        self.linears = nn.Sequential(
            nn.Linear(257, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
        )
        
        self.linears_2 = nn.Sequential(
            nn.Linear(768, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, 768),
        )
        
        # self.linear_action = nn.Sequential(
        #     nn.Linear(self.in_dim + self.position_dim, 512),
        #     nn.SiLU(),
        #     nn.Linear(512, 512),
        #     nn.SiLU(),
        #     nn.Linear(512, out_dim),
        # )

        self.null_fore_feature = torch.nn.Parameter(torch.zeros([257*768]))
        #self.null_action_feature = torch.nn.Parameter(torch.zeros([self.in_dim]))
        #self.null_position_feature = torch.nn.Parameter(torch.zeros([768]))

    # def get_between_box(self, bbox1, bbox2):
    #     """ Between Set Operation
    #     Operation of Box A between Box B from Prof. Jiang idea
    #     """
    #     all_x = torch.cat([bbox1[:, :, 0::2], bbox2[:, :, 0::2]],dim=-1)
    #     all_y = torch.cat([bbox1[:, :, 1::2], bbox2[:, :, 1::2]],dim=-1)
    #     all_x, _ = all_x.sort()
    #     all_y, _ = all_y.sort()
    #     return torch.stack([all_x[:,:,1], all_y[:,:,1], all_x[:,:,2], all_y[:,:,2]],2)

    def forward(self, subject_boxes, object_boxes, masks,
                subject_fore_embeddings, object_fore_embeddings):
        B, N, _ = subject_boxes.shape
        fore_masks = masks.unsqueeze(-1).unsqueeze(-1).repeat(1,1,257,768)
        #xyxy_masks = masks.unsqueeze(-1).unsqueeze(-1).repeat(1,1,1,768)
        # embedding position (it may include padding as placeholder)
        #action_boxes = self.get_between_box(subject_boxes, object_boxes)
        #subject_xyxy_embedding = self.fourier_embedder(subject_boxes).unsqueeze(-1).view(B,N,1,-1)  # B*N*4 --> B*N*1024, align with fore_feature which is B*N*257*1024,then concat as B*N*258*1024
        #object_xyxy_embedding = self.fourier_embedder(object_boxes).unsqueeze(-1).view(B,N,1,-1)  # B*N*4 --> B*N*C
        #action_xyxy_embedding = self.fourier_embedder(action_boxes)  # B*N*4 --> B*N*C

        # learnable null embedding
        fore_null = self.null_fore_feature.view(1, 1, 257, 768)
        #xyxy_null = self.null_position_feature.view(1, 1, 1, 768)
        #action_null = self.null_action_feature.view(1, 1, -1)

        # replace padding with learnable null embedding
        subject_fore_embeddings = subject_fore_embeddings * fore_masks + (1 - fore_masks) * fore_null
        object_fore_embeddings = object_fore_embeddings * fore_masks + (1 - fore_masks) * fore_null

        #subject_xyxy_embedding = subject_xyxy_embedding * xyxy_masks + (1 - xyxy_masks) * xyxy_null
        #object_xyxy_embedding = object_xyxy_embedding * xyxy_masks + (1 - xyxy_masks) * xyxy_null
        #action_xyxy_embedding = action_xyxy_embedding * masks + (1 - masks) * xyxy_null

        #action_positive_embeddings = action_positive_embeddings * masks + (1 - masks) * action_null

        #objs_subject = self.linears(torch.cat([subject_fore_embeddings, subject_xyxy_embedding], dim=2).permute(0,1,3,2)).squeeze(3)  #input dim : B*N*258*1024 ->B*N*1024*258 -> B*N*1024
        #objs_object = self.linears(torch.cat([object_fore_embeddings, object_xyxy_embedding], dim=2).permute(0,1,3,2)).squeeze(3) #output dim :   
        objs_subject = self.linears(subject_fore_embeddings.permute(0,1,3,2)).squeeze(3)  #input dim : B*N*258*1024 ->B*N*1024*258 -> B*N*1024
        objs_object = self.linears(object_fore_embeddings.permute(0,1,3,2)).squeeze(3) #output dim :   
        objs_subject = self.linears_2(objs_subject)
        objs_object = self.linears_2(objs_object)
        #objs_action = self.linear_action(torch.cat([action_positive_embeddings, action_xyxy_embedding], dim=-1))

        # objs_subject = objs_subject + self.interaction_embedding(objs_subject)
        # objs_object = objs_object + self.interaction_embedding(objs_object)
        # objs_action = objs_action + self.interaction_embedding(objs_action)

        # objs_subject = objs_subject + self.position_embedding.emb(torch.tensor(0).to(objs_subject.device))
        # objs_object = objs_object + self.position_embedding.emb(torch.tensor(1).to(objs_object.device))
        # objs_action = objs_action + self.position_embedding.emb(torch.tensor(2).to(objs_action.device))

        objs = torch.cat([objs_subject, objs_object], dim=1)

        assert objs.shape == torch.Size([B, N*2, self.out_dim])
        return objs