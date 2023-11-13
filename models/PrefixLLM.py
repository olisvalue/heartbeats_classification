import torch
from torch import nn, Tensor
import torch.nn as nn
from torch.nn import functional as nnf
from transformers import GPT2Model, GPT2Tokenizer
import numpy as np
import math

from torch.nn import functional as nnf

from models.PANClassifier import PANClassifier
from .Transformer import * # transformer

num_head = 8

class PrefixLLM(nn.Module):
     
    def __init__(self, prefix_size_dict, encoder_freeze = True,
                 decoder_freeze = True, header_freeze = False,
                 map_networks_freeze = False, temporal_num_layers = 4,
                 global_num_layers = 4, device = 0):
        super(PrefixLLM, self).__init__()

        # self.beam_search = beam_search
        self.device = device

        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        
        self.temporal_prefix_length = prefix_size_dict["temporal_prefix_size"]
        self.global_prefix_length = prefix_size_dict["global_prefix_size"]
        self.prefix_length = self.temporal_prefix_length + self.global_prefix_length
        # same with prefix_length
        temporal_clip_length = prefix_size_dict["temporal_prefix_size"]
        global_clip_length = prefix_size_dict["global_prefix_size"]
        
        self.audio_encoder = PANClassifier(num_classes=2, device=device)
        weights_path = '/data/valerii/heartbeats_classification/data/train_record/pann_balanced2/best_model'
        params = torch.load(weights_path, map_location='cuda:' + str(device))
        self.audio_encoder.load_state_dict(params)

        self.gpt = GPT2Model.from_pretrained("gpt2")

        self.gpt_embedding_size = self.gpt.wte.weight.shape[1] # 768
        
        self.temporal_mappingnetwork = MappingNetwork_forTemporalFeature(dim_embedding = self.gpt_embedding_size, 
                                        prefix_length = self.temporal_prefix_length, clip_length = temporal_clip_length, 
                                        num_layers = temporal_num_layers, device = device)   
    
        self.global_mappingnetwork = MappingNetwork_forGlobalFeature(dim_embedding = self.gpt_embedding_size, 
                                        prefix_length = self.global_prefix_length, clip_length = global_clip_length, 
                                        num_layers = global_num_layers, device = device)
        
        self.language_header = nn.Linear(768, 50257, bias=False) # 50257 : original vocabulary size of GPT2
        header_gpt2_header_params = '/data/valerii/heartbeats_classification/models/weights/PreTrained_GPT2Header.pt'
        self.language_header.load_state_dict(torch.load(header_gpt2_header_params)) # use pre-trained header
        # nn.init.kaiming_uniform_(self.language_header.weight)

        if encoder_freeze == True :
            for param in self.audio_encoder.parameters():
                param.requires_grad = False
            print("Encoder freezing")

        if map_networks_freeze == True:
            for param in self.temporal_mappingnetwork.parameters():
                param.requires_grad = False
            for param in self.global_mappingnetwork.parameters():
                param.requires_grad = False
            print("Mapping network freezed")
            
        if decoder_freeze == True :
            for param in self.gpt.parameters():
                param.requires_grad = False
            print("GPT2 has been freezed")  

        if header_freeze == True:
            for param in self.language_header.parameters():
                param.requires_grad = False
                is_header_freeze = True
                print("header has been freezed")
        else :
            print("header is training")

    def forward(self, audio, tokens = None, mask = None, beam_search = True, with_softmax = True, check_prefix = False):
        
        temporal_feature, global_feature = self.audio_encoder(audio)
        
        temporal_prefix_vector = self.temporal_mappingnetwork(temporal_feature).view(-1, self.temporal_prefix_length, self.gpt_embedding_size)
        global_prefix_vector = self.global_mappingnetwork(global_feature).view(-1, self.global_prefix_length, self.gpt_embedding_size)

        prefix_vectors = torch.cat((temporal_prefix_vector, global_prefix_vector), dim=1) 

        if self.training :
            embedding_text = self.gpt.wte(tokens.to(self.device))
            embedding_cat = torch.cat((prefix_vectors, embedding_text), dim=1)
    
            out = self.gpt(inputs_embeds=embedding_cat.to(self.device), attention_mask=mask.to(self.device))
            out_hidden_states = out[0]
            
            logits = self.language_header(out_hidden_states)

            return logits
        else :
            if beam_search == True and check_prefix == True :
                return self.generate_beam(prefix_vectors, with_softmax = with_softmax), prefix_vectors  
            elif  beam_search == True and check_prefix == False :
                return self.generate_beam(prefix_vectors, with_softmax = with_softmax)
            elif  beam_search == False and check_prefix == True:   
                return self.generate(prefix_vectors, with_softmax = with_softmax), prefix_vectors
            elif  beam_search == False and check_prefix == False:   
                return self.generate(prefix_vectors, with_softmax = with_softmax)


    #not differentiable!!!!      
    def get_semantic(self, prefix_vectors):
        (batch_size, sequence_length, emb_size) = prefix_vectors.shape 
        prefix_reshaped = prefix_vectors.view(-1, emb_size)
        # print('shape prefix reshaped: ', prefix_reshaped.shape)
        semantic_logits = self.prefix_projection(prefix_reshaped)
        # print('semantic logits after linear shape: ', semantic_logits.shape)
        semantic_logits = semantic_logits.view(batch_size, sequence_length, 50257)
        # print('shape of semantic logits after reshape back: ', semantic_logits.shape)
        semantic_logits = semantic_logits.argmax(-1)

        # print('after softmax: ', semantic_logits.shape)
        semantic_vectors = self.gpt.wte(semantic_logits)
        # print('after gpt wte: ', prefix_vectors)
        return semantic_vectors


 
    def get_logits_for_inference(self, generated) :
        
        out = self.gpt(inputs_embeds=generated)
        out_hidden_states = out[0]
        logits = self.language_header(out_hidden_states)
            
        return logits
    
    def generate_beam(self, prefix_projections, beam_size = 5, with_softmax = True) :
        
        entry_count = prefix_projections.size()[0]
        entry_length = 67
        temperature=1.0
                
       
        stop_token_index = self.tokenizer.encode(".")[0]
        
        output_texts_list = []
        
        for entry_idx in range(entry_count):
            
            generated = prefix_projections[entry_idx,:,:].unsqueeze(0)
            scores = None
            tokens = None
            seq_lengths = torch.ones(beam_size, device=self.device)
            is_stopped = torch.zeros(beam_size, device=self.device, dtype=torch.bool)
            
            for i in range(entry_length):
                
                logits = self.get_logits_for_inference(generated)
                
                logits = logits[:, -1, :] / (temperature)
                if with_softmax == True:
                    logits = logits.softmax(-1).log()
                if scores is None:
                    scores, next_tokens = logits.topk(beam_size, -1)
                    generated = generated.expand(beam_size, *generated.shape[1:])
                    next_tokens, scores = next_tokens.permute(1, 0), scores.squeeze(0)
                    if tokens is None:
                        tokens = next_tokens
                    else:
                        tokens = tokens.expand(beam_size, *tokens.shape[1:])
                        tokens = torch.cat((tokens, next_tokens), dim=1)
                else:
                    logits[is_stopped] = -float(np.inf)
                    logits[is_stopped, 0] = 0
                    scores_sum = scores[:, None] + logits
                    seq_lengths[~is_stopped] += 1
                    scores_sum_average = scores_sum / seq_lengths[:, None]
                    scores_sum_average, next_tokens = scores_sum_average.view(-1).topk(beam_size, -1)
                    next_tokens_source = torch.div(next_tokens, scores_sum.shape[1], rounding_mode='floor')
#                     next_tokens_source = next_tokens // scores_sum.shape[1]
                    seq_lengths = seq_lengths[next_tokens_source]
                    next_tokens = next_tokens % scores_sum.shape[1]
                    next_tokens = next_tokens.unsqueeze(1)
                    tokens = tokens[next_tokens_source]
                    tokens = torch.cat((tokens, next_tokens), dim=1)
                    generated = generated[next_tokens_source]
                    scores = scores_sum_average * seq_lengths
                    is_stopped = is_stopped[next_tokens_source]
                next_token_embed = self.gpt.wte(next_tokens.squeeze()).view(
                    generated.shape[0], 1, -1
                )
                generated = torch.cat((generated, next_token_embed), dim=1)
                is_stopped = is_stopped + next_tokens.eq(stop_token_index).squeeze()
                del logits 
                if is_stopped.all():
                    del generated
                    break
            scores = scores / seq_lengths
            output_list = tokens.cpu().numpy()
            output_texts = [
                self.tokenizer.decode(output[: int(length)])
                for output, length in zip(output_list, seq_lengths)
            ]
            order = scores.argsort(descending=True)
            output_texts = [output_texts[i] for i in order]

            output_texts_list.append(output_texts)
        
        return output_texts_list
    
    def generate(self, prefix_projections, with_softmax = True) :
        temperature = 1.0
        entry_length = 67
        top_p = 0.8
        
        # print('prefix shape: ',prefix_projections.shape)
        
        stop_token_index = self.tokenizer.encode(".")[0]

        filter_value = -float("Inf")
        generated_list = []
        
        entry_count = prefix_projections.size()[0]
        
        
        for entry_idx in range(entry_count):

            generated = prefix_projections[entry_idx]
            
            tokens = None 
            
            for i in range(entry_length):
                
                logits = self.get_logits_for_inference(generated)
                # print(logits.shape, logits)
                logits = logits[-1, :] / (temperature)

                #use log softmax or not use? 
                if with_softmax == True:
                    logits = logits.softmax(-1).log()

                # print(logits.shape)


                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(
                            nnf.softmax(sorted_logits, dim=-1), dim=-1
                        )
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                            ..., :-1
                        ].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]


                logits[indices_to_remove] = filter_value

                # print("logits right before next token: ", logits.shape,logits)
                next_token = torch.argmax(logits, -1).unsqueeze(0)
                
                # probs = torch.softmax(logits, -1)
                # max_proba = torch.max(probs, -1)
                # max_proba_pos = torch.argmax(probs, -1)
                # print("probs:", probs)
                # print("next token (argmax):", next_token)
                # print("max probability ", max_proba, "and its position: ", max_proba_pos)

                # print('next token shape: ', next_token.shape)
                next_token_embed = self.gpt.wte(next_token)
                
                if tokens is None:
                    tokens = next_token
                else:
                    tokens = torch.cat((tokens, next_token), dim=0)

                # print('genetated shape:', generated.shape)
                # print('next tok emb shape:', next_token_embed.shape)
                generated = torch.cat((generated, next_token_embed), dim=0)

                # print("next token:", next_token)
                if stop_token_index == next_token:
                    break

            output_list = list(tokens.squeeze().cpu().numpy())
            output_text = self.tokenizer.decode(output_list)
            generated_list.append(output_text)
        return generated_list
     
def get_PANNs_enc(device):
    audio_encoder = Cnn14(sample_rate=16000, window_size=512, 
                hop_size=320, mel_bins=64, fmin=50, fmax=14000, 
                classes_num=527)

    checkpoint_path = "./models/PANNs/Cnn14_16k_mAP=0.438.pth"
    if type(device) == int:
        checkpoint = torch.load(checkpoint_path, map_location='cuda:'+str(device))
    else:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    audio_encoder.load_state_dict(checkpoint['model'])

    audio_encoder = audio_encoder.to(device)

    return audio_encoder

# PE from PyTorch(link : ) 
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class MappingNetwork_forTemporalFeature(nn.Module):
    def forward(self, x):
        
        x = self.conv(x) # [batch_size, 2048, 15, 2] -> [batch_size, 768, 15, 1]
        x = self.bn_conv(x) 
        
        x = self.relu_conv(x)
        
        x = torch.squeeze(x, 3) # [batch_size, 768, 15, 1] -> [batch_size, 768, 15]
        
        x = x.permute(2, 0, 1).contiguous() # [batch_size, 768, 15] -> [15, batch_size, 768]
        x = self.pos_encoder(x) # positional encoding
        
        x = x.permute(1, 0, 2).contiguous() # [15, batch_size, 768] -> [batch_size, 15, 768]
        
        prefix = self.prefix_const.unsqueeze(0).expand(x.shape[0], *self.prefix_const.shape)
        prefix = torch.cat((x, prefix), dim=1)
        out = self.transformer(prefix)[:, -self.prefix_length:]
        return out

    def __init__(self, dim_embedding: int, prefix_length: int, clip_length: int, num_layers: int = 8, device = 'cuda:1'):
        super(MappingNetwork_forTemporalFeature, self).__init__()
        
        self.prefix_length = prefix_length

        self.device = device
        self.transformer = Transformer(dim_embedding, num_head, num_layers)
        
        self.conv = nn.Conv2d(2048, dim_embedding, (1, 2), stride=(1, 1), padding=(0, 0)) # [2048, 15, 2] -> [768, 15, 1]
        torch.nn.init.kaiming_uniform_(self.conv.weight)
        
        self.bn_conv = nn.BatchNorm2d(dim_embedding)
        self.relu_conv = nn.ReLU()
        
        self.prefix_const = nn.Parameter(torch.randn(prefix_length, dim_embedding), requires_grad=True)
        torch.nn.init.kaiming_uniform_(self.prefix_const)
        
        self.pos_encoder = PositionalEncoding(d_model=dim_embedding, dropout = 0.5) # positional encoding
        
        print("temporal feature ver's mapping network : num_head =", num_head, "num_layers =", num_layers, "prefix_vector_length =", prefix_length)
        

class MappingNetwork_forGlobalFeature(nn.Module):

    def forward(self, x):
        dummy_val = torch.zeros(x.size()[0], 1).to(self.device)
        x = torch.cat((x, dummy_val), dim=1) # [batch_size, 527] -> [batch_size, 528]
        
        x = (x.unsqueeze(1)).unsqueeze(1) # [batch_size, 528] -> [batch_size, 1, 1, 528]
        x = self.conv(x) # [batch_size, 1, 1, 528] -> [batch_size, 768, 1, 11] 
        x = self.bn_conv(x)
      
        x = self.relu_conv(x)
        
        x = torch.squeeze(x, 2) # [batch_size, 768, 1, 11] -> [batch_size, 768, 11]
        
        x = x.permute(0, 2, 1).contiguous() # [batch_size, 768, 11] -> [batch_size, 11, 768]
        
        prefix = self.prefix_const.unsqueeze(0).expand(x.shape[0], *self.prefix_const.shape)
        prefix = torch.cat((x, prefix), dim=1)
        out = self.transformer(prefix)[:, -self.prefix_length:]
        return out

    def __init__(self, dim_embedding: int, prefix_length: int, clip_length: int, num_layers : int = 8, device = 'cuda:1'):
        super(MappingNetwork_forGlobalFeature, self).__init__()

        self.device = device

        self.prefix_length = prefix_length
        self.transformer = Transformer(dim_embedding, num_head, num_layers)
        
        self.conv = nn.Conv2d(1, 768, (1, 48), stride=(1, 48), padding=(0, 0))
        torch.nn.init.kaiming_uniform_(self.conv.weight)
        
        self.bn_conv = nn.BatchNorm2d(dim_embedding)
        self.relu_conv = nn.ReLU()
        
        self.prefix_const = nn.Parameter(torch.randn(prefix_length, dim_embedding), requires_grad=True)
        torch.nn.init.kaiming_uniform_(self.prefix_const)
        
        print("global feature ver's mapping network : num_head =", num_head, "num_layers =", num_layers, "prefix_vector_length =", prefix_length)