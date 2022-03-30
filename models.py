#coding=utf-8
import torch.nn as nn
import torch
import torch.nn.functional as F


def attnetwork(encoder_out, final_hidden):
    hidden = final_hidden.squeeze(0)
    #M = torch.tanh(encoder_out)
    attn_weights = torch.bmm(encoder_out, hidden.unsqueeze(2)).squeeze(2)
    soft_attn_weights = F.softmax(attn_weights, 1)
    new_hidden = torch.bmm(encoder_out.transpose(1,2), soft_attn_weights.unsqueeze(2)).squeeze(2)
    #print (wt.shape, new_hidden.shape)
    # new_hidden = torch.tanh(new_hidden)
    #print ('UP:', new_hidden, new_hidden.shape)
    
    return new_hidden


class arc(nn.Module):
    def __init__(self):
        super(arc, self).__init__()

        self.embedding = nn.Sequential(
            nn.Conv1d(in_channels=100,  #100，99
                      out_channels=64,
                      kernel_size=5,
                      stride=1,
                      padding=2),
            nn.ReLU(True)           #64， 99   
        )

        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=128,  #128，99
                      out_channels=64,
                      kernel_size=5,
                      stride=1,
                      padding=2),
            nn.ReLU(True),           #64， 99   
            #nn.MaxPool1d(2)         #64, 49
        )
        self.cnn2 = nn.Sequential(
            nn.Conv1d(in_channels=64,  #64，99
                      out_channels=32,
                      kernel_size=5,
                      stride=1,
                      padding=2),
            nn.ReLU(True),           #32，99   
            # nn.MaxPool1d(2)          #32, 25
        )

        self.BiLSTM = nn.Sequential(
            nn.LSTM(input_size=32,
                    hidden_size=16,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=True,
                    bias=True)              #(64,16,32)
        )       

        self.Prediction = nn.Sequential(
            nn.Linear(16,8),
            nn.Dropout(0.2),
            nn.Linear(8,1),
            nn.Sigmoid(),
        )

    def forward(self, input):
        cnn_output1 = self.cnn(input)
        cnn_output = self.cnn2(cnn_output1)
        cnn_output = cnn_output.transpose(1,2)
        bilstm_out, (hn, cn) = self.BiLSTM(cnn_output)
        bilstm_out = bilstm_out[:, :, :16] + bilstm_out[:, :, 16:] #sum bidir outputs F+B
        fbhn = (hn[-2,:,:]+hn[-1,:,:]).unsqueeze(0)
        att_out = attnetwork(bilstm_out, fbhn)
        result = self.Prediction(att_out)
        return result

