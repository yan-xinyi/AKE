# -*- coding: utf-8 -*-
# @Time    : 2023/7/30 11:25
# @Author  : Xinyi Yan

'''
   This python file mainly contains code for training, testing and evaluation of AKE models for self-attention based Bi-LSTM.
   The main function is satt_bl(), called from main().
      1* Through the TextDataSet (Dataset): build the dictionary for vovabulary and cognitive signals.
      2* Build SATTBiLSTM() model
      3* Start training and calculate the loss value.
      4* Conduct testing, again read the data first, build the model, in the prediction, evaluate its results.
'''

import numpy as np
import logging
from torch import nn, optim
from tqdm import tqdm
from ..utils import *
from ..config import *
from ..evaluate import *
from torch.utils.data import Dataset, DataLoader


# LayerNorm Function for the self-attention layer
class LayerNorm(nn.Module):
    # Transform the q,k,v vector with dimension [batch_size * seq_length * hidden_size] into [batch_size * num_attention_heads * seq_length * attention_head_size]，便于后面做 Multi-Head Attention。
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


# Define Self-Attention Mechanism Function
class SelfAttention(nn.Module):
    def __init__(self, num_attention_heads, input_size, hidden_size, hidden_dropout_prob):
        super(SelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = hidden_size

        self.query = nn.Linear(input_size, self.all_head_size)
        self.key = nn.Linear(input_size, self.all_head_size)
        self.value = nn.Linear(input_size, self.all_head_size)
        self.value1 = nn.Linear(self.all_head_size + fs_num, self.all_head_size)

        self.attn_dropout = nn.Dropout(p=dropout_value)

        # After self-attention, do a feed-forward fully-connected LayerNorm output.
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.Linear = nn.Linear(2*hidden_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size, eps=1e-12)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor, inputs):

        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)
        mixed_value_layer = torch.cat([mixed_value_layer, inputs['eeg'][:,:,4:6]], dim=-1)
        mixed_value_layer = self.value1(mixed_value_layer)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the *raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # [batch_size heads seq_len seq_len] scores
        # [batch_size 1 1 seq_len]

        # attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        # Fixme
        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        input_tensor = self.Linear(input_tensor)
        outputs = hidden_states + input_tensor
        hidden_states = self.LayerNorm(outputs)

        return hidden_states
    

# Define SATTBiLSTM() model
class SATTBiLSTM(nn.Module):

    def __init__(self, vocab_size, embed_dim,hidden_dim,
                 num_layers=1, num_tags=6):
        super(SATTBiLSTM, self).__init__()

        # glove_model.vocab = m(glove_model.vocab)
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.bilstm = nn.LSTM(input_size=embed_dim + fs_num,
                              hidden_size=hidden_dim, num_layers=num_layers,
                              bidirectional=True, batch_first=True)
        self.layernorm = nn.LayerNorm(normalized_shape=2 * hidden_dim)
        self.fc = nn.Linear(2 * hidden_dim, num_tags)
        self.lstm_dropout = nn.Dropout(p=dropout_value)
        self.linear_dropout = nn.Dropout(p=dropout_value)
        self.self_atten = SelfAttention(num_attention_heads=1, input_size = 2 * hidden_dim, hidden_size=hidden_dim, hidden_dropout_prob=dropout_value)


    def forward(self, inputs, is_training=True):
        # word embedding
        input = self.embedding(inputs['input_ids'])

        input, _ = self.bilstm(input)                                             # bilstm layer 
        input = self.lstm_dropout(torch.relu(self.layernorm(input)))              # layernorm layer 
        outputs = self.self_atten(input, inputs)                                  # self-attention layer
        reps = self.linear_dropout(outputs)

        output = self.fc(reps)
        # print(output.shape)

        return output


# Main Function
def satt_bl():

    # Load the data
    trainLoader = DataLoader(dataset=TextDataSet(train_path, vocab_path, max_length, tag2ids), batch_size=batch_size)
    testLoader = DataLoader(TextDataSet(test_path, vocab_path, max_length, tag2ids), batch_size=batch_size)

    # Define the model
    model = SATTBiLSTM(vocab_size=vocab_size,
                       embed_dim=embed_dim,
                       hidden_dim=hidden_dim).to(device)
    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_P, best_R, best_F, best_epoch = 0.0, 0.0, 0.0, 0
    for epoch in range(epochs):
        print("epoch " + str(epoch + 1) + " is starting!")
        
        # Train the model
        model.train()
        avg_loss = []
        with tqdm(trainLoader) as pbar_train:
            for inputs in pbar_train:
                loss = model(inputs)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pbar_train.set_description('loss: %.3f' % loss.item())
                avg_loss.append(loss.item())
        print("avg_loss: %.2f" % np.average(avg_loss))

        # Test the model
        model.eval()
        y_true, y_pred = [], []

        targets = []
        prediction = []
        with tqdm(testLoader) as pbar_test:
            for inputs in pbar_test:
                logit = model(inputs, is_training=False)
                output = get_outputs(logit)
                outputs = output.tolist()
                true_tags = [tag[:inputs['text_len'][i]] for i, tag in enumerate(inputs['tags'].cuda().tolist())]
                y_true.extend([i for item in true_tags for i in item])
                y_pred.extend([i for output in outputs for i in output])

                # Get id2word to words_set
                words_set = get_words(inputs, testLoader)
                # Get ground truth keyphrase from true_tags to targets
                targets = get_gold(true_tags, targets, words_set)
                # Get predicted keyphrase from out_list to prediction
                prediction = get_pred(outputs, prediction, words_set)



        # calculate the P, R, F1 value of the test data
        P, R, F = evaluate(prediction, targets)

        if F > best_F:
            best_F = F
            best_P = P
            best_R = R
            best_epoch += 1

        P_str = "F:" + "\t" + str(P)
        R_str = "F:" + "\t" + str(R)
        F_str = "F:" + "\t" + str(F)

        logging.info("Epoch" + str(epoch + 1) + "'s Evaluation:")
        logging.info("P is :" + P_str)
        logging.info("R is :" + R_str)
        logging.info("F is :" + F_str)

    return best_P, best_R, best_F, best_epoch
