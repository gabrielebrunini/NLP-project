import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel, RobertaModel

class Classifier(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.0, use_activation=True):
        super(Classifier, self).__init__()
        self.use_activation = use_activation
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, output_dim)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.dropout(x)
        x = self.linear(x)
        if self.use_activation:
            x = self.tanh(x)
        return x

class FCLayer(nn.Module):
    def __init__(self, input_dim, output_dim, config, dropout_rate=0.0, use_activation=True):
        super().__init__()
        self.dense = torch.nn.Linear(input_dim, output_dim)
        self.layer_norm = torch.nn.LayerNorm(output_dim, eps=config.layer_norm_eps)

        self.gelu = torch.nn.GELU()

    def forward(self, features, **kwargs):
        hidden = self.dense(features)
        hidden = self.gelu(hidden)
        x = self.layer_norm(features + hidden)

        return x



class RBERT(RobertaModel):
    def __init__(self, config, args):
        super(RBERT, self).__init__(config)
        try:
            self.bert = RobertaModel(config=config).from_pretrained(args.model_name_or_path, cache_dir="/cluster/work/lawecon/Work/dominik/transformer_models") # Load pretrained bert
        except:
            self.bert = RobertaModel(config=config).from_pretrained(args.model_name_or_path) # Load pretrained bert

        self.num_labels = config.num_labels

        #self.cls_fc_layer = FCLayer(config.hidden_size, config.hidden_size, config, args.dropout_rate)
        #self.entity_fc_layer = FCLayer(config.hidden_size, config.hidden_size, config, args.dropout_rate)
        self.cls_fc_layer = Classifier(config.hidden_size, config.hidden_size, args.dropout_rate)
        self.entity_fc_layer = Classifier(config.hidden_size, config.hidden_size, args.dropout_rate)

        self.label_classifier = Classifier(
            config.hidden_size * 3,
            config.num_labels,
            args.dropout_rate,
            use_activation=False,
        )

    @staticmethod
    def entity_average(hidden_output, e_mask):
        """
        Average the entity hidden state vectors (H_i ~ H_j)
        :param hidden_output: [batch_size, j-i+1, dim]
        :param e_mask: [batch_size, max_seq_len]
                e.g. e_mask[0] == [0, 0, 0, 1, 1, 1, 0, 0, ... 0]
        :return: [batch_size, dim]
        """
        #print("e_mask", e_mask.shape)

        e_mask_unsqueeze = e_mask.unsqueeze(1)  # [b, 1, j-i+1]
        #print("e_mask", e_mask_unsqueeze.shape)
        length_tensor = (e_mask != 0).sum(dim=1).unsqueeze(1)  # [batch_size, 1]


        # [b, 1, j-i+1] * [b, j-i+1, dim] = [b, 1, dim] -> [b, dim]
        sum_vector = torch.bmm(e_mask_unsqueeze.float(), hidden_output).squeeze(1)
        avg_vector = sum_vector.float() / length_tensor.float()  # broadcasting
        return avg_vector

    """
    def forward(self, input_ids, attention_mask, labels, e1_mask, e2_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)  # sequence_output, pooled_output, (hidden_states), (attentions)
        pooled_output = outputs[1]  # [CLS]
        logits = self.label_classifier(outputs[1])
        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            print(logits.view(-1, self.num_labels), logits.argmax(dim=-1), labels.view(-1))
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        print (loss)
        return outputs  # (loss), logits, (hidden_states), (attentions)


    """

    def forward(self, input_ids, attention_mask, labels, e1_mask, e2_mask):
        #print (input_ids, attention_mask)
        outputs = self.bert(
            input_ids, attention_mask=attention_mask)  # sequence_output, pooled_output, (hidden_states), (attentions)

        sequence_output = outputs[0]
        #print(sequence_output.shape)
        pooled_output = outputs[1]  # [CLS]

        # Average
        e1_h = self.entity_average(sequence_output, e1_mask)
        e2_h = self.entity_average(sequence_output, e2_mask)

        # Dropout -> tanh -> fc_layer (Share FC layer for e1 and e2)
        # perhaps add residual connections?
        pooled_output = self.cls_fc_layer(outputs[1])
        e1_h = self.entity_fc_layer(e1_h)
        e2_h = self.entity_fc_layer(e2_h)

        # Concat -> fc_layer
        concat_h = torch.cat([pooled_output, e1_h, e2_h], dim=-1)

        logits = self.label_classifier(concat_h)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        # Softmax
        if labels is not None:
            if self.num_labels == 1:
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

                #loss = loss_fct(logits, labels)

            outputs = (loss,) + outputs
        print (loss)
        return outputs  # (loss), logits, (hidden_states), (attentions)

