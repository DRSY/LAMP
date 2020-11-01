'''
Author: roy
Date: 2020-10-30 22:18:56
LastEditTime: 2020-11-01 11:04:04
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /LAMA/utils.py
'''
from transformers import BertForMaskedLM, BertTokenizer
import copy
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torch.distributions import Bernoulli

device = torch.device("cpu")
print(device)


class FoobarPruning(prune.BasePruningMethod):
    """
    Customized Pruning Method
    """
    PRUNING_TYPE = 'unstructured'

    def __init__(self, pregenerated_mask) -> None:
        super().__init__()
        self.pre_generated_mask = pregenerated_mask

    def compute_mask(self, t, default_mask):
        """
        """
        mask = self.pre_generated_mask
        return mask


def Foobar_pruning(module, name, mask):
    """
    util function for pruning parameters of given module.name using corresponding mask generated by relation-specific mask generator
    Parameters:
    module: subclass of nn.Module
    name: name of parameters to be pruned
    id: id for the parameters in the parameters_tobe_pruned list
    """
    sub_module = getattr(module, name)
    shape = sub_module.size()
    assert shape == mask.size(
    ), "size of mask and parameters not consistent: {} != {}".format(mask.size(), shape)
    FoobarPruning.apply(module, name, pregenerated_mask=mask)
    return module


def remove_prune_reparametrization(module, name):
    """
    make pruning permanent
    """
    prune.remove(module, name)

def restore_init_state(model: torch.nn.Module, init_state):
    """
    load copyed initial state dict after prune.remove
    """
    model.load_state_dict(init_state)


def freeze_parameters(model):
    """
    freeze all parameters of input model
    """
    for p in model.parameters():
        p.requires_grad = False


def bernoulli_hard_sampler(probs):
    """
    Hard sampler for bernoulli distribution
    """
    Bernoulli_Sampler = Bernoulli(probs=probs)
    sample = Bernoulli_Sampler.sample()
    log_probs_of_sample = Bernoulli_Sampler.log_prob(sample)
    return sample, log_probs_of_sample


def bernoulli_soft_sampler(logits, temperature: float = 0.1):
    """
    Soft sampler for bernoulli distribution
    """
    uniform_variables = torch.rand(*logits.size())
    assert uniform_variables.shape == logits.shape
    samples = torch.sigmoid(
        (logits + torch.log(uniform_variables) - torch.log(1-uniform_variables)) / temperature)
    return samples

def LAMA(model, tokenizer, input_w_mask, topk=5):
    inputs = tokenizer(input_w_mask, return_tensors='pt')
    mask_id = inputs['input_ids'][0].tolist().index(tokenizer.mask_token_id)
    inputs.to(device)
    outputs = model(**inputs)
    logits = outputs.logits
    probs = torch.softmax(logits[0, mask_id], dim=-1)
    _, indices = torch.topk(probs, k=topk)
    predictions = []
    for token in tokenizer.decode(indices).split(" "):
        predictions.append(token)
    return predictions


if __name__ == "__main__":
    model = nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 768))
    model.train()
    inputs = torch.randn(12, 32)
    pruning_mask_logits = model(inputs)
    assert pruning_mask_logits.requires_grad == True
    pruning_mask_probs = torch.sigmoid(pruning_mask_logits)
    soft_samples = bernoulli_soft_sampler(pruning_mask_logits, temperature=0.1)
    hard_samples, log_probs = bernoulli_hard_sampler(pruning_mask_probs)
    assert soft_samples.requires_grad == True, "no grad associated with soft samples"

    # testing
    bert = BertForMaskedLM.from_pretrained('bert-base-cased', return_dict=True)
    bert.eval()
    freeze_parameters(bert)
    init_state = copy.deepcopy(bert.state_dict())
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    parameters_tobe_pruned = [
        (bert.bert.encoder.layer[0].attention.self.query, 'bias')]
    # prune!
    for module, name in parameters_tobe_pruned:
        Foobar_pruning(module, name, soft_samples[0])

    text = "The capital of England is [MASK]"
    obj_label = "London"
    input_dict = tokenizer(text, return_tensors='pt')
    mask_index = input_dict['input_ids'][0].tolist().index(
        tokenizer.mask_token_id)
    labels = input_dict['input_ids'].clone()
    labels.fill_(-100)
    labels[0, mask_index] = tokenizer.convert_tokens_to_ids([obj_label])[0]
    outputs = bert(**input_dict, labels=labels)
    logits = outputs.logits
    loss = outputs.loss
    print(loss)
    print(logits.shape)
    # for module, name in parameters_tobe_pruned:
    #     remove_prune_reparametrization(module, name)
