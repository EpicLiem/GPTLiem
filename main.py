import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from transformers import GPT2Tokenizer, GPT2LMHeadModel
from gptliem.model import GPT
from gptliem.trainer import Trainer
from gptliem.utils import set_seed, setup_logging, Config
from gptliem.bpe import BPETokenizer

import os
import sys
import json


def loadgpt2():
    set_seed(3407)

    use_mingpt = True # use minGPT or huggingface/transformers model?
    model_type = 'gpt2'

    model = GPT.from_pretrained(model_type)

    model.eval()


    def generate(prompt='', num_samples=10, steps=20, do_sample=True):
        # tokenize the input prompt into integer input sequence
        if use_mingpt:
            tokenizer = BPETokenizer()
            if prompt == '':
                # to create unconditional samples...
                # manually create a tensor with only the special <|endoftext|> token
                # similar to what openai's code does here https://github.com/openai/gpt-2/blob/master/src/generate_unconditional_samples.py
                x = torch.tensor([[tokenizer.encoder.encoder['<|endoftext|>']]], dtype=torch.long)
            else:
                x = tokenizer(prompt)
        else:
            tokenizer = GPT2Tokenizer.from_pretrained(model_type)
            if prompt == '':
                # to create unconditional samples...
                # huggingface/transformers tokenizer special cases these strings
                prompt = '<|endoftext|>'
            encoded_input = tokenizer(prompt, return_tensors='pt')
            x = encoded_input['input_ids']

        # we'll process all desired num_samples in a batch, so expand out the batch dim
        x = x.expand(num_samples, -1)

        # forward the model `steps` times to get samples, in a batch
        y = model.generate(x, max_new_tokens=steps, do_sample=do_sample, top_k=40)

        for i in range(num_samples):
            out = tokenizer.decode(y[i].cpu().squeeze())
            print('-' * 80)
            print(out)




    while True:
        print("ready for input!")
        prompt = input("prompt:")
        steps = int(input("steps:"))
        samples = int(input("samples:"))
        generate(prompt=prompt, steps=steps, num_samples=samples)

def get_config():

    C = Config()

    # system
    C.system = Config()
    C.system.seed = 3407
    C.system.work_dir = './out/adder'

    # data
    C.data = AdditionDataset.get_default_config()

    # model
    C.model = GPT.defaultconfig()
    C.model.model_type = 'gpt-nano'

    # trainer
    C.trainer = Trainer.get_default_config()
    C.trainer.learning_rate = 5e-4 # the model we're using is so small that we can go a bit faster

    return C

# -----------------------------------------------------------------------------

class AdditionDataset(Dataset):
    """
    Creates n-digit addition problems. For example, if n=2, then an example
    addition problem would be to add 85 + 50 = 135. This problem would be
    represented as the following string for the GPT:
    "8550531"
    This is because:
    - we are discarding the + and =, which are not necessary. We just encode the digits
      of the input numbers concatenated together.
    - the result 135 is encoded backwards to make the addition easier to learn for the
      GPT model, because of how the addition algorithm works.
    As one more example, the problem 6 + 39 = 45 would be encoded as:
    "0639054"
    where you will notice that we are padding with zeros to make sure that we always
    produce strings of the exact same size: n + n + (n + 1). When n=2, this is 7.
    At test time, we will feed in an addition problem by giving the first 2n digits,
    and hoping that the GPT model completes the sequence with the next (n+1) digits
    correctly.
    """

    @staticmethod
    def get_default_config():
        C = Config()
        C.ndigit = 2
        return C

    def __init__(self, config, split):
        self.config = config
        self.split = split # train/test

        # split up all addition problems into either training data or test data
        ndigit = self.config.ndigit
        assert ndigit <= 3, "the lines below would be very memory inefficient, in future maybe refactor to support"
        num = (10**ndigit)**2 # total number of possible addition problems with ndigit numbers
        rng = torch.Generator()
        rng.manual_seed(1337)
        perm = torch.randperm(num, generator=rng)
        num_test = min(int(num*0.2), 500) # 20% of the whole dataset, or only up to 500
        self.ixes = perm[:num_test] if split == 'test' else perm[num_test:]

    def get_vocab_size(self):
        return 10 # digits 0..9

    def get_block_size(self):
        # a,b,a+b, and +1 due to potential carry overflow,
        # but then also -1 because very last digit doesn't ever plug back
        # as there is no explicit <EOS> token to predict, it is implied
        return 3*self.config.ndigit + 1 - 1

    def __len__(self):
        return self.ixes.nelement()

    def __getitem__(self, idx):
        ndigit = self.config.ndigit
        # given a problem index idx, first recover the associated a + b
        idx = self.ixes[idx].item()
        nd = 10**ndigit
        a = idx // nd
        b = idx %  nd
        # calculate the "label" of the addition problem a + b
        c = a + b
        # encode the digits of a, b, c into strings
        astr = f'%0{ndigit}d' % a
        bstr = f'%0{ndigit}d' % b
        cstr = (f'%0{ndigit+1}d' % c)[::-1] # reverse c to make addition easier
        render = astr + bstr + cstr
        dix = [int(s) for s in render] # convert each character to its token index
        # x will be input to GPT and y will be the associated expected outputs
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long) # predict the next token in the sequence
        y[:ndigit*2-1] = -1 # we will only train in the output locations. -1 will mask loss to zero
        return x, y

# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # get default config and overrides from the command line, if any
    config = get_config()
    config.merge_from_args(sys.argv[1:])
    print(config)
    setup_logging(config)
    set_seed(config.system.seed)

    # construct train and test datasets
    train_dataset = AdditionDataset(config.data, split='train')
    test_dataset  = AdditionDataset(config.data, split='test')

    # construct the model
    config.model.vocab_size = train_dataset.get_vocab_size()
    config.model.block_size = train_dataset.get_block_size()
    model = GPT(config.model)

