import pytest
from training import *
from transformer import *

# Test cases to use as the basis for creating queues.
list_init_cases = [
    [],
    [1],
    [1, 2, 3],
    [1.0, 2.0, 3.0],
    ["a", "b", "c"],
    [True, False, True],
    [1, True, "c"],
]


# Test cases to use for appending new items to a queue.
new_item_cases = [
    42,
    "Hello",
    3.14,
    True,
]

def test_vocab_load():
    """
    Test that vocabulary is properly loaded from a CSV.
    """
    dataset = AdventureDataset("data/test.csv")
    itos = {0: "<PAD>",
            1: "<SOS>",
            2: "<EOS>",
            3: "<UNK>",
            4: "one",
            5: "two",
            6: "three",
            7: "four",
            8: "five",
            9: "six",
            10: "seven",
            11: "eight",
            12: "nine",
            13: "ten",
            14: "uno",
            15: "due",
            16: "tre",
            17: "quattro",
            18: "cinque",
            19: "sei",
            20: "sette",
            21: "otto",
            22: "nove",
            23: "dieci"}
    stoi = {"<PAD>": 0,
            "<SOS>": 1,
            "<EOS>": 2,
            "<UNK>": 3,
            "one": 4,
            "two": 5,
            "three": 6,
            "four": 7,
            "five": 8,
            "six": 9,
            "seven": 10,
            "eight": 11,
            "nine": 12,
            "ten": 13,
            "uno": 14,
            "due": 15,
            "tre": 16,
            "quattro": 17,
            "cinque": 18,
            "sei": 19,
            "sette": 20,
            "otto": 21,
            "nove": 22,
            "dieci": 23}
    assert dataset.vocab.itos == itos
    assert dataset.vocab.stoi == stoi


def test_simple_model():
    """
    Test that model can properly be created and trained.

    The model is trained on a simple copy task which should output the same
    sentence that it was given. In this case it should copy the sentence 1,
    2, 3, 4, 5, 6, 7, 8, 9, 10.

    NOTE; This test will not pass every time due to a decreased training time.
    However, if the output of the model is not relatively similar to the
    input tensor then the transformer and training python files should be
    reviewed.
    """
    # Train the simple copy task.
    V = 11
    criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
    model = Transformer.make_model(V, V, N=2, d_model=512, d_ff=1024)
    model_opt = NoamOpt(model.src_embed[0].d_model, 1, 400,
                        torch.optim.Adam(model.parameters(),
                                         lr=0,
                                         betas=(0.9, 0.98),
                                         eps=1e-9))
    for epoch in range(15):
        model.train()
        Training.run_epoch(Training.data_gen(V, 30, 20), model,
                           SimpleLossCompute(model.generator,
                                             criterion,
                                             model_opt))
        model.eval()
        Training.run_epoch(Training.data_gen(V, 30, 5), model,
                           SimpleLossCompute(model.generator,
                                             criterion,
                                             None))

    # Run and decode model copy
    model.eval()
    src = Variable(torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]))
    src_mask = Variable(torch.ones(1, 1, 10))
    output = Transformer.greedy_decode(model,
                                       src,
                                       src_mask,
                                       max_len=10,
                                       start_symbol=1)
    assert torch.equal(output, torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]))
