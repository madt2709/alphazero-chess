# Alphazero-chess

This is a repo to implement a simplified version of the model described in the Alphazero [paper](https://arxiv.org/pdf/1712.01815.pdf). The purpose was for me to understand the methods described in the paper more deeply. I have not attempted to recreate the training ran in the paper due to computing ressource limitations.

## Content

What you'll find in this repo is:

- `mcts.py` which has all the code required to run a Monte Carlo Tree Search.
- `nnet` folder which has all the code to create the neural network described. I have opted to only have 4 res blocks compared to the rather than the 19 or 39 quoted in the AlphaGo Zero [paper](https://www.nature.com/articles/nature24270.epdf?author_access_token=VJXbVjaSHxFoctQQ4p2k4tRgN0jAjWel9jnR3ZoTv0PVW4gB86EEpGqTRDtpIz-2rmo8-KG06gqVobU5NSCFeHILHcVFUeMsbvwS-lxjqQGg98faovwjxeTUgZAUMnRQ). This can be easily adjusted in the `settings.py`.
- `representations` folder which has all the encoder/decoders for a chess position, board and move set.
- `pipeline.py` the script to run the entire process of self-play + learning.
- `settings.py` a file which has a list of variables to easily change various model parameters + process etc...
- `tests` folder with all the tests written to help development. For the sake of time, only crucial part of the code have unit tests however test coverage should be extended if anyone plans to train this model for longer.

### How to start process

Clone the repo:

```bash
git clone git@github.com:madt2709/alphazero-chess.git
```

Create a venv and install the requirements:

```bash
pip install -r requirements.txt
```

Run the pipeline script:

```bash
python3 pipeline.py
```

You can follow the games on the terminal as they are played. Model parameters will be saved after every training as well as a loss_vs_epoch graph.

### What could be worked on

- Currently, there is no multi-processing functionality. However, this could be added fairly easily and would aid the training process.
- Extend the test coverage
- many other things...
