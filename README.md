# DivineNN

DivineNN is a deep-learning chess engine written in Rust, partially compliant with the UCI protocol. DivineNN uses the Monte Carlo Tree Search (MCTS) algorithm to enhance the neural network policy using the neural network's value output.

## Training

Training code is not available at this time. A decently-strong network (~2150 elo) is included in this repo.

## Compiling

DivineNN is written in Rust. As such, the Rust compiler is required to compile DivineNN. See:

https://rustup.rs

Compiling & running DivineNN is as simple as:
```bash
RUSTFLAGS='-C target-cpu=native' cargo run --release
```

## Strength

DivineNN can theoretically perform at the level of Stockfish with a good network (such as one of the lc0 nets). The network included in this repo is not that strong, only achieving 2150 elo on the Lichess bot list.

### Resources 

https://github.com/coreylowman/synthesis

https://github.com/geochri/AlphaZero_Chess
