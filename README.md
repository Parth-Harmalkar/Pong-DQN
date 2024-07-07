# PONG with Deep Q-Network (DQN)

This repository contains the implementation of the classic game Pong using the Pygame library, enhanced with a Deep Q-Network (DQN) for training an AI to play the game.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Files](#files)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [License](#license)

## Overview

This project demonstrates the use of Reinforcement Learning (RL) to train an AI agent to play Pong. The agent uses a Deep Q-Network (DQN) to learn optimal policies through interaction with the environment.

## Installation

### Prerequisites

- Python 3.6+
- Pygame
- TensorFlow
- NumPy

### Steps

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/pong-dqn.git
    cd pong-dqn
    ```

2. Create a virtual environment and activate it:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To start training the DQN agent, run:
```bash
python main.py
