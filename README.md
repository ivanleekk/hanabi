# Hanabi Reinforcement Learning Environment

This is a Hanabi card game implementation designed as a reinforcement learning environment with a **Q-Learning agent** implementation. The game follows the OpenAI Gym interface for easy integration with RL algorithms.

## Game Overview

Hanabi is a cooperative card game where players work together to play cards in the correct order. Players can see everyone's cards except their own, and must use hints to coordinate plays.

## Q-Learning Agent

The project includes a fully implemented **tabular Q-learning agent** that learns to play Hanabi through trial and error.

### Features

- **Epsilon-greedy exploration**: Balances exploration vs exploitation
- **Cooperative learning**: All players share the same Q-table and learn together
- **State abstraction**: Simplifies state representation for tractable learning
- **Persistent storage**: Save and load trained agents
- **Training statistics**: Track performance over episodes

### Quick Start

Run the Q-learning agent:

```bash
python main.py
```

Then choose:
1. **Train Q-Learning agent** - Train for custom number of episodes
2. **Demo trained agent** - Watch a trained agent play
3. **Quick training demo** - Train for 100 episodes and demo (recommended for first try)

### Training the Agent

```python
from main import train_q_learning

# Train for 1000 episodes
agent, scores = train_q_learning(num_episodes=1000, save_path="q_agent.pkl")
```

The agent will:
- Start with high exploration (epsilon = 1.0)
- Gradually decrease exploration as it learns
- Save progress every 100 episodes
- Display average scores and Q-table size

### Using a Trained Agent

```python
from main import demo_q_agent

# Play 5 games with the trained agent
demo_q_agent(agent_path="q_agent.pkl", num_games=5)
```

### Q-Learning Parameters

- **learning_rate** (α): 0.1 - How much to update Q-values
- **discount_factor** (γ): 0.95 - Importance of future rewards
- **epsilon**: 1.0 → 0.01 - Exploration rate (decays over time)
- **epsilon_decay**: 0.995 - Rate of exploration decrease

### How It Works

1. **State Representation**: The agent converts game state to a hash key including:
   - Hints and lives remaining
   - Deck size and hand size
   - Played cards for each color
   - Knowledge hints received

2. **Action Selection**: 
   - During training: epsilon-greedy (explore vs exploit)
   - During demo: greedy (always pick best known action)

3. **Q-Value Updates**: Uses the Q-learning formula:
   ```
   Q(s,a) = Q(s,a) + α * [r + γ * max_a' Q(s',a') - Q(s,a)]
   ```

4. **Cooperative Learning**: All players update the same Q-table with the same reward signal

## Reinforcement Learning Interface

### State Representation

The `get_state(player_idx)` method returns a dictionary containing all observable information for a specific player:

```python
{
    'current_player': int,              # Whose turn it is
    'my_index': int,                    # This player's index
    'my_knowledge': dict,               # Hints received about own cards
    'my_hand_size': int,                # Number of cards in hand
    'other_players_hands': dict,        # Visible cards of other players
    'played_cards': dict,               # Current state of played cards
    'deck_size': int,                   # Cards remaining in deck
    'hints': int,                       # Hint tokens available
    'lives': int,                       # Lives remaining
    'turn_number': int,                 # Current turn index
    'final_round_countdown': int        # Turns left after deck empty (-1 if deck not empty)
}
```

### Action Space

Actions are represented as tuples `(action_type, params)`:

- **Play card**: `('play', card_index)`
- **Discard card**: `('discard', card_index)`
- **Give color hint**: `('hint_colour', (target_player_idx, colour))`
- **Give rank hint**: `('hint_rank', (target_player_idx, rank))`

Use `get_legal_actions(player_idx)` to get all valid actions for a player.

### Step Function

The `step(action)` method executes an action and returns:

```python
state, reward, done, info = game.step(action)
```

- **state**: New state dictionary for the next player
- **reward**: Float reward value
  - +1.0 for each point scored
  - -1.0 for losing a life
  - +0.1 * final_score bonus if game is won
- **done**: Boolean indicating if game is over
- **info**: Dictionary with additional information

### Reset Function

```python
initial_state = game.reset()
```

Resets the game to initial state and returns the state for the first player.

## Usage Example

```python
from main import Game

# Create game
players = ["Alice", "Bob", "Charlie", "Diana"]
game = Game(players)

# Get initial state
state = game.get_state(game.turn)

# Game loop
done = False
while not done:
    # Get legal actions
    legal_actions = game.get_legal_actions(game.turn)
    
    # Choose action (replace with your RL agent)
    action = random.choice(legal_actions)
    
    # Take action
    state, reward, done, info = game.step(action)
    
    print(f"Score: {info['score']}, Lives: {info['lives']}")

# Reset for next episode
state = game.reset()
```

## Features

- **Full state observation**: All observable game information is included
- **Legal action masking**: Only valid actions are returned
- **Cooperative rewards**: All players share the same reward signal
- **Turn-based**: State is returned for the next player after each action
- **Visualization**: Use `game.render()` to print human-readable game state

## RL Training Tips

1. **Action masking**: Always use `get_legal_actions()` to avoid invalid moves
2. **Multi-agent**: This is a cooperative multi-agent problem - consider shared learning
3. **Partial observability**: Players cannot see their own cards directly
4. **Credit assignment**: Delayed rewards make this challenging
5. **Communication**: Learning effective hinting strategies is key

## Requirements

- Python 3.7+
- No external dependencies for the base game
- For RL training, consider using: PyTorch, TensorFlow, or Stable-Baselines3

## Performance

With 100 episodes of training:
- Average score: ~3-4 points
- Learning is visible but limited

With 1000+ episodes:
- Average score: ~5-8 points
- Better coordination and hint usage
- Q-table size: 500-1000 unique states

The tabular Q-learning approach is limited by state space complexity. For better performance, consider:
- Deep Q-Networks (DQN)
- Policy Gradient methods (PPO, A3C)
- Multi-agent RL algorithms (QMIX, COMA)

## Example Training Output

```
Training Q-Learning Agent for 100 episodes...
============================================================
Episode 100/100
  Avg Score (last 100): 3.80
  Epsilon: 0.606
  Q-table size: 78 states

Training completed!
Final average score (last 100): 3.80
Best score: 7
```
