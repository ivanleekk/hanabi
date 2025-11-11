import random
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import pickle
import os

class Colour:
    colour: str = None
    
    def __init__(self, colour: str):
        self.colour = colour

    def __str__(self):
        return f"Colour: {self.colour}"

class Card(Colour):
    rank: int = None
    
    def __init__(self, colour: str, rank: int):
        super().__init__(colour)
        self.rank = rank

    def __str__(self):
        return f"Card: {self.colour} {self.rank}"

class Deck:
    
    def __init__(self):
        colours = ['Red', 'Blue', 'Green', 'Yellow', 'White']
        ranks = ((1, 3), (2, 2), (3, 2), (4, 2), (5, 1))
        self.cards = [Card(colour, rank) for colour in colours for rank, count in ranks for _ in range(count)]
    
    def __str__(self):
        return f"Deck with {len(self.cards)} cards"
    
    def shuffle(self) -> None:
        random.shuffle(self.cards)

    def draw(self) -> Card | None:
        return self.cards.pop() if self.cards else None
    
class Player:
    
    def __init__(self, name: str, game: 'Game'):
        self.name = name
        self.hand = {}
        self.knowledge = {}
        self.game = game

    def __str__(self):
        return f"Player: {self.name} with hand {[str(card) for card in self.hand.values()]} and knowledge {self.knowledge}"
    
    def draw_card(self, deck: Deck) -> None:
        card = deck.draw()
        if not card:
            return
        for i in range(4):
            if i not in self.hand:
                self.hand[i] = card
                break
    
    def show_hand(self) -> dict[int: Card]:
        return self.hand
    
    def discard_card(self, index: int) -> Card | None:
        current_card = self.remove_card_and_draw(index)
        self.game.discard_card(self, current_card)
        return 
    
    def receive_colour_hint(self, colour: str) -> None:
        for idx, card in self.hand.items():
            if card.colour == colour:
                current_knowledge = self.knowledge.get(idx, (None, None))
                self.knowledge[idx] = (colour, current_knowledge[1])

    def receive_rank_hint(self, rank: int) -> None:
        for idx, card in self.hand.items():
            if card.rank == rank:
                current_knowledge = self.knowledge.get(idx, (None, None))
                self.knowledge[idx] = (current_knowledge[0], rank)

    def play_card(self, index: int) -> Card | None:
        card: Card = self.remove_card_and_draw(index)
        return self.game.play_card(self, card)
    
    def remove_card_and_draw(self, index: int) -> Card | None:
        if index in self.hand:
            current_card = self.hand[index]
            self.hand.pop(index)
            if index in self.knowledge.keys():
                self.knowledge.pop(index)
            
            self.draw_card(self.game.deck)
            return current_card
        return None
    
    
class Game:
    
    def __init__(self, player_names: list[str]):
        self.players = [Player(name, self) for name in player_names]
        self.deck = Deck()
        self.deck.shuffle()
        self.lives = 3
        self.hints = 8
        self.played_cards = {}
        self.turn = 0
        self.actions = [self.hint_colour, self.hint_rank, self.discard_card, self.play_card]
        self.final_round = len(self.players)
        self.deal_cards(4)

    def __str__(self):
        return f"Game with players {[str(player) for player in self.players]} and {str(self.deck)}, Lives: {self.lives}, Hints: {self.hints}, Played Cards: {self.played_cards}"
    
    def deal_cards(self, cards_per_player: int) -> None:
        for _ in range(cards_per_player):
            for player in self.players:
                player.draw_card(self.deck)
    
    def hint_colour(self, player: Player, colour: str) -> None:
        if self.hints <= 0:
            print("No hints left!")
            return
        player.receive_colour_hint(colour)
        self.hints -= 1
        if not self.is_game_over():
            self.next_turn()
        
    def hint_rank(self, player: Player, rank: int) -> None:
        if self.hints <= 0:
            print("No hints left!")
            return
        player.receive_rank_hint(rank)
        self.hints -= 1
        if not self.is_game_over():
            self.next_turn()

    def discard_card(self, player: Player, card: Card) -> None:
        if card:
            print(f"{player.name} discarded {card}")
            self.hints = min(self.hints + 1, 8)
        else:
            print("Invalid discard index")
        
        if not self.is_game_over():
            self.next_turn()

    def play_card(self, player: Player, card: Card) -> None:
        if card:
            current_rank = self.played_cards.get(card.colour, 0)
            if card.rank == current_rank + 1:
                self.played_cards[card.colour] = card.rank
                print(f"{player.name} played {card} successfully!")
            else:
                self.lives -= 1
                print(f"{player.name} played {card} incorrectly! Lives left: {self.lives}")
        else:
            print("Invalid play index")
        
        if not self.is_game_over():
            self.next_turn()

    
    def next_turn(self) -> None:
        if len(self.deck.cards) == 0:
            self.final_round -= 1
        self.turn = (self.turn + 1) % len(self.players)

    def get_points(self) -> int:
        return sum(self.played_cards.values())
    
    def is_game_over(self) -> bool:
        return self.lives <= 0 or self.final_round < 0
    
    def get_state(self, player_idx: int) -> Dict[str, Any]:
        """
        Get the observable state for a specific player.
        This includes everything the player can see except their own cards.
        
        Returns a dictionary with:
        - current_player: index of whose turn it is
        - my_knowledge: what hints I've received about my cards
        - other_players_hands: visible cards of all other players
        - played_cards: current state of played cards for each colour
        - discard_pile: all discarded cards
        - deck_size: number of cards remaining in deck
        - hints: number of hints available
        - lives: number of lives remaining
        - legal_actions: list of legal action indices
        """
        player = self.players[player_idx]
        
        # Get knowledge about own hand
        my_knowledge = {}
        for idx in range(4):
            if idx in player.knowledge:
                my_knowledge[idx] = player.knowledge[idx]
            else:
                my_knowledge[idx] = (None, None)
        
        # Get other players' visible hands
        other_players_hands = {}
        for i, p in enumerate(self.players):
            if i != player_idx:
                other_players_hands[i] = {
                    'name': p.name,
                    'cards': [(idx, card.colour, card.rank) for idx, card in p.hand.items()]
                }
        
        # Get discard pile
        discard_pile = []
        # Note: We need to track discarded cards. Adding this to the Game class.
        
        state = {
            'current_player': self.turn,
            'my_index': player_idx,
            'my_knowledge': my_knowledge,
            'my_hand_size': len(player.hand),
            'other_players_hands': other_players_hands,
            'played_cards': dict(self.played_cards),
            'deck_size': len(self.deck.cards),
            'hints': self.hints,
            'lives': self.lives,
            'turn_number': self.turn,
            'final_round_countdown': self.final_round if len(self.deck.cards) == 0 else -1
        }
        
        return state
    
    def get_legal_actions(self, player_idx: int) -> List[Tuple[str, Any]]:
        """
        Get list of legal actions for the current player.
        Returns list of tuples: (action_type, action_params)
        
        Action types:
        - ('play', card_index): play card at index
        - ('discard', card_index): discard card at index
        - ('hint_colour', (target_player_idx, colour)): give colour hint
        - ('hint_rank', (target_player_idx, rank)): give rank hint
        """
        player = self.players[player_idx]
        actions = []
        
        # Play and discard actions for each card in hand
        for idx in player.hand.keys():
            actions.append(('play', idx))
            actions.append(('discard', idx))
        
        # Hint actions (only if hints available)
        if self.hints > 0:
            for i, other_player in enumerate(self.players):
                if i != player_idx:
                    # Colour hints
                    for colour in ['Red', 'Blue', 'Green', 'Yellow', 'White']:
                        # Only allow hints that actually match at least one card
                        if any(card.colour == colour for card in other_player.hand.values()):
                            actions.append(('hint_colour', (i, colour)))
                    
                    # Rank hints
                    for rank in range(1, 6):
                        if any(card.rank == rank for card in other_player.hand.values()):
                            actions.append(('hint_rank', (i, rank)))
        
        return actions
    
    def step(self, action: Tuple[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Execute an action and return the new state, reward, done flag, and info.
        This follows the OpenAI Gym interface.
        
        Args:
            action: Tuple of (action_type, action_params)
        
        Returns:
            state: New state for the current player
            reward: Immediate reward
            done: Whether the game is over
            info: Additional information
        """
        action_type, params = action
        current_player_idx = self.turn
        current_player = self.players[current_player_idx]
        
        prev_score = self.get_points()
        prev_lives = self.lives
        
        # Execute the action
        if action_type == 'play':
            card_idx = params
            current_player.play_card(card_idx)
        
        elif action_type == 'discard':
            card_idx = params
            current_player.discard_card(card_idx)
        
        elif action_type == 'hint_colour':
            target_player_idx, colour = params
            target_player = self.players[target_player_idx]
            self.hint_colour(target_player, colour)
        
        elif action_type == 'hint_rank':
            target_player_idx, rank = params
            target_player = self.players[target_player_idx]
            self.hint_rank(target_player, rank)
        
        # Calculate reward
        reward = 0.0
        new_score = self.get_points()
        
        # Reward for increasing score
        reward += (new_score - prev_score) * 1.0
        
        # Penalty for losing a life
        if self.lives < prev_lives:
            reward -= 1.0
        
        # Get new state
        next_player_idx = self.turn
        state = self.get_state(next_player_idx)
        
        done = self.is_game_over()
        
        # Bonus for winning
        if done and self.lives > 0:
            reward += new_score * 0.1
        
        info = {
            'score': new_score,
            'lives': self.lives,
            'hints': self.hints,
            'action_taken': action
        }
        
        return state, reward, done, info
    
    def reset(self) -> Dict[str, Any]:
        """
        Reset the game to initial state.
        Returns the initial state for the first player.
        """
        self.__init__([p.name for p in self.players])
        return self.get_state(0)
    
    def render(self) -> None:
        """
        Print the current game state in a human-readable format.
        """
        print("\n" + "="*60)
        print(f"Turn: Player {self.turn} ({self.players[self.turn].name})")
        print(f"Lives: {self.lives} | Hints: {self.hints} | Deck: {len(self.deck.cards)} cards")
        print(f"Score: {self.get_points()}")
        print("\nPlayed cards:")
        for colour in ['Red', 'Blue', 'Green', 'Yellow', 'White']:
            rank = self.played_cards.get(colour, 0)
            print(f"  {colour}: {rank}")
        print("\nPlayers:")
        for i, player in enumerate(self.players):
            print(f"  {i}. {player.name}:")
            print(f"     Hand: {[(idx, card.colour, card.rank) for idx, card in player.hand.items()]}")
            if i == self.turn:
                print(f"     Knowledge: {player.knowledge}")
        print("="*60 + "\n")
    

class QLearningAgent:
    """
    Q-Learning agent for Hanabi.
    Uses epsilon-greedy policy and tabular Q-learning.
    """
    
    def __init__(self, 
                 learning_rate: float = 0.1,
                 discount_factor: float = 0.95,
                 epsilon: float = 1.0,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01):
        """
        Initialize Q-Learning agent.
        
        Args:
            learning_rate: Alpha, how much to update Q-values
            discount_factor: Gamma, importance of future rewards
            epsilon: Initial exploration rate
            epsilon_decay: How much to decay epsilon after each episode
            epsilon_min: Minimum epsilon value
        """
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Q-table: state -> action -> value
        self.q_table: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        
        # Track statistics
        self.episodes_trained = 0
    
    def state_to_key(self, state: Dict[str, Any]) -> str:
        """
        Convert state dictionary to a hashable string key.
        We simplify the state to make it tractable for tabular Q-learning.
        """
        # Extract key features
        key_parts = [
            f"hints:{state['hints']}",
            f"lives:{state['lives']}",
            f"deck:{state['deck_size']}",
            f"hand:{state['my_hand_size']}",
        ]
        
        # Add played cards state
        played = state['played_cards']
        for color in ['Red', 'Blue', 'Green', 'Yellow', 'White']:
            key_parts.append(f"{color[0]}:{played.get(color, 0)}")
        
        # Add my knowledge (simplified)
        knowledge_str = []
        for idx in range(4):
            color, rank = state['my_knowledge'].get(idx, (None, None))
            c = color[0] if color else 'X'
            r = rank if rank else 'X'
            knowledge_str.append(f"{c}{r}")
        key_parts.append(f"know:{'_'.join(knowledge_str)}")
        
        return "|".join(key_parts)
    
    def action_to_key(self, action: Tuple[str, Any]) -> str:
        """Convert action tuple to a hashable string key."""
        action_type, params = action
        if action_type in ['play', 'discard']:
            return f"{action_type}_{params}"
        elif action_type == 'hint_colour':
            player_idx, colour = params
            return f"hint_c_p{player_idx}_{colour}"
        elif action_type == 'hint_rank':
            player_idx, rank = params
            return f"hint_r_p{player_idx}_{rank}"
        return str(action)
    
    def get_q_value(self, state: Dict[str, Any], action: Tuple[str, Any]) -> float:
        """Get Q-value for state-action pair."""
        state_key = self.state_to_key(state)
        action_key = self.action_to_key(action)
        return self.q_table[state_key][action_key]
    
    def choose_action(self, state: Dict[str, Any], legal_actions: List[Tuple[str, Any]], training: bool = True) -> Tuple[str, Any]:
        """
        Choose an action using epsilon-greedy policy.
        
        Args:
            state: Current game state
            legal_actions: List of legal actions
            training: If True, use epsilon-greedy; if False, use greedy
        
        Returns:
            Chosen action
        """
        if not legal_actions:
            raise ValueError("No legal actions available")
        
        # Exploration: random action
        if training and random.random() < self.epsilon:
            return random.choice(legal_actions)
        
        # Exploitation: best known action
        state_key = self.state_to_key(state)
        q_values = []
        for action in legal_actions:
            action_key = self.action_to_key(action)
            q_values.append(self.q_table[state_key][action_key])
        
        # Choose action with highest Q-value (random tie-breaking)
        max_q = max(q_values)
        best_actions = [action for action, q in zip(legal_actions, q_values) if q == max_q]
        return random.choice(best_actions)
    
    def update(self, state: Dict[str, Any], action: Tuple[str, Any], 
               reward: float, next_state: Dict[str, Any], 
               next_legal_actions: List[Tuple[str, Any]], done: bool) -> None:
        """
        Update Q-value using Q-learning update rule.
        
        Q(s,a) = Q(s,a) + α * [r + γ * max_a' Q(s',a') - Q(s,a)]
        """
        state_key = self.state_to_key(state)
        action_key = self.action_to_key(action)
        
        current_q = self.q_table[state_key][action_key]
        
        if done:
            # No future rewards if episode is done
            target = reward
        else:
            # Find max Q-value for next state
            next_state_key = self.state_to_key(next_state)
            if next_legal_actions:
                max_next_q = max(
                    self.q_table[next_state_key][self.action_to_key(a)] 
                    for a in next_legal_actions
                )
            else:
                max_next_q = 0.0
            
            target = reward + self.discount_factor * max_next_q
        
        # Q-learning update
        self.q_table[state_key][action_key] = current_q + self.learning_rate * (target - current_q)
    
    def decay_epsilon(self) -> None:
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.episodes_trained += 1
    
    def save(self, filepath: str) -> None:
        """Save Q-table to file."""
        data = {
            'q_table': dict(self.q_table),
            'epsilon': self.epsilon,
            'episodes_trained': self.episodes_trained
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"Agent saved to {filepath}")
    
    def load(self, filepath: str) -> None:
        """Load Q-table from file."""
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            self.q_table = defaultdict(lambda: defaultdict(float), data['q_table'])
            self.epsilon = data.get('epsilon', self.epsilon)
            self.episodes_trained = data.get('episodes_trained', 0)
            print(f"Agent loaded from {filepath}")
        else:
            print(f"No saved agent found at {filepath}")


def train_q_learning(num_episodes: int = 1000, save_path: str = "q_agent.pkl"):
    """
    Train a Q-learning agent to play Hanabi.
    
    Args:
        num_episodes: Number of episodes to train
        save_path: Path to save the trained agent
    """
    players = ["Alice", "Bob", "Charlie", "Diana"]
    game = Game(players)
    agent = QLearningAgent(
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01
    )
    
    # Try to load existing agent
    if os.path.exists(save_path):
        agent.load(save_path)
    
    scores = []
    recent_scores = []
    
    print(f"Training Q-Learning Agent for {num_episodes} episodes...")
    print("="*60)
    
    for episode in range(num_episodes):
        # Reset game
        state = game.reset()
        done = False
        episode_reward = 0
        
        # Store previous state and action for multi-agent learning
        prev_states = {}
        prev_actions = {}
        
        while not done:
            current_player = game.turn
            
            # Get legal actions
            legal_actions = game.get_legal_actions(current_player)
            if not legal_actions:
                break
            
            # Agent chooses action
            action = agent.choose_action(state, legal_actions, training=True)
            
            # Store state and action
            prev_states[current_player] = state
            prev_actions[current_player] = action
            
            # Take action
            next_state, reward, done, info = game.step(action)
            episode_reward += reward
            
            # Update Q-values for all players (cooperative learning)
            for player_idx in range(len(players)):
                if player_idx in prev_states and player_idx in prev_actions:
                    next_legal = game.get_legal_actions(game.turn) if not done else []
                    agent.update(
                        prev_states[player_idx],
                        prev_actions[player_idx],
                        reward,  # All players get same reward (cooperative)
                        next_state,
                        next_legal,
                        done
                    )
            
            state = next_state
        
        # Track scores
        final_score = game.get_points()
        scores.append(final_score)
        recent_scores.append(final_score)
        if len(recent_scores) > 100:
            recent_scores.pop(0)
        
        # Decay epsilon
        agent.decay_epsilon()
        
        # Print progress
        if (episode + 1) % 100 == 0:
            avg_score = sum(recent_scores) / len(recent_scores)
            print(f"Episode {episode + 1}/{num_episodes}")
            print(f"  Avg Score (last 100): {avg_score:.2f}")
            print(f"  Epsilon: {agent.epsilon:.3f}")
            print(f"  Q-table size: {len(agent.q_table)} states")
            print()
    
    # Save agent
    agent.save(save_path)
    
    print("="*60)
    print("Training completed!")
    print(f"Final average score (last 100): {sum(recent_scores)/len(recent_scores):.2f}")
    print(f"Best score: {max(scores)}")
    print(f"Q-table size: {len(agent.q_table)} states")
    
    return agent, scores


def demo_q_agent(agent_path: str = "q_agent.pkl", num_games: int = 5):
    """
    Demonstrate trained Q-learning agent playing Hanabi.
    
    Args:
        agent_path: Path to saved agent
        num_games: Number of games to play
    """
    players = ["Alice", "Bob", "Charlie", "Diana"]
    game = Game(players)
    agent = QLearningAgent()
    
    # Load trained agent
    agent.load(agent_path)
    
    print(f"\n{'='*60}")
    print(f"Demonstrating Trained Q-Learning Agent")
    print(f"{'='*60}\n")
    
    scores = []
    
    for game_num in range(num_games):
        print(f"\n=== Game {game_num + 1}/{num_games} ===\n")
        
        state = game.reset()
        done = False
        step_count = 0
        
        game.render()
        
        while not done and step_count < 100:  # Max 100 steps per game
            current_player = game.turn
            
            # Get legal actions
            legal_actions = game.get_legal_actions(current_player)
            if not legal_actions:
                break
            
            # Agent chooses action (greedy, no exploration)
            action = agent.choose_action(state, legal_actions, training=False)
            
            print(f"Player {current_player} ({game.players[current_player].name}) takes action: {action}")
            
            # Take action
            state, reward, done, info = game.step(action)
            step_count += 1
            
            if reward != 0:
                print(f"  → Reward: {reward:.2f}, Score: {info['score']}, Lives: {info['lives']}")
            
            if done:
                print(f"\n{'='*40}")
                print(f"Game Over! Final Score: {game.get_points()}")
                print(f"{'='*40}\n")
        
        scores.append(game.get_points())
    
    print(f"\n{'='*60}")
    print(f"Results:")
    print(f"  Games played: {num_games}")
    print(f"  Average score: {sum(scores)/len(scores):.2f}")
    print(f"  Best score: {max(scores)}")
    print(f"  Worst score: {min(scores)}")
    print(f"{'='*60}\n")


def main():
    print("Hanabi Q-Learning Agent")
    print("="*60)
    print("\nOptions:")
    print("1. Train Q-Learning agent")
    print("2. Demo trained agent")
    print("3. Quick training demo (100 episodes)")
    
    choice = input("\nEnter choice (1-3, or press Enter for quick demo): ").strip()
    
    if choice == "1":
        episodes = int(input("Number of episodes to train (default 1000): ") or "1000")
        train_q_learning(num_episodes=episodes)
    elif choice == "2":
        num_games = int(input("Number of games to demo (default 5): ") or "5")
        demo_q_agent(num_games=num_games)
    else:
        # Quick demo - train and show results
        print("\nRunning quick training demo (10000 episodes)...")
        agent, scores = train_q_learning(num_episodes=10000, save_path="q_agent_demo.pkl")
        
        print("\nNow demonstrating the trained agent...")
        demo_q_agent(agent_path="q_agent_demo.pkl", num_games=3)
    

if __name__ == "__main__":
    main()
