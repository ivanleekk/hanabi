import random

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
    cards: list[Card] = []
    
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
    name: str = None
    hand: dict[int: Card]
    knowledge: dict[int: (str, int)]
    game: Game
    
    def __init__(self, name: str, game: Game):
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
    players: list[Player] = []
    deck: Deck = None
    lives: int = 3
    hints: int = 8
    played_cards: dict[str, int] = {}
    turn: int = 0
    actions: list[function]
    final_round: int = 0
    
    def __init__(self, player_names: list[str]):
        self.players = [Player(name, self) for name in player_names]
        self.deck = Deck()
        self.deck.shuffle()
        self.actions = [self.hint_colour, self.hint_rank, self.discard_card, self.play_card]
        self.deal_cards(4)
        self.final_round = len(self.players)

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
        self.next_turn()
        
    def hint_rank(self, player: Player, rank: int) -> None:
        if self.hints <= 0:
            print("No hints left!")
            return
        player.receive_rank_hint(rank)
        self.hints -= 1
        self.next_turn()

    def discard_card(self, player: Player, card: Card) -> None:
        if card:
            print(f"{player.name} discarded {card}")
            self.hints = min(self.hints + 1, 8)
        else:
            print("Invalid discard index")
        
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
        self.next_turn()

    
    def next_turn(self) -> None:
        if len(self.deck.cards) == 0:
            self.final_round -= 1
        self.turn = (self.turn + 1) % len(self.players)

    def get_points(self) -> int:
        return sum(self.played_cards.values())
    
    def is_game_over(self) -> bool:
        return self.lives <= 0 or self.final_round < 0
    
def main():
    print("Creating a deck of cards...")
    players = ["Alice", "Bob", "Charlie", "Diana"]
    game = Game(players)
    while not game.is_game_over():
        current_player = game.players[game.turn]
        print(f"\nIt's {current_player.name}'s turn.")
        print("Current game state:", game)
        print(current_player)
        # print(f"What {current_player.name} can see:")
        # for player in game.players:
        #     if player != current_player:
        #         print(player)

        # if current player has a playable card based on knowledge, play it
        playable_indices = []
        for idx, (colour, rank) in current_player.knowledge.items():
            if colour and rank:
                current_rank = game.played_cards.get(colour, 0)
                if rank == current_rank + 1:
                    playable_indices.append(idx)
        print(f"Playable indices based on knowledge: {playable_indices}")
        if playable_indices:
            index_to_play = playable_indices[0]
            current_player.play_card(index_to_play)
            continue

        # if other players have playable cards based on knowledge, give hint
        if game.hints > 0:
            for player in game.players:
                hint_given = False
                if player != current_player:
                    for idx, card in player.show_hand().items():
                        # print(f"Checking {player.name}'s {card}")
                        
                        current_rank = game.played_cards.get(card.colour, 0)
                        # print(f"Current rank for colour {card.colour} is {current_rank}")
                        if card.rank == current_rank + 1:
                            # check other players' knowledge to avoid redundant hints
                            colour, rank = player.knowledge.get(idx, (None, None))
                            if not rank:
                                game.hint_rank(player, card.rank)
                                hint_given = True
                                print(f"Gave rank hint {card.rank} to {player.name}")
                                break
                            elif not colour:
                                game.hint_colour(player, card.colour)
                                hint_given = True
                                print(f"Gave colour hint {card.colour} to {player.name}")
                                break
                    if hint_given:
                        break
            if hint_given:
                continue
                            
                        
                

        # discard a random card
        index = random.choice(list(current_player.hand.keys()))
        current_player.discard_card(index)
    print(game.played_cards, game.get_points())

if __name__ == "__main__":
    main()
