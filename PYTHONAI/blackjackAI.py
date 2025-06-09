import random
import numpy as np

# Define card values according to Hi-Lo system
card_values = {
    '2': 1, '3': 1, '4': 1, '5': 1, '6': 1,
    '7': 0, '8': 0, '9': 0,
    '10': -1, 'J': -1, 'Q': -1, 'K': -1, 'A': -1
}

# Initialize deck(s)
num_decks = 6  # Number of decks in the shoe
shoe = list(card_values.keys()) * 4 * num_decks  # Each deck has 4 suits
random.shuffle(shoe)
running_count = 0
# Q-learning parameters
learning_rate = 0.1
discount_factor = 0.95
exploration_rate = 1.0
max_exploration_rate = 1.0
min_exploration_rate = 0.01
exploration_decay_rate = 0.001

# Initialize Q-table
action_space = [0, 1]  # 0: Stand, 1: Hit
state_space = (32, 11, 2)  # Number of features in the state representation
num_actions = len(action_space)
Q_table = np.zeros(state_space + (num_actions,))


# Function to deal a card from the shoe
def deal_card():
    global shoe, running_count
    if len(shoe) == 0:  # Check if shoe is empty
        # Replenish and shuffle the shoe
        shoe = list(card_values.keys()) * 4 * num_decks
        random.shuffle(shoe)
        running_count = 0

    card = shoe.pop()

    if card in card_values:
            running_count += card_values[card]

    return card

# Function to calculate hand value
def calculate_hand_value(hand):
    value = 0
    num_aces = 0  # Count the number of aces separately
    for card in hand:
        if card.isdigit():
            value += int(card)
        elif card in ['J', 'Q', 'K']:
            value += 10
        elif card == 'A':
            num_aces += 1
            value += 11  # Initially count ace as 11
        else:
            raise ValueError(f"Invalid card: {card}")

    # Adjust the value for aces
    while value > 21 and num_aces:
        value -= 10  # Change the value of ace from 11 to 1
        num_aces -= 1

    return value

def encode_state(player_hand, dealer_card, running_count):
    """
    Encode state of the game.
    Currently, the state representation includes the player's hand value, the dealer's visible card value,
    the true count, and the number of decks remaining.
    """
    player_value = calculate_hand_value(player_hand)
    dealer_value = card_values[dealer_card]
    # Ensure all state values fit within the Q-table dimensions
    player_value = min(max(player_value, 0), 31)
    dealer_value = min(max(dealer_value, 0), 10)
    running_count = min(max(running_count, 0), 1)
    state = (player_value, dealer_value, running_count)
    return state


# Training loop
num_episodes = 150000000
for episode in range(num_episodes):
    # Initialize game variables
    player_hand = [deal_card(), deal_card()]
    dealer_hand = [deal_card(), deal_card()]
    round_counter = 0
    game_over = False

    reward = 0
    while not game_over:
        round_counter += 1

        # Encode state
        state = encode_state(player_hand, dealer_hand[0], running_count)

        # Epsilon-greedy action selection
        exploration_rate_threshold = random.uniform(0, 1)
        if exploration_rate_threshold > exploration_rate:
            action = np.argmax(Q_table[state])
        else:
            action = random.choice(action_space)

        # Perform action
        
        if action == 1:
            player_hand.append(deal_card())
            if calculate_hand_value(player_hand) > 21:
                reward = -1  # Player busts
                game_over = True
            elif calculate_hand_value(player_hand) == 21:
                reward = 1
                game_over = True
            else:
                reward = 0  # No immediate reward if player hits
        else:
            # Player stands
            # Dealer's turn
            while calculate_hand_value(dealer_hand) < 17:
                dealer_hand.append(deal_card())
            player_value = calculate_hand_value(player_hand)
            dealer_value = calculate_hand_value(dealer_hand)
            if dealer_value > 21 or player_value > dealer_value:
                reward = 1
            elif player_value == dealer_value:
                reward = 0
            else:
                reward = -1

        # Update Q-table
        #print("State:", state)
        #print("Action:", action)
        #print("Q-table shape:", Q_table.shape)
        new_state = encode_state(player_hand, dealer_hand[0], running_count)
        #print("New state:", new_state)
        max_future_q = np.max(Q_table[new_state])
        #print("Max future Q-value:", max_future_q)

        current_q = Q_table[state + (action,)]
        new_q = (1 - learning_rate) * current_q + learning_rate * (reward + discount_factor * max_future_q)
        Q_table[state + (action,)] = new_q

        game_over = True

    # Decay exploration rate
    exploration_rate = min_exploration_rate + \
                        (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate * episode)

print("Training completed.")

# Function to test AI's performance
def test_ai_performance(Q_table, num_games):
    wins = 0
    pushes = 0
    losses = 0
    for game in range(1, num_games + 1):
        print(f"Game {game}:")
        # Initialize game variables
        player_hand = [deal_card(), deal_card()]
        dealer_hand = [deal_card(), deal_card()]
        print(f"AI's cards: {player_hand}")
        print(f"Dealer's cards: {dealer_hand[0]}")
        print("running_count: ", running_count)
        game_over = False

        # Main game loop
        while not game_over:
            # Encode state
            state = encode_state(player_hand, dealer_hand[0], running_count)
            print(f"State: {state}")

            # Epsilon-greedy action selection
            exploration_rate_threshold = random.uniform(0, 1)
            if exploration_rate_threshold > min_exploration_rate:
                action = np.argmax(Q_table[state])
            else:
                action = random.choice(action_space)
            
            # Perform action
            if action == 1:
                player_hand.append(deal_card())
                print(f"AI hits. AI's cards: {player_hand}")
                if calculate_hand_value(player_hand) > 21:
                    losses += 1
                    print("AI busts.")
                    game_over = True
                elif calculate_hand_value(player_hand) == 21:
                    wins += 1
                    game_over = True
                else:
                    reward = 0  # No immediate reward if player hits
            else:
                # Player stands
                print("AI stands.")
                # Dealer's turn
                while calculate_hand_value(dealer_hand) < 17:
                    dealer_hand.append(deal_card())
                print(f"Dealer's final hand: {dealer_hand}")
                player_value = calculate_hand_value(player_hand)
                dealer_value = calculate_hand_value(dealer_hand)
                if dealer_value > 21 or player_value > dealer_value:
                    wins += 1
                    print("AI wins.")
                elif player_value == dealer_value:
                    pushes += 1
                    print("Push.")
                else:
                    losses += 1
                    print("AI loses.")
                game_over = True

    return wins, pushes, losses
# Call the function to test AI's performance
num_games = 10000
# Initialize deck(s)
num_decks = 6  # Number of decks in the shoe
shoe = list(card_values.keys()) * 4 * num_decks  # Each deck has 4 suits
random.shuffle(shoe)
running_count = 0
wins, pushes, losses = test_ai_performance(Q_table, num_games)

# Calculate win rate
win_rate = wins / num_games * 100

print(f"Win rate: {win_rate}%")
print("wins: ", wins)
print("losses: ", losses)
print("pushes: ", pushes)