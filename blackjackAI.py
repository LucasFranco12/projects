import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
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
num_episodes = 1000000
# Initialize running count
running_count = 0

# Function to deal a card from the shoe
def deal_card():
    global shoe, running_count
    if len(shoe) == 0:  # Check if shoe is empty
        # Replenish and shuffle the shoe
        shoe = list(card_values.keys()) * 4 * num_decks
        random.shuffle(shoe)
        running_count = 0  # Reset running_count when the shoe is replenished

    card = shoe.pop()
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

def calculate_true_count(player_hand, dealer_hand, running_count):
    """
    Calculate the true count based on the cards in the player's and dealer's hands.
    """
    # Count cards in the player's hand
    for card in player_hand:
        if card in card_values:
            running_count += card_values[card]

    # Count cards in the dealer's hand (excluding the hidden card)
    for card in dealer_hand[1:]:
        if card in card_values:
            running_count += card_values[card]

    true_count = running_count / (num_decks - len(shoe) / 52)

    return true_count

# Function to encode state of the game
def encode_state(player_hand, dealer_card, true_count):
    state = [card_values[card] for card in player_hand]  # Player hand
    state.append(card_values[dealer_card])  # Dealer's visible card
    state.append(true_count)  # True count
    state.append(num_decks - len(shoe) / 52)  # Decks remaining
    state = state[:4]  
    return np.array(state)

# Build the neural network model
model = keras.Sequential([
    layers.Dense(128, input_shape=(4,), activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Output: probability of hitting
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Function to determine the action based on hand values, dealer's upcard, and true count
def determine_action(player_hand, dealer_upcard, true_count):
    player_value = calculate_hand_value(player_hand)
    dealer_value = card_values[dealer_upcard]

    # Implement your decision logic here based on the player's hand value, dealer's upcard, and true count
    # You can use a set of rules or a pre-computed strategy table
    # This is just an example, you should replace this with your own decision logic

    if true_count >= 2:  # Positive true count (more high cards remaining)
        if player_value < 17:
            return 1  # Hit
        else:
            return 0  # Stand
    elif true_count <= -2:  # Negative true count (more low cards remaining)
        if player_value < 12:
            return 1  # Hit
        else:
            return 0  # Stand
    else:  # Neutral true count
        if player_value < 17 and dealer_value >= 7:
            return 1  # Hit
        else:
            return 0  # Stand

# Training loop
for episode in range(num_episodes):
    print(f"Episode {episode + 1}")
    player_hand = [deal_card(), deal_card()]
    dealer_hand = [deal_card(), deal_card()]
    round_counter = 0
    game_over = False
    true_count = calculate_true_count(player_hand, dealer_hand, running_count)

    while not game_over:
        round_counter += 1
        print(f"Round {round_counter}")
        print(f"Player hand: {player_hand}, Dealer card: {dealer_hand[0]}, True count: {true_count}")

        # Encode state
        state = encode_state(player_hand, dealer_hand[0], true_count)

        # Determine action based on hand values, dealer's upcard, and true count
        action = determine_action(player_hand, dealer_hand[0], true_count)
        print(f"Action: {'Hit' if action == 1 else 'Stand'}")

        if action == 1:
            player_hand.append(deal_card())
            card = player_hand[-1]
            running_count += card_values[card]
            true_count = calculate_true_count(player_hand, dealer_hand, running_count)
            if calculate_hand_value(player_hand) > 21:
                reward = -1  # Player busts
                game_over = True
                break
        else:
            # Player stands
            break

        # Dealer's turn
        if not game_over:
            while calculate_hand_value(dealer_hand) < 17:
                dealer_hand.append(deal_card())

            player_value = calculate_hand_value(player_hand)
            dealer_value = calculate_hand_value(dealer_hand)
            if dealer_value > 21 or player_value > dealer_value:
                reward = 1  # Player wins
            elif player_value == dealer_value:
                reward = 0  # Push
            else:
                reward = -1  # Player loses

        print("True count after round = ", true_count)

        # Update model weights (train on the current state and reward)
        target = np.array([reward])
        model.fit(np.array([state]), target, epochs=1, verbose=0)

        # Reset game variables for the next round
        player_hand = [deal_card(), deal_card()]
        dealer_hand = [deal_card(), deal_card()]
        game_over = True

print("Training completed.")

# Call the function to test AI's performance
num_games = 1000

def test_ai_performance(model, num_games):
    wins = 0
    pushes = 0
    losses = 0
    running_count = 0
    for _ in range(num_games):
        # Initialize game variables
        player_hand = [deal_card(), deal_card()]
        dealer_hand = [deal_card(), deal_card()]

        # Calculate true count after hands are dealt
        true_count = calculate_true_count(player_hand, dealer_hand, running_count)

        game_over = False
        # Main game loop
        while not game_over:
            # Encode state
            state = encode_state(player_hand, dealer_hand[0], true_count)

            # Determine action based on hand values, dealer's upcard, and true count
            action = determine_action(player_hand, dealer_hand[0], true_count)

            # Perform action
            if action == 1:
                player_hand.append(deal_card())
                player_value = calculate_hand_value(player_hand)
                if player_value > 21:
                    losses += 1
                    game_over = True
                    break

            # Dealer's turn (after player's turn)
            if not game_over:
                while calculate_hand_value(dealer_hand) < 17:
                    dealer_hand.append(deal_card())
                player_value = calculate_hand_value(player_hand)
                dealer_value = calculate_hand_value(dealer_hand)
                if dealer_value > 21 or player_value > dealer_value:
                    wins += 1
                    game_over = True
                elif player_value == dealer_value:
                    pushes += 1
                    game_over = True
                else:
                    losses += 1
                    game_over = True

            # Update running count and true count after the round
            running_count = calculate_true_count(player_hand, dealer_hand, running_count)
            true_count = calculate_true_count(player_hand, dealer_hand, running_count)

    return wins, pushes, losses

# Call the function to test AI's performance
wins, pushes, losses = test_ai_performance(model, num_games)

# Calculate win rate
win_rate = wins / num_games * 100
print(f"Win rate: {win_rate}%")