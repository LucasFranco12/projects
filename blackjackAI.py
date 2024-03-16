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
num_episodes = 2
# Initialize running count
running_count = 0

# Function to deal a card from the shoe
def deal_card():
    global shoe
    if len(shoe) == 0:  # Check if shoe is empty
        # Replenish and shuffle the shoe
        shoe = list(card_values.keys()) * 4 * num_decks
        random.shuffle(shoe)

    card = shoe.pop()
    return card

# Function to calculate hand value
def calculate_hand_value(hand):
    value = 0
    for card in hand:
        value += card_values[card]
    return value

# Function to encode state of the game
def encode_state(player_hand, dealer_card):
    state = [card_values[card] for card in player_hand]  # Player hand
    state.append(card_values[dealer_card])  # Dealer's visible card
    state.append(running_count)  # Running count
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


# Training loop
for episode in range(num_episodes):
    # Initialize game variables
    player_hand = [deal_card(), deal_card()]
    dealer_hand = [deal_card(), deal_card()]
    running_count = 0  # Reset running count for each episode
    game_over = False
    max_rounds_per_episode = 100  # Define the maximum rounds per episode
    round_counter = 0
    
    while not game_over:

        round_counter += 1
        if round_counter > max_rounds_per_episode:
            print("Maximum rounds reached. Ending episode.")
            break


        # Encode state
        state = encode_state(player_hand, dealer_hand[0])
        
        # Model predicts action (hit or stand)
        action_prob = model.predict(np.array([state]))[0][0]
        action = 1 if action_prob > 0.5 else 0  # Convert probability to action
        
        # Perform action (hit or stand)
        if action == 1:
            player_hand.append(deal_card())
            player_value = calculate_hand_value(player_hand)
            if player_value > 21:
                reward = -1  # Player busts
                game_over = True
            else:
                reward = 0  # No immediate reward for hitting
        else:
            # Dealer's turn
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
            game_over = True
        
        # Update running count
        card = player_hand[-1] if action == 1 else dealer_hand[-1]
        running_count += card_values[card]
        
        # Update model weights (train on the current state and reward)
        target = np.array([reward])
        model.fit(np.array([state]), target, epochs=1, verbose=0)
# Call the function to test AI's performance
num_games = 2
test_ai_performance(model, num_games)

# Test the AI's performance against the house
def test_ai_performance(model, num_games):
    wins = 0
    pushes = 0
    losses = 0

    for _ in range(num_games):
        # Initialize game variables
        player_hand = [deal_card(), deal_card()]
        dealer_hand = [deal_card(), deal_card()]
        running_count = 0  # Reset running count for each game
        game_over = False
        max_iterations = 1000  # Define the maximum iterations
        iteration_counter = 0
        # Main game loop
        while not game_over:
            iteration_counter += 1
            if iteration_counter > max_iterations:
                print("Maximum iterations reached. Exiting loop.")
                break
      

            # Encode state
            state = encode_state(player_hand, dealer_hand[0])
            
            # Decision based on AI model
            action_prob = model.predict(np.array([state]))[0][0]
            action = 1 if action_prob > 0.5 else 0
            
            # Perform action
            if action == 1:
                player_hand.append(deal_card())
                player_value = calculate_hand_value(player_hand)
                if player_value > 21:
                    losses += 1
                    game_over = True
            else:
                # Dealer's turn
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
    
    # Calculate win rate
    win_rate = wins / num_games * 100
    print(f"Win rate: {win_rate}%")