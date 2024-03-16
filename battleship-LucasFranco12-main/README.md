# **Battle Ship**
**Author:** Lucas Franco
![Imgur](https://i.imgur.com/4gk5lr2.png)

## Description

This project implements a multiplayer Battleship game client that supports network play with up to 30 people. The game allows users to place ships on their board, engage in battles with opponents connected to the server. It features a simple and intuitive interface for interacting with the game, making it an exciting multiplayer experience.

## How to Play

1. Place your ships on the ocean grid.
2. Wait for everyone playing to be done placing their ships
3. Starting from the player with the lowest player number, each player will take turns firing shots.
4. After a player has fired their shot, please respond telling them if they hit, missed, or sunk so the game can update their boards accordingly.
5. If all of your ships have been destroyed then you lose.

## How to Run
1. Clone this repository
    ```bash
   git clone https://github.com/TCNJ-degoodj/battleship-LucasFranco12
    ```
2. Enter 'make' to make sure its up to date
3. Start the server
    ```bash
    ./server
    ```
4. Start the client
    ```bash
    ./test_server
    ```
## License
This project is licensed under the [MIT License](LICENSE).
---
:+1: Instructor's GitHub username: [@jdegood](https://github.com/jdegood)
