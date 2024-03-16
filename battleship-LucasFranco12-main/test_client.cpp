//Lucas Franco, Project3
//This file contains the implementation of a multiplayer battleship game client. It  prompts the user to place ships on their board and updates it. After all ships are placed the client establishes connection with the server using the IP address and port number of the server. The client then enters a loop seeing their grids and a menu which informs the client on how to facilate gameplay between clients connected to the server. The client also waits for incoming messages from the server and acts upon the game updates communicated by the server. It handles shooting actions, updates game grids based on the server's responses, and continuously displays the state of the player's ocean and target grids.
//The Board class inside this file manages the gameboard logic, ship placement, and tracking of hits and misses.

#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include "network_packet.h"
#include <iostream>
#include <iomanip>  
#include <algorithm>
#include <cstdlib>  


#define LOCALHOST "127.0.0.1"
#define SERVER_PORT 8888
#define BUFFER_SIZE 100

int client_socket = -1;
int player_number = -1;

//represent orientation of the ship
enum class Orientation {
    Horizontal,
    Vertical
};

//This class manages updating and displaying the grids, ship placement, and tracking of hits and misses. Along with this it keeps track of which ships are sunk.
class Board {
private:
    static const int SIZE = 10;
    char oceanGrid[SIZE][SIZE];
    char targetGrid[SIZE][SIZE];
    std::vector<int> ships = {2, 3, 3, 4, 5};
    std::vector<int> shipsSunk = {2, 3, 3, 4, 5};

public:
    Board() {
        // Initialize ocean and target grid 
        for (int i = 0; i < SIZE; ++i) {
            for (int j = 0; j < SIZE; ++j) {
                oceanGrid[i][j] = 'O';
                targetGrid[i][j] = 'O';
            }
        }
     
    }
    
   // Displays the ocean grid
   void displayOceanGrid() const {
        std::cout << "Ocean Grid: \n";
        std::cout << "  ";
        for (int i = 0; i < SIZE; ++i) {
            std::cout << i << ' ';
        }
        std::cout << "\n";

        for (int i = 0; i < SIZE; ++i) {
            std::cout  << i << ' ';
            for (int j = 0; j < SIZE; ++j) {
                std::cout << oceanGrid[i][j] << ' ';
            }
            std::cout << '\n';
        }
   }


   //Returns ships vector
   const std::vector<int>& getAvailableShip() const {
      return ships;
   }

   // Displays the target grid
   void displayTargetGrid() const {
       std::cout << "Target Grid: \n";
       std::cout << "  ";
       for (int i = 0; i < SIZE; ++i) {
           std::cout  << i << ' ';
       }
       std::cout << "\n";

       for (int i = 0; i < SIZE; ++i) {
           std::cout << i << ' ';
           for (int j = 0; j < SIZE; ++j) {
               std::cout << targetGrid[i][j] << ' ';
           }
           std::cout << '\n';
       }
   }
    
    //This method takes the ship type given and removes it from the shipsSunk vector. If all your ships have sunk then tells you you lost.
    void shipSunk(int shipType) {
        auto it = std::find(shipsSunk.begin(), shipsSunk.end(), shipType);
        if (it != shipsSunk.end()) {
            shipsSunk.erase(it);

            if (shipsSunk.empty()) {
                printf("All your ships are sunk. You lost!\n");
                printf("Please enter 5 5 5 5 to announce your defeat\n");
            }
        }
    }

   //This returns true if the given x y do not have a O meaning that it shot a ship. Also checks to see if the coordinates are within the valid range
   bool shot(int x, int y) const {
    if (x < 0 || x >= SIZE || y < 0 || y >= SIZE) {
        std::cout << "Invalid coordinates.\n";
        return false;
    }
    return oceanGrid[y][x] != 'O';
}

   // Updates the target grid, if the isHit recieved from the shot function is true or false it updates that cell with a X or a M. If there was already a X in the target grid it cannot be over written.
   void updateTargetGrid(int x, int y, bool isHit) {
   
       if (!isHit && targetGrid[y][x] == 'X') {
           return;
       }
       if (isHit) {
           targetGrid[y][x] = 'X';
         
       } else {
            targetGrid[y][x] = 'M';
      
       }
   }

   // Update the Ocean grid, if the isHit recieved from the shot function is true or false it updates that cell with a X or a M.
   void updateOceanGrid(int x, int y, bool isHit) {
       if (isHit) {
  
           oceanGrid[y][x] = 'X';
       } else {

            oceanGrid[y][x] = 'M';
       }
   }

    // Display the available ships
    void displayAvailableShip() const {
        std::cout << "Place boats: ";
        for (int size : ships) {
            std::cout << size << ' ';
        }
        std::cout << "\n";
    }

    // This method places a ship. It checks if the ship size is available, checks if the ship can fit in the specified location and orientation, then places the ship and removes it from the ships vector
  bool placeShip(int x, int y, int size, int orientation) {
   
    auto i = std::find(ships.begin(), ships.end(), size);
    if (i == ships.end()) {
        std::cout << "Ship size " << size << " not available.\n";
        return false;
    }

    if (orientation == 0) {  // Horizontal
        if (x + size > SIZE) {
            std::cout << "Cannot place the ship out of bounds.\n";
            return false;
        }
        for (int i = x; i < x + size; ++i) {
            if (oceanGrid[y][i] != 'O') {
                std::cout << "Cannot place the ship on top of another ship.\n";
                return false;
            }
        }
       
        for (int i = x; i < x + size; ++i) {
            oceanGrid[y][i] = size + '0';
        }
    } else {  // Vertical
        if (y + size > SIZE) {
            std::cout << "Cannot place the ship out of bounds.\n";
            return false;
        }
        for (int i = y; i < y + size; ++i) {
            if (oceanGrid[i][x] != 'O') {
                std::cout << "Cannot place the ship on top of another ship.\n";
                return false;
            }
        }
       
        for (int i = y; i < y + size; ++i) {
            oceanGrid[i][x] = size + '0';
        }
    }

    // Remove the ship size from the vector
    ships.erase(i);

    return true;
}

};


// network send thread 
void *tx(void *arg) {
    int ret;
    char buffer[BUFFER_SIZE];
    network_packet pkt;
    Board *gameBoard = static_cast<Board*>(arg); 


    // delay to allow recv of connect opcode
    sleep(1);

    // loop forever
    while (1) {
     
        // prompt
        gameBoard->displayOceanGrid();
        std::cout << "\n";
        gameBoard->displayTargetGrid();  
        std::cout << "Menu:\n";
    	std::cout << "Opcode:1. Shoot\n";
    	std::cout << "Opcode:2. Inform opponent of a missed shot\n";
    	std::cout << "Opcode:3. Inform opponent of a hit\n";
    	std::cout << "Opcode:4. Inform opponent of a sunk ship\n";
        std::cout << "For Opcode (1-3) enter 0 as the ship_type\n";
        printf("Enter opcode, x, y, ship_type: ");
        fflush(stdout);

        // read and parse test packet struct members from user
        fgets(buffer, BUFFER_SIZE, stdin);
        if ((ret = sscanf(buffer, "%hhd %hhd %hhd %hhd", &pkt.opcode, &pkt.x, &pkt.y, &pkt.ship_type)) != 4) {
            printf("bad input!\n");
            fflush(stdout);
        }
        else {
            // send test packet to server
            pkt.player_num = player_number;
            if ((ret = send(client_socket, &pkt, sizeof(network_packet), 0) != sizeof(network_packet))) {
                perror("send error");
            }
            else {
                printf("Sent: %hhd %hhd %hhd %hhd %hhd\n", pkt.player_num, pkt.opcode, pkt.x, pkt.y, pkt.ship_type);
            }
            if (pkt.opcode == 4) {
               gameBoard->shipSunk(pkt.ship_type); 
            }
            
        }
    }
}

// network recv thread
void *rx(void *arg) {
    int ret;
    char buffer[BUFFER_SIZE];
    network_packet pkt;
    Board *gameBoard = static_cast<Board*>(arg);

    // loop forever
    while (1) {
        // block on read
        if ((ret = recv(client_socket, buffer, BUFFER_SIZE, 0)) == 0) {
            perror("recv failed");
            exit(EXIT_FAILURE);
        }
        // data was read
        else {
            // check for valid packet length
            if (ret != sizeof(network_packet)) {
                printf("WARNING: packet size %d from client_socket ignored\n", ret);
            }
            else {
                // copy the receive buffer into the pkt struct
                memcpy(&pkt, buffer, sizeof(network_packet));


                if (pkt.opcode == OPCODE_CONNECT) {
                    player_number = pkt.player_num;
                    printf("I am player number %d\n", player_number);
                }
                else {
                    // display received packet fields
                    printf("\nReceived: %hhd %hhd %hhd %hhd %hhd\n", pkt.player_num, pkt.opcode, pkt.x, pkt.y, pkt.ship_type);
                    
                    //Losers broadcast
                    if (pkt.opcode == 5 && pkt.x == 5 && pkt.y == 5 && pkt.ship_type == 5) {
                       printf("Player: %hhd, is out of the game\n", pkt.player_num);
                    }
                    
                    //recieved opcode 1, signifing a shot. Updates Ocean grid accordingly.
                    if (pkt.opcode == 1) {
                        if (gameBoard->shot(pkt.x, pkt.y)) {
                            printf("Player: %hhd shot and hit (%hhd,%hhd). Please inform your opponents\n", pkt.player_num,pkt.x,pkt.y);
                            gameBoard->updateOceanGrid(pkt.x, pkt.y, true);
                        } else {
                            printf("Player: %hhd shot and missed at (%hhd,%hhd). Please inform your opponent\n", pkt.player_num,pkt.x,pkt.y);
                            gameBoard->updateOceanGrid(pkt.x, pkt.y, false);
                        }
                      //recieved opcode 2, signifying a miss. Updates the target grid accordingly.
                    } else if (pkt.opcode == 2) {
                        printf("Player: %hhd was missed at (%hhd,%hhd)\n", pkt.player_num,pkt.x,pkt.y);
                        gameBoard->updateTargetGrid(pkt.x, pkt.y, false);
                        
                       //recieved opcode 3, signifying a shot hit, updates target grid accordingly.
                    } else if (pkt.opcode == 3) {
                        printf("Player: %hhd was hit at (%hhd,%hhd)\n", pkt.player_num,pkt.x,pkt.y);
                        gameBoard->updateTargetGrid(pkt.x, pkt.y, true);
                        
                        //recieved opcdoe 4, player announces which ship of theirs was sunk
                    } else if (pkt.opcode == 4) {
                        printf("Player: %hhd ship sunk it was size %hhd\n", pkt.player_num,pkt.ship_type);
                        gameBoard->updateTargetGrid(pkt.x, pkt.y, true);
                    }

		    //displays grids and a menu.
                    gameBoard->displayOceanGrid();
        	    std::cout << "\n";
                    gameBoard->displayTargetGrid();  
                    std::cout << "Menu:\n";
    	            std::cout << "Opcode:1. Shoot\n";
    	            std::cout << "Opcode:2. Inform opponent of a missed shot\n";
    	            std::cout << "Opcode:3. Inform opponent of a hit\n";
    	            std::cout << "Opcode:4. Inform opponent of a sunk ship\n";
                    std::cout << "For Opcode (1-3) enter 0 as the ship_type\n";
                    printf("Enter opcode, x, y, ship_type: ");
                    fflush(stdout);
                }
            }
        }
    }
    return 0;
}
//This method prompts the user for coordinates, ship size, and orientation so it can be sent to the placeShip method inside the board class.
void getUserInput(int& x, int& y, int& size, Orientation& orientation, Board& gameBoard) {

    gameBoard.displayAvailableShip();

    std::cout << "Enter the starting coordinates (x y) for the ship: ";
    std::cin >> x >> y;
    std::cout << "Enter the size of the ship: ";
    std::cin >> size;
    std::cout << "Enter the orientation (0 for Horizontal, 1 for Vertical): ";
    int orientationInput;
    std::cin >> orientationInput;

    if (orientationInput == 0) {
      orientation = Orientation::Horizontal;
    } else {
      orientation = Orientation::Vertical;
    }
}

int main(int argc, char *argv[]) {
    struct sockaddr_in serv_addr;
    char const *server_address;

    Board gameBoard;

    gameBoard.displayOceanGrid();

    //loops until all ships are placed.
    while (!gameBoard.getAvailableShip().empty()) {
        int x, y, size;
        Orientation orientation;

        // Prompt user for ship placement
        getUserInput(x, y, size, orientation, gameBoard);

        if (gameBoard.placeShip(x, y, size, static_cast<int>(orientation))) {
            std::cout << "Ship placed successfully.\n";
        } else {
            std::cout << "Cannot place the ship at the specified location.\n";
        }

        // Display the updated ocean grid
        gameBoard.displayOceanGrid();
        
    }


    // use server addr if specified on command line, otherwise use localhost
    if (argc > 1) {
        server_address = argv[1];
    }
    else {
        server_address = LOCALHOST;
    }

    // create client socket
    if ((client_socket = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
        perror("socket failed");
        exit(EXIT_FAILURE);
    }

    // initialize sockaddr_in fields
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(SERVER_PORT);
    if (inet_pton(AF_INET, server_address, &serv_addr.sin_addr) <= 0) {
        perror("inet_pton failed");
        exit(EXIT_FAILURE);
    }

    // connect to server
    if (connect(client_socket, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) {
        perror("connect failed");
        exit(EXIT_FAILURE);
    }

    // Create two threads
    pthread_t thread0, thread1;
    pthread_create(&thread0, NULL, tx,  &gameBoard);
    pthread_create(&thread1, NULL, rx,  &gameBoard);

    // Wait for threads to finish
    pthread_join(thread0, NULL);
    pthread_join(thread1, NULL);

    return 0;
}
