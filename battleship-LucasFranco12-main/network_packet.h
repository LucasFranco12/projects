#ifndef NETWORK_PACKET
#define NETWORK_PACKET

/*
    ======================================
    =====> DO NOT MODIFY THIS FILE! <=====
    ======================================
    
    Network Packet Specification
        player_num
            0..N - player number
        opcode
            0 - connected to server as player_num
            1 - shot from player_num at (x,y)
            2 - miss from player_num at (x,y)
            3 - hit from player_num ship_type at (x,y)
            4 - hit from player_num and sunk ship_type at (x,y)
        x
            1..10 - x coordinate
        y
            1..10 - y coordinate
        ship_type
            1 - carrier     (5 holes)
            2 - battleship  (4 holes)
            3 - destroyer   (3 holes)
            4 - submarine   (3 holes)
            5 - patrol boat (2 holes)
*/

#define OPCODE_CONNECT   0
#define OPCODE_SHOT      1
#define OPCODE_MISS      2
#define OPCODE_HIT       3
#define OPCODE_HIT_SUNK  4

#define SHIP_CARRIER     1
#define SHIP_BATTLESHIP  2
#define SHIP_DESTROYER   3
#define SHIP_SUBMARINE   4
#define SHIP_PATROL_BOAT 5

typedef struct {
    char player_num;
    char opcode;
    char x;
    char y;
    char ship_type;
} network_packet;

#endif