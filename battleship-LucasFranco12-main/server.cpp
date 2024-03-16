/*
    ======================================
    =====> DO NOT MODIFY THIS FILE! <=====
    ======================================
    
    Battleship server application

    This code creates a server socket and listens for incoming connections.
    When a client connects, the server adds the new socket to a list of
    connected clients. The server then enters a loop where it waits for
    activity on any of the connected sockets using select(). When a message
    is received from a client, the server sends that message to all
    connected clients except the sender.

    How to compile:
        g++ server.cpp -o server
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <errno.h>
#include "network_packet.h"

#define MAX_CLIENTS 30
#define BACKLOG 10
#define BUFFER_SIZE 100
#define SERVER_PORT 8888

int main(int argc, char *argv[]) {
    int server_socket, addrlen, new_socket, client_socket[MAX_CLIENTS], activity, i, j, valread, sd;
    int max_sd;
    struct sockaddr_in address;

    char buffer[BUFFER_SIZE];
    fd_set readfds;

    // initialize client array
    for (i = 0; i < MAX_CLIENTS; i++) {
        client_socket[i] = 0;
    }

    // create server socket
    if ((server_socket = socket(AF_INET, SOCK_STREAM, 0)) == 0) {
        perror("socket failed");
        exit(EXIT_FAILURE);
    }

    // bind server socket
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(SERVER_PORT);
    addrlen = sizeof(address);
    if (bind(server_socket, (struct sockaddr *)&address, addrlen) < 0) {
        perror("bind failed");
        exit(EXIT_FAILURE);
    }

    // listen on server socket
    if (listen(server_socket, BACKLOG) < 0) {
        perror("listen failed");
        exit(EXIT_FAILURE);
    }
    printf("Server listening:  socket fd %d, ip %s, port %d\n",
            server_socket, inet_ntoa(address.sin_addr), ntohs(address.sin_port));

    // loop on select
    while (1) {
        // clear the file descriptor set
        FD_ZERO(&readfds);

        // add server_socket to the file descriptor set
        FD_SET(server_socket, &readfds);
        max_sd = server_socket;

        // add each client_socket to the file descriptor set
        for (i = 0; i < MAX_CLIENTS; i++) {
            sd = client_socket[i];
            if (sd > 0) {
                FD_SET(sd, &readfds);
            }
            if (sd > max_sd) {
                max_sd = sd;
            }
        }

        // wait on select()
        activity = select(max_sd + 1, &readfds, NULL, NULL, NULL);
        if ((activity < 0) && (errno != EINTR)) {
            printf("select error");
        }

        // new connection?
        if (FD_ISSET(server_socket, &readfds)) {
            // accept connection
            if ((new_socket = accept(server_socket, (struct sockaddr *)&address, (socklen_t *)&addrlen)) < 0) {
                perror("accept failed");
                exit(EXIT_FAILURE);
            }

            // add new connection to first empty client_socket element
            for (i = 0; i < MAX_CLIENTS; i++) {
                if (client_socket[i] == 0) {
                    client_socket[i] = new_socket;
                    break;
                }
            }
            printf("Client connect:    socket fd %d, ip %s, port %d, client_socket[%d]\n",
                    new_socket, inet_ntoa(address.sin_addr), ntohs(address.sin_port), i);

            // upon connection, send client its player_num
            network_packet player;
            player.opcode = OPCODE_CONNECT;
            player.player_num = i; // use index as player_num
            player.x = player.y = player.ship_type = 0;
            send(client_socket[i], &player, sizeof(network_packet), 0);
        }

        // check each client in the file descriptor set
        for (i = 0; i < MAX_CLIENTS; i++) {
            sd = client_socket[i];

            // is client ready for reading?
            if (FD_ISSET(sd, &readfds)) {
                // handle client disconnection
                if ((valread = read(sd, buffer, BUFFER_SIZE)) == 0) {
                    printf("Client disconnect: socket fd %d, client_socket[%d]\n", client_socket[i], i);
                    close(sd);
                    client_socket[i] = 0;
                }
                // data was read
                else {
                    // check for valid packet length
                    if (valread != sizeof(network_packet)) {
                        printf("WARNING: packet size %d from client_socket[%i] ignored\n", valread, i);
                    }
                    else {
                        // echo received packet to all clients except the sender
                        for (j = 0; j < MAX_CLIENTS; j++) {
                            if (client_socket[j] != 0 && client_socket[j] != sd) {
                                send(client_socket[j], buffer, valread, 0);
                            }
                        }
                    }
                }
            }
        }
    }

    return 0;
}
