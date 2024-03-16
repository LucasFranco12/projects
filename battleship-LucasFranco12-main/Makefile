CC=gcc
CXX=g++
RM=rm -f
CPPFLAGS=-Wall -g
LDFLAGS=-g -lpthread
LDLIBS=

C_SRCS=test_client.cpp
C_OBJS=$(subst .cpp,.o,$(C_SRCS))

S_SRCS=server.cpp
S_OBJS=$(subst .cpp,.o,$(S_SRCS))

all: test_client server

test_client: $(C_OBJS)
	$(CXX) $(LDFLAGS) -o test_client $(C_OBJS) $(LDLIBS)

server: $(S_OBJS)
	$(CXX) $(LDFLAGS) -o server $(S_OBJS) $(LDLIBS)

test_client.o: test_client.cpp network_packet.h
server.o: server.cpp network_packet.h

clean:
	$(RM) $(C_OBJS) $(S_OBJS)

distclean: clean
	$(RM) test_client server

