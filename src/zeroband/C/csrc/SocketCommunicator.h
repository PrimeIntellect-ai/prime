#ifndef SOCKET_COMMUNICATOR_H
#define SOCKET_COMMUNICATOR_H

#include <string>
#include <thread>
#include <atomic>
#include <functional>
#include <netinet/in.h>
#include <sys/socket.h>

class SocketCommunicator {
public:
    // Constructor: takes listening address and port
    SocketCommunicator(const std::string& listen_address, unsigned short listen_port);

    // Method to set target address and port for sending data
    void setTarget(const std::string& target_address, unsigned short target_port);

    // Method to send data
    void sendData(const std::string& data);

    // Method to set the data callback
    void setDataCallback(std::function<void(const std::string& data)> callback);

    // Destructor: closes sockets and stops listening
    ~SocketCommunicator();

private:
    int send_sockfd;
    int recv_sockfd;
    struct sockaddr_in target_addr;
    std::atomic<bool> listening;
    std::thread listen_thread;

    // Data callback function
    std::function<void(const std::string& data)> data_callback;

    // Starts the listening thread
    void startListening();

    // The listening loop that receives data and calls the callback
    void listenLoop();

    // Disable copying
    SocketCommunicator(const SocketCommunicator&) = delete;
    SocketCommunicator& operator=(const SocketCommunicator&) = delete;
};

#endif // SOCKET_COMMUNICATOR_H
