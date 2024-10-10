#include "SocketCommunicator.h"
#include <iostream>
#include <cstring>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <Python.h>

constexpr size_t BUFFER_SIZE = 1024;

SocketCommunicator::SocketCommunicator(const std::string& listen_address, unsigned short listen_port)
    : send_sockfd(-1), recv_sockfd(-1), listening(false)
{
    // Create the receiving socket
    recv_sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    if (recv_sockfd < 0) {
        perror("Failed to create receive socket");
        throw std::runtime_error("Failed to create receive socket");
    }

    // Set up the address structure
    struct sockaddr_in recv_addr;
    std::memset(&recv_addr, 0, sizeof(recv_addr));
    recv_addr.sin_family = AF_INET;
    recv_addr.sin_port = htons(listen_port);
    if (listen_address.empty()) {
        recv_addr.sin_addr.s_addr = INADDR_ANY;
    } else {
        if (inet_aton(listen_address.c_str(), &recv_addr.sin_addr) == 0) {
            perror("Invalid listen address");
            throw std::runtime_error("Invalid listen address");
        }
    }

    // Bind the receiving socket to the specified address and port
    if (bind(recv_sockfd, (struct sockaddr*)&recv_addr, sizeof(recv_addr)) < 0) {
        perror("Failed to bind receive socket");
        throw std::runtime_error("Failed to bind receive socket");
    }

    // Start the listening thread
    startListening();
}

SocketCommunicator::~SocketCommunicator()
{
    listening = false;
    if (listen_thread.joinable()) {
        listen_thread.join();
    }

    if (recv_sockfd >= 0) {
        close(recv_sockfd);
    }

    if (send_sockfd >= 0) {
        close(send_sockfd);
    }
}

void SocketCommunicator::startListening()
{
    listening = true;
    listen_thread = std::thread(&SocketCommunicator::listenLoop, this);
}

void SocketCommunicator::listenLoop()
{
    char buffer[BUFFER_SIZE];

    while (listening) {
        struct sockaddr_in sender_addr;
        socklen_t addrlen = sizeof(sender_addr);
        ssize_t received_bytes = recvfrom(recv_sockfd, buffer, BUFFER_SIZE, 0, (struct sockaddr*)&sender_addr, &addrlen);
        if (received_bytes < 0) {
            perror("Failed to receive data");
            continue;
        }

        // Print received bytes to stdout
        std::cout.write(buffer, received_bytes);
        std::cout.flush();
    }
}

void SocketCommunicator::setTarget(const std::string& target_address, unsigned short target_port)
{
    // Create the sending socket
    send_sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    if (send_sockfd < 0) {
        perror("Failed to create send socket");
        throw std::runtime_error("Failed to create send socket");
    }

    // Set the target address and port
    std::memset(&target_addr, 0, sizeof(target_addr));
    target_addr.sin_family = AF_INET;
    target_addr.sin_port = htons(target_port);
    if (inet_aton(target_address.c_str(), &target_addr.sin_addr) == 0) {
        perror("Invalid target address");
        throw std::runtime_error("Invalid target address");
    }
}

void SocketCommunicator::sendData(const std::string& data)
{
    if (send_sockfd < 0) {
        std::cerr << "Send socket not initialized. Call setTarget() first." << std::endl;
        return;
    }

    ssize_t sent_bytes = sendto(send_sockfd, data.c_str(), data.size(), 0,
                                (struct sockaddr*)&target_addr, sizeof(target_addr));
    if (sent_bytes < 0) {
        perror("Failed to send data");
    }
}

void SocketCommunicator::setDataCallback(std::function<void(const std::string& data)> callback)
{
    data_callback = callback;
}

void SocketCommunicator::listenLoop()
{
    const size_t BUFFER_SIZE = 1024;
    char buffer[BUFFER_SIZE];

    while (listening) {
        struct sockaddr_in sender_addr;
        socklen_t addrlen = sizeof(sender_addr);
        ssize_t received_bytes = recvfrom(recv_sockfd, buffer, BUFFER_SIZE, 0,
                                          (struct sockaddr*)&sender_addr, &addrlen);
        if (received_bytes < 0) {
            perror("Failed to receive data");
            continue;
        }

        // Create a string from the received data
        std::string data(buffer, received_bytes);

        // Call the data callback if set
        if (data_callback) {
            // Acquire GIL before calling Python function
            PyGILState_STATE gstate;
            gstate = PyGILState_Ensure();

            try {
                data_callback(data);
            } catch (...) {
                // Handle exceptions to prevent thread termination
                PyErr_Print();
            }

            PyGILState_Release(gstate);
        } else {
            // If no callback is set, print to stdout (optional)
            std::cout.write(buffer, received_bytes);
            std::cout.flush();
        }
    }
}
