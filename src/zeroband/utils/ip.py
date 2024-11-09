import socket
import fcntl
import struct

# Taken from https://stackoverflow.com/questions/24196932/how-can-i-get-the-ip-address-from-a-nic-network-interface-controller-in-python
def get_ip_address(ifname: str) -> str:
    """Get the IP address of the specified network interface.
    
    Args:
        ifname (str): The name of the network interface.
    Returns:
        str: The IP address of the network interface.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    ret = socket.inet_ntoa(fcntl.ioctl(
        s.fileno(),
        0x8915,  # SIOCGIFADDR
        struct.pack('256s', ifname.encode('utf-8')[:15])
    )[20:24])
    s.close()
    return ret
