from typing import Optional
import socket
import fcntl
import struct

MULTIPLIER = {"Kbits/sec": 1e3, "Mbits/sec": 1e6, "Gbits/sec": 1e9, "Tbits/sec": 1e12}


def parse_iperf_output(output: str) -> Optional[int]:
    try:
        value, mult = output.strip().split()[-2:]
        return int(float(value) * MULTIPLIER[mult])
    except Exception:
        return None


# Taken from https://stackoverflow.com/questions/24196932/how-can-i-get-the-ip-address-from-a-nic-network-interface-controller-in-python
def get_ip_address(ifname: str) -> str:
    """Get the IP address of the specified network interface.

    Args:
        ifname (str): The name of the network interface.
    Returns:
        str: The IP address of the network interface.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    ret = socket.inet_ntoa(
        fcntl.ioctl(
            s.fileno(),
            0x8915,  # SIOCGIFADDR
            struct.pack("256s", ifname.encode("utf-8")[:15]),
        )[20:24]
    )
    s.close()
    return ret
