from pccl import *

HOST: str = '0.0.0.0:48148'


def main():
    print(f"Starting master node on {HOST}")
    master: MasterNode = MasterNode(listen_address=HOST)
    master.run()


if __name__ == '__main__':
    main()
