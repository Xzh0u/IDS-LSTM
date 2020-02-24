from thrift.protocol import TBinaryProtocol
from thrift.transport import TTransport
from thrift.transport import TSocket
from thrift import Thrift
from py.predict import Predictor
import sys
import glob


def main():
    # Make socket
    transport = TSocket.TSocket('localhost', 9090)

    # Buffering is critical. Raw sockets are very slow
    transport = TTransport.TBufferedTransport(transport)

    # Wrap in a protocol
    protocol = TBinaryProtocol.TBinaryProtocol(transport)

    # Create a client to use the protocol encoder
    client = Predictor.Client(protocol)

    # Connect!
    transport.open()

    client.ping()
    print('ping()')

    print(client.pong([2.0, 2.0]))

    li = [-0.075299, 0.229284, -0.327069, 0.374839, 0.664286, 0.308727, -0.241830, -0.095738, 0.127902, -0.121672, 0.253674, 0.692829, -0.224319, -0.484372, 0.658943, -0.253818, 1.802471, -0.155913, -0.230774, 0.104397, 0.172798, 0.100519, 0.144300, 0.055570, -0.221357, 0.065689,
          0.066313, 0.246596, 0.140299, 0.082253, -0.218322, 0.053356, 0.066365, 0.260274, 0.197073, 0.194764, 0.029705, -0.052890, 0.108787, -0.016712, 0.072082, -0.185158, 0.032485, -0.315530, -0.476972, -0.089156, 0.015487, 0.692601, 0.658959, -0.211038, 0.059118, -0.578951]

    result = []
    result = client.predict(li)
    print(result)
    # Close!
    transport.close()


main()
