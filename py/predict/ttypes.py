#
# Autogenerated by Thrift Compiler (0.13.0)
#
# DO NOT EDIT UNLESS YOU ARE SURE THAT YOU KNOW WHAT YOU ARE DOING
#
#  options string: py
#

from thrift.Thrift import TType, TMessageType, TFrozenDict, TException, TApplicationException
from thrift.protocol.TProtocol import TProtocolException
from thrift.TRecursive import fix_spec

import sys

from thrift.transport import TTransport
all_structs = []


class pred(object):
    """
    Attributes:
     - type
     - loss
     - timestamp

    """


    def __init__(self, type=None, loss=None, timestamp=None,):
        self.type = type
        self.loss = loss
        self.timestamp = timestamp

    def read(self, iprot):
        if iprot._fast_decode is not None and isinstance(iprot.trans, TTransport.CReadableTransport) and self.thrift_spec is not None:
            iprot._fast_decode(self, iprot, [self.__class__, self.thrift_spec])
            return
        iprot.readStructBegin()
        while True:
            (fname, ftype, fid) = iprot.readFieldBegin()
            if ftype == TType.STOP:
                break
            if fid == 1:
                if ftype == TType.I32:
                    self.type = iprot.readI32()
                else:
                    iprot.skip(ftype)
            elif fid == 2:
                if ftype == TType.DOUBLE:
                    self.loss = iprot.readDouble()
                else:
                    iprot.skip(ftype)
            elif fid == 3:
                if ftype == TType.I32:
                    self.timestamp = iprot.readI32()
                else:
                    iprot.skip(ftype)
            else:
                iprot.skip(ftype)
            iprot.readFieldEnd()
        iprot.readStructEnd()

    def write(self, oprot):
        if oprot._fast_encode is not None and self.thrift_spec is not None:
            oprot.trans.write(oprot._fast_encode(self, [self.__class__, self.thrift_spec]))
            return
        oprot.writeStructBegin('pred')
        if self.type is not None:
            oprot.writeFieldBegin('type', TType.I32, 1)
            oprot.writeI32(self.type)
            oprot.writeFieldEnd()
        if self.loss is not None:
            oprot.writeFieldBegin('loss', TType.DOUBLE, 2)
            oprot.writeDouble(self.loss)
            oprot.writeFieldEnd()
        if self.timestamp is not None:
            oprot.writeFieldBegin('timestamp', TType.I32, 3)
            oprot.writeI32(self.timestamp)
            oprot.writeFieldEnd()
        oprot.writeFieldStop()
        oprot.writeStructEnd()

    def validate(self):
        return

    def __repr__(self):
        L = ['%s=%r' % (key, value)
             for key, value in self.__dict__.items()]
        return '%s(%s)' % (self.__class__.__name__, ', '.join(L))

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.__dict__ == other.__dict__

    def __ne__(self, other):
        return not (self == other)
all_structs.append(pred)
pred.thrift_spec = (
    None,  # 0
    (1, TType.I32, 'type', None, None, ),  # 1
    (2, TType.DOUBLE, 'loss', None, None, ),  # 2
    (3, TType.I32, 'timestamp', None, None, ),  # 3
)
fix_spec(all_structs)
del all_structs
