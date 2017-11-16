import sys
sys.path.append('gen-py')

from data_format import DataService
from data_format.ttypes import *

from thrift import Thrift
from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol

transport = TSocket.TSocket('localhost', 9090)
transport = TTransport.TBufferedTransport(transport)
protocol = TBinaryProtocol.TBinaryProtocol(transport)
client = DataService.Client(protocol)
transport.open()
client.ping()
type = 1
datas = [Data([2,13,12.275,12.6841,0.0508,1]),Data([5,10,14.275,11.6841,0.0908,1]),
         Data([3,10,14.875,14.1382,0.0821,1]),Data([5,10,15.075,14.241,0.0808,1]),
         Data([1,11,13.275,11.6841,0.0208,1])]
print(Parameters(client.buildModel(BussinessType(type), datas)))
transport.close()


