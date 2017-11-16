class Data(object):
    """
    Attributes:
     - nodeS
     - nodeD
     - bandwidth
     - delay
     - loss
     - flag
    """

    # thrift_spec = (
    #     None,  # 0
    #     (1, TType.I32, 'nodeS', None, None, ),  # 1
    #     (2, TType.I32, 'nodeD', None, None, ),  # 2
    #     (3, TType.DOUBLE, 'bandwidth', None, None, ),  # 3
    #     (4, TType.DOUBLE, 'delay', None, None, ),  # 4
    #     (5, TType.DOUBLE, 'loss', None, None, ),  # 5
    #     (6, TType.I32, 'flag', None, None, ),  # 6
    # )

    def __init__(self, nodeS=None, nodeD=None, bandwidth=None, delay=None, loss=None, flag=None,):
        self.nodeS = nodeS
        self.nodeD = nodeD
        self.bandwidth = bandwidth
        self.delay = delay
        self.loss = loss
        self.flag = flag
