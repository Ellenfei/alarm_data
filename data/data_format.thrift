enum BussinessType {
  VIDEO = 1,
  ECOMMERCE = 2,
  EMAIL = 3,
  FILE_TRANSFER = 4
}

struct Data {
    1: i32 nodeS
    2: i32 nodeD
    3: double bandwidth
    4: double delay
    5: double loss
    6: i32 flag
}

struct Parameters {  
    1: list<list<double>> w1
    2: list<double> b1  
    3: list<list<double>> w2
    4: list<double> b2
}

service DataService {
    Parameters buildModel(1:BussinessType type, 2:list<Data> datas) 
}

