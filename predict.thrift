namespace py predict

struct pred  {
    1:i32 type;
    2:double loss;
    3:string timestamp;
}

service Predictor {
    void ping(),

    list<double> pong(1:list<double> data),

    pred predict(1:list<double> data, 2:string timestamp),
}
