from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

income_coefficients = {
    "ndfl": 0.18,
    "npp": 0.14,
    "total": 1.00
}
spending_coefficients = {
    "gov": 0.0342,
    "def": 0.0004,
    "sec": 0.0176,
    "econ": 0.1795,
    "comm": 0.0810,
    "ecol": 0.0008,
    "edu": 0.2025,
    "cult": 0.0230,
    "med": 0.1049,
    "soc": 0.2904,
    "sport": 0.0071,
    "media": 0.0021,
    "debt": 0.0210,
    "cross": 0.0356,
    "total": 1.0000
}
model = {"coefficients": [
    [54349722.057218164, -154532215.97964692, 320483062.39440435, -274131247.61546963, -354321475.5543245,
     269248539.7067261, -175659418.7339717, 67320574.72961102, 5680388.870238143, -21851601.549673665,
     72559.88124599053, 212265.27803407636, 278988.0266015916, 70993645.73021837, 228067361.93513376,
     2845585681.6621666, 0.7504901761861549]], "intercept": [[-37963988467.95378]],
         "type": "LinearRegression"}
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=['*'])


def predict(params):
    return sum(map(lambda x, y: x * y, model["coefficients"][0], params)) + model["intercept"][0][0]


@app.get("/results")
def results(population: float, index1: float, index2: float, index_cde: float, index_c: float,
            index_d: float, index_e: float, ingex_f: float, alco: float, inv2: float, roz1: float,
            salary: float, tax: float, import2: float, unemp1: float, unemp2: float, forecast: float):
    prediction = predict([
        population, index1, index2, index_cde, index_c, index_d, index_e, ingex_f, alco,
        inv2, roz1, salary, tax, import2, unemp1, unemp2, forecast
    ])
    return {key: value * prediction for (key, value) in income_coefficients.items()}


@app.get("/spending")
def spending(debt: float, deficit: float, total: float):
    budget = total + debt + deficit
    return {key: value * budget for (key, value) in spending_coefficients.items()}
