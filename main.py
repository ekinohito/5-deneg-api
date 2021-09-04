from fastapi import FastAPI
from catboost import CatBoostRegressor
from fastapi.middleware.cors import CORSMiddleware
template = [616.2, 104.7, 104, 285000, 101.4, 98.5, 107, 100, 100.1, 100,
        42500, 100, 130000, 102, 95400, 98, 62100, 190, 95, 6.5, 2.1]
cbr = CatBoostRegressor().load_model('model1.cbm')
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=['*'])


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/results")
def results(param1: float, param2: float, param3: float, param4: float, param5: float, param6: float, param7: float):
    form = template.copy()
    form[0] = param1
    form[1] = param2
    form[2] = param3
    form[3] = param4
    form[4] = param5
    form[5] = param6
    form[6] = param7
    print(form)
    try:
        cbr.predict(form)
    except Exception as e:
        print(e)
        return {}
    return {
        "ndfl": cbr.predict(form),
        "npp": "2001",
        "total": "500"
    }

@app.get("/spending")
def spending(debt: float, deficit: float, total: float):
    return {
        "total": "12345"
    }

