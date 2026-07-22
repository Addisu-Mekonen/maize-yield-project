from pydantic import BaseModel

class TabularInput(BaseModel):
    Rainfall_mm: float
    Temperature_Celsius: float
    Fertilizer_Used: int
    Irrigation_Used: int
    Days_to_Harvest: int
    Region_East: int
    Region_North: int
    Region_South: int
    Region_West: int
    Soil_Type_Chalky: int
    Soil_Type_Clay: int
    Soil_Type_Loam: int
    Soil_Type_Peaty: int
    Soil_Type_Sandy: int
    Soil_Type_Silt: int
    Weather_Condition_Cloudy: int
    Weather_Condition_Rainy: int
    Weather_Condition_Sunny: int

class PredictionResponse(BaseModel):
    predicted_disease: str
    disease_confidence: float
    p_healthy: float
    severity_scale: float
    base_yield_tons_per_hectare: float
    yield_adjustment_factor: float
    final_yield_tons_per_hectare: float