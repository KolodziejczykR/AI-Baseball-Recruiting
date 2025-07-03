from fastapi import FastAPI
from backend.ml.ml_router import router as ml_router

app = FastAPI(title="AI/ML Baseball Recruitment Backend")

# Import and include the ML router
app.include_router(ml_router, prefix="/predict")

@app.get("/")
def read_root():
    return {"message": "Welcome to the AI/ML Baseball Recruitment API!"}

@app.get("/ping")
def health_check():
    return {"status": "ok"}

# from backend.api.llm import router as llm_router
# app.include_router(llm_router, prefix="/llm") 

# Placeholder for future router imports