from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.ml.ml_router import router as ml_router
from backend.ml.infielder_router import router as infielder_router
from backend.ml.outfielder_router import router as outfielder_router
from backend.scraper.team_scraper import router as scraper_router

app = FastAPI(title="AI/ML Baseball Recruitment Backend")

# CORS middleware for local frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import and include the ML routers
app.include_router(ml_router, prefix="/predict")
app.include_router(infielder_router, prefix="/infielder")
app.include_router(outfielder_router, prefix="/outfielder")
app.include_router(scraper_router)

@app.get("/")
def read_root():
    return {"message": "Welcome to the AI/ML Baseball Recruitment API!"}

@app.get("/ping")
def health_check():
    return {"status": "ok"}

# from backend.api.llm import router as llm_router
# app.include_router(llm_router, prefix="/llm") 

# Placeholder for future router imports