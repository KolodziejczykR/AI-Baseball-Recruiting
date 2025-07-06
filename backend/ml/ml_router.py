from fastapi import APIRouter
from pydantic import BaseModel
from typing import Dict, Any

# Import the position-specific routers
from .infielder_router import router as infielder_router
from .outfielder_router import router as outfielder_router

router = APIRouter()

# Include the position-specific routers
router.include_router(infielder_router, prefix="/infielder", tags=["infielder"])
router.include_router(outfielder_router, prefix="/outfielder", tags=["outfielder"])
