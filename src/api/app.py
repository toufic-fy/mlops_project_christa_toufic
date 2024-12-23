from fastapi import FastAPI
from api.endpoints import router as api_router

# Create FastAPI app instance
app = FastAPI(
    title="Email Classifier API",
    description="API for health check and email classification",
    version="1.0.0"
)

# Include the API router
app.include_router(api_router)
