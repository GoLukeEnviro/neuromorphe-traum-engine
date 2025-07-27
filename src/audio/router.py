from typing import List
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse

from .service import AudioProcessingService
from .dependencies import get_audio_service
from .schemas import (
    AudioUploadRequest,
    AudioProcessingResponse,
    EmbeddingResponse,
    AudioFileInfo,
    ProcessingStatus
)

router = APIRouter(prefix="/audio", tags=["audio"])


@router.post("/upload", response_model=AudioProcessingResponse)
async def upload_audio(
    file: UploadFile = File(...),
    category: str = Form(None),
    bpm: int = Form(None),
    service: AudioProcessingService = Depends(get_audio_service)
):
    """Upload and process audio file"""
    # Validate file type
    if not file.content_type or not file.content_type.startswith('audio/'):
        raise HTTPException(
            status_code=400, 
            detail="File must be an audio file"
        )
    
    # Validate file size (max 50MB)
    content = await file.read()
    if len(content) > 50 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="File too large. Maximum size is 50MB"
        )
    
    try:
        # Create upload request
        upload_request = AudioUploadRequest(
            filename=file.filename,
            category=category,
            bpm=bpm
        )
        
        # Save file
        file_id = await service.save_uploaded_file(content, upload_request)
        
        # Start processing
        response = await service.process_audio_file(file_id)
        
        return response
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing audio file: {str(e)}"
        )


@router.get("/files", response_model=List[str])
async def list_audio_files(
    service: AudioProcessingService = Depends(get_audio_service)
):
    """List all audio file IDs"""
    try:
        files = await service.list_audio_files()
        return files
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error listing audio files: {str(e)}"
        )


@router.get("/files/{file_id}", response_model=AudioFileInfo)
async def get_audio_file_info(
    file_id: str,
    service: AudioProcessingService = Depends(get_audio_service)
):
    """Get information about a specific audio file"""
    try:
        info = await service.get_audio_info(file_id)
        if not info:
            raise HTTPException(
                status_code=404,
                detail="Audio file not found"
            )
        return info
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting audio file info: {str(e)}"
        )


@router.post("/files/{file_id}/process", response_model=AudioProcessingResponse)
async def process_audio_file(
    file_id: str,
    service: AudioProcessingService = Depends(get_audio_service)
):
    """Process audio file and generate CLAP embedding"""
    try:
        # Check if file exists
        info = await service.get_audio_info(file_id)
        if not info:
            raise HTTPException(
                status_code=404,
                detail="Audio file not found"
            )
        
        # Process file
        response = await service.process_audio_file(file_id)
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing audio file: {str(e)}"
        )


@router.get("/files/{file_id}/embedding")
async def get_audio_embedding(
    file_id: str,
    service: AudioProcessingService = Depends(get_audio_service)
):
    """Get CLAP embedding for audio file"""
    try:
        embedding = await service.get_embedding(file_id)
        if embedding is None:
            raise HTTPException(
                status_code=404,
                detail="Embedding not found. Process the audio file first."
            )
        
        return {
            "file_id": file_id,
            "embedding_shape": embedding.shape,
            "embedding_size": embedding.size,
            "embedding": embedding.tolist()  # Convert to list for JSON serialization
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting embedding: {str(e)}"
        )


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "audio"}