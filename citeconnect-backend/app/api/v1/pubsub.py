"""
Pub/Sub endpoint for triggering ground truth initialization.
This endpoint receives notifications when papers are updated and triggers ground truth processing.
"""
from fastapi import APIRouter, BackgroundTasks, HTTPException, Request, Header
from pydantic import BaseModel
from typing import Optional
import asyncio
import hashlib
import hmac
from datetime import datetime, timedelta

from app.utils.logger import get_logger
from app.config import settings

logger = get_logger(__name__)

# Create router without prefix (prefix will be added in main.py)
router = APIRouter(tags=["pubsub"])

# Simple in-memory rate limiting
_last_trigger_time: Optional[datetime] = None
_trigger_lock = asyncio.Lock()
TRIGGER_COOLDOWN_SECONDS = 10  # 10 seconds for DEV/testing (change to 300 for production)


class PubSubMessage(BaseModel):
    """Pub/Sub message format"""
    message: dict
    subscription: Optional[str] = None


class GroundTruthTrigger(BaseModel):
    """Trigger payload for ground truth initialization"""
    trigger_type: str  # "manual", "pipeline", "scheduled"
    papers_count: Optional[int] = None
    domains: Optional[list[str]] = None
    triggered_by: Optional[str] = None


async def run_ground_truth_initialization():
    """
    Run ground truth initialization in background.
    This imports and runs the initialization script.
    
    CRITICAL: Manages its own database connection to avoid conflicts with FastAPI's pool.
    """
    from app.db.connection import DatabaseConnection
    
    # Create a dedicated connection for this background task
    background_db = DatabaseConnection()
    
    try:
        logger.info("Starting ground truth initialization from Pub/Sub trigger")
        
        # Connect with dedicated pool
        await background_db.connect()
        logger.info("Background task database connection established")
        
        # Import the initialization functions
        from scripts.initialize_ground_truth import (
            identify_ground_truth_papers,
            compute_ground_truth_relationships,
            identify_canonical_papers
        )
        
        # Temporarily replace the global db with our background db
        import scripts.initialize_ground_truth as gt_module
        original_db = gt_module.db
        gt_module.db = background_db
        
        try:
            # Run initialization steps
            gt_count = await identify_ground_truth_papers()
            logger.info("Identified ground truth papers", count=gt_count)
            
            await compute_ground_truth_relationships()
            logger.info("Computed ground truth relationships")
            
            # Pass the count to canonical papers function
            await identify_canonical_papers(gt_count)
            logger.info("Identified canonical papers")
            
            logger.info(
                "Ground truth initialization completed successfully",
                ground_truth_count=gt_count
            )
            
            return {
                "status": "success",
                "ground_truth_papers": gt_count,
                "timestamp": datetime.now().isoformat()
            }
        finally:
            # Restore original db
            gt_module.db = original_db
        
    except Exception as e:
        logger.error(
            "Ground truth initialization failed",
            error=str(e),
            exc_info=True
        )
        return {
            "status": "failed",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }
    finally:
        # Always clean up the background connection
        try:
            await background_db.disconnect()
            logger.info("Background task database connection closed")
        except Exception as e:
            logger.error("Error closing background connection", error=str(e))


@router.post("/trigger-ground-truth")
async def trigger_ground_truth(
    background_tasks: BackgroundTasks,
    trigger: GroundTruthTrigger,
    x_webhook_secret: Optional[str] = Header(None)
):
    """
    Endpoint to manually trigger ground truth initialization.
    Can be called from Airflow DAG or other services.
    
    Security: Validates webhook secret from header.
    Includes rate limiting to prevent spam from batch inserts.
    """
    global _last_trigger_time
    
    # Validate webhook secret if configured
    if settings.WEBHOOK_SECRET:
        if not x_webhook_secret:
            raise HTTPException(status_code=401, detail="Webhook secret required")
        
        if not hmac.compare_digest(x_webhook_secret, settings.WEBHOOK_SECRET):
            raise HTTPException(status_code=403, detail="Invalid webhook secret")
    
    # Rate limiting to prevent spam from batch inserts
    async with _trigger_lock:
        now = datetime.now()
        
        if _last_trigger_time is not None:
            time_since_last = (now - _last_trigger_time).total_seconds()
            
            if time_since_last < TRIGGER_COOLDOWN_SECONDS:
                logger.info(
                    "Ground truth trigger rate limited",
                    time_since_last=time_since_last,
                    cooldown=TRIGGER_COOLDOWN_SECONDS
                )
                return {
                    "status": "rate_limited",
                    "message": f"Ground truth initialization already running or recently completed. Cooldown: {TRIGGER_COOLDOWN_SECONDS}s",
                    "time_since_last_trigger": time_since_last,
                    "retry_after": TRIGGER_COOLDOWN_SECONDS - time_since_last
                }
        
        # Update last trigger time
        _last_trigger_time = now
    
    logger.info(
        "Ground truth trigger received",
        trigger_type=trigger.trigger_type,
        triggered_by=trigger.triggered_by,
        papers_count=trigger.papers_count
    )
    
    # Run in background to avoid timeout
    background_tasks.add_task(run_ground_truth_initialization)
    
    return {
        "status": "processing",
        "message": "Ground truth initialization started in background",
        "trigger_type": trigger.trigger_type,
        "timestamp": datetime.now().isoformat()
    }


@router.post("/gcs-notification")
async def gcs_notification(
    request: Request,
    background_tasks: BackgroundTasks
):
    """
    Endpoint for GCS Pub/Sub notifications.
    Triggered when files are uploaded to GCS bucket.
    
    Note: This receives notifications from GCS, not directly from Supabase.
    For Supabase triggers, use the webhook endpoint instead.
    """
    global _last_trigger_time
    
    try:
        body = await request.json()
        
        logger.info(
            "GCS notification received",
            body=body
        )
        
        # Parse GCS notification
        message = body.get("message", {})
        attributes = message.get("attributes", {})
        
        # Check if this is a papers-related upload
        object_name = attributes.get("objectId", "")
        bucket_name = attributes.get("bucketId", "")
        
        # Only trigger if it's in the raw or processed data folders
        if "raw/" in object_name or "processed/" in object_name:
            # Check rate limiting
            async with _trigger_lock:
                now = datetime.now()
                
                if _last_trigger_time is not None:
                    time_since_last = (now - _last_trigger_time).total_seconds()
                    
                    if time_since_last < TRIGGER_COOLDOWN_SECONDS:
                        return {
                            "status": "rate_limited",
                            "message": "Ground truth initialization recently triggered"
                        }
                
                _last_trigger_time = now
            
            logger.info(
                "Triggering ground truth initialization from GCS upload",
                bucket=bucket_name,
                object=object_name
            )
            
            background_tasks.add_task(run_ground_truth_initialization)
            
            return {
                "status": "processing",
                "message": "Ground truth initialization triggered from GCS upload"
            }
        else:
            return {
                "status": "ignored",
                "message": "Upload not in relevant folder"
            }
            
    except Exception as e:
        logger.error(
            "Error processing GCS notification",
            error=str(e),
            exc_info=True
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/supabase-webhook")
async def supabase_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    x_supabase_signature: Optional[str] = Header(None)
):
    """
    Webhook endpoint for Supabase database triggers.
    Triggered when papers table is updated.
    
    This is the recommended approach for triggering from Supabase.
    """
    global _last_trigger_time
    
    try:
        body = await request.json()
        
        # Validate Supabase webhook signature if configured
        if settings.SUPABASE_WEBHOOK_SECRET and x_supabase_signature:
            # Verify signature
            payload = await request.body()
            expected_signature = hmac.new(
                settings.SUPABASE_WEBHOOK_SECRET.encode(),
                payload,
                hashlib.sha256
            ).hexdigest()
            
            if not hmac.compare_digest(x_supabase_signature, expected_signature):
                raise HTTPException(status_code=403, detail="Invalid signature")
        
        logger.info(
            "Supabase webhook received",
            event_type=body.get("type"),
            table=body.get("table"),
            record_count=len(body.get("record", {}))
        )
        
        # Check if this is a papers table update
        if body.get("table") == "papers":
            event_type = body.get("type")  # INSERT, UPDATE, DELETE
            
            # Only trigger on INSERT or UPDATE
            if event_type in ["INSERT", "UPDATE"]:
                # Check rate limiting
                async with _trigger_lock:
                    now = datetime.now()
                    
                    if _last_trigger_time is not None:
                        time_since_last = (now - _last_trigger_time).total_seconds()
                        
                        if time_since_last < TRIGGER_COOLDOWN_SECONDS:
                            return {
                                "status": "rate_limited",
                                "message": "Ground truth initialization recently triggered"
                            }
                    
                    _last_trigger_time = now
                
                logger.info(
                    "Triggering ground truth initialization from Supabase update",
                    event_type=event_type
                )
                
                background_tasks.add_task(run_ground_truth_initialization)
                
                return {
                    "status": "processing",
                    "message": f"Ground truth initialization triggered from {event_type}",
                    "timestamp": datetime.now().isoformat()
                }
        
        return {
            "status": "ignored",
            "message": "Event not relevant for ground truth initialization"
        }
        
    except Exception as e:
        logger.error(
            "Error processing Supabase webhook",
            error=str(e),
            exc_info=True
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """Health check endpoint for Pub/Sub system"""
    return {
        "status": "healthy",
        "service": "ground-truth-pubsub",
        "timestamp": datetime.now().isoformat(),
        "rate_limiting": {
            "cooldown_seconds": TRIGGER_COOLDOWN_SECONDS,
            "last_trigger": _last_trigger_time.isoformat() if _last_trigger_time else None
        }
    }