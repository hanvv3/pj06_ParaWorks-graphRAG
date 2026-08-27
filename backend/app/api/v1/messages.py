from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.app.core.demo_auth import DemoUser, get_demo_user
from backend.app.db.session import get_db
from backend.app.messages.service import (
    append_message,
    create_review_item_from_message,
    get_channel,
    get_message_record,
    list_channels,
    list_messages,
)
from backend.app.schemas.messages import (
    ChannelMessagesResponse,
    MessageChannelsResponse,
    MessageCreateRequest,
    MessageResponse,
)

router = APIRouter(prefix='/messages', tags=['messages'])


def require_channel(db: Session, channel_id: str) -> dict:
    channel = get_channel(db, channel_id)
    if channel is None:
        raise HTTPException(status_code=404, detail='message channel not found')
    return channel


@router.get('/channels', response_model=MessageChannelsResponse)
def get_message_channels(
    db: Annotated[Session, Depends(get_db)],
) -> dict:
    return {'channels': list_channels(db)}


@router.get('/channels/{channel_id}/messages', response_model=ChannelMessagesResponse)
def get_channel_messages(
    channel_id: str,
    db: Annotated[Session, Depends(get_db)],
) -> dict:
    channel = require_channel(db, channel_id)
    return {'channel': channel, 'messages': list_messages(db, channel_id)}


@router.post('/channels/{channel_id}/messages', response_model=MessageResponse)
def create_channel_message(
    channel_id: str,
    request: MessageCreateRequest,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[DemoUser, Depends(get_demo_user)],
) -> dict:
    require_channel(db, channel_id)
    return append_message(db, channel_id, request.body, user)


@router.post('/messages/{message_id}/send-to-review')
def send_message_to_review(
    message_id: str,
    db: Annotated[Session, Depends(get_db)],
) -> dict:
    message = get_message_record(db, message_id)
    if message is None:
        raise HTTPException(status_code=404, detail='message not found')

    review_item = create_review_item_from_message(db, message)
    return {
        'id': review_item.id,
        'item_type': review_item.item_type,
        'payload': review_item.payload,
        'source_links': review_item.source_links,
        'source_snippets': review_item.source_snippets,
        'confidence_score': review_item.confidence_score,
        'permission_level': review_item.permission_level,
        'status': review_item.status,
    }
