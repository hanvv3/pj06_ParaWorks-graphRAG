from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.app.agents.rag_orchestrator_agent.service import (
    candidates_from_vector_matches,
    citation_from_candidate,
    filter_live_serving_candidates,
    retrieve_matching_evidence_candidates,
)
from backend.app.core.config import Settings, get_settings
from backend.app.core.demo_auth import DemoUser, get_demo_user
from backend.app.db.session import get_db
from backend.app.permissions.service import can_access_permission
from backend.app.rag.search_store import build_pgvector_search_store
from backend.app.schemas.search import SearchRequest

router = APIRouter(prefix='/search', tags=['search'])
DbSession = Annotated[Session, Depends(get_db)]
CurrentUser = Annotated[DemoUser, Depends(get_demo_user)]
AppSettings = Annotated[Settings, Depends(get_settings)]


@router.post('')
def search_knowledge(
    request: SearchRequest,
    db: DbSession,
    user: CurrentUser,
    settings: AppSettings,
) -> dict:
    vector_store = _pgvector_search_store(db=db, settings=settings)
    if vector_store is None:
        candidates = retrieve_matching_evidence_candidates(db=db, question=request.query)
        candidates = filter_live_serving_candidates(db=db, candidates=candidates)
        visible_candidates = [
            candidate for candidate in candidates if can_access_permission(user, candidate.permission_level)
        ]
        hidden_matches = len(candidates) - len(visible_candidates)
        retrieval_backend = 'deterministic_lexical'
        embedding_query_call = False
    else:
        vector_result = vector_store.search(query=request.query, user=user, limit=5)
        visible_candidates = candidates_from_vector_matches(vector_result.matches)
        visible_candidates = filter_live_serving_candidates(
            db=db, candidates=visible_candidates
        )
        hidden_matches = vector_result.hidden_match_count
        retrieval_backend = 'pgvector'
        embedding_query_call = True

    response = {
        'retrieval_backend': retrieval_backend,
        'cost_policy': {
            'embedding_query_call': embedding_query_call,
            'paid_llm_call': False,
            'requires_pgvector_flag': True,
        },
        'hidden_match_count': hidden_matches,
        'results': [
            {
                'id': int(candidate.metadata.get('chunk_id') or index + 1),
                'source_id': candidate.source_id,
                'text': candidate.text,
                'source_snippet': candidate.source_snippet,
                'source_url': candidate.source_url,
                'source_type': candidate.metadata.get('source_type'),
                'permission_level': candidate.permission_level,
                'relevance_score': candidate.relevance_score,
                'matched_terms': candidate.matched_terms,
                'citation': citation_from_candidate(candidate),
                'parser_status': candidate.metadata.get('parser_status'),
                'parser_status_reason': candidate.metadata.get('parser_status_reason'),
                'revision_id': candidate.metadata.get('revision_id'),
            }
            for index, candidate in enumerate(visible_candidates)
        ]
    }
    if hidden_matches:
        response['permission_notice'] = 'Some sources may be hidden by permissions.'
    return response


def _pgvector_search_store(*, db: Session, settings: Settings):
    return build_pgvector_search_store(db=db, settings=settings)
