from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass, replace
from decimal import Decimal

import pytest
from langchain_core.runnables import Runnable
from sqlalchemy.orm import Session

from backend.app.admin.auto_review_keys import fingerprint_key_material_verifier
from backend.app.agent_runtime.rag_v2_identity import (
    SecurityScope,
    security_scope_fingerprint,
)
from backend.app.core.config import Settings
from backend.app.ingestion.source_content_signature import (
    SERVER_CHUNK_POLICY_VERSION,
    SERVER_PARSER_POLICY_VERSION,
    SERVER_PARSER_VERSION,
)
from backend.app.models import (
    DecisionRecord,
    Document,
    DocumentChunk,
    DocumentParserRun,
    DocumentVersion,
    RagLexicalServingProjection,
    RagServingCorpusGeneration,
    ReviewItem,
    Source,
)
from backend.app.rag.keyword_retriever import KeywordEvidenceRetriever
from backend.app.rag.lexical_projection import (
    score_rag_lexical_candidate,
    tokenize_rag_lexical_query,
)
from backend.app.rag.retrieval import (
    ClassifiedRetrievalCandidate,
    PreparedPaidCallBudget,
    RagRetrieverRegistry,
    RetrievalRequest,
    SanitizedRetrievalTrace,
    StrictProviderUsage,
)
from backend.app.rag.search_store import (
    POSTGRES_KEYWORD_SEARCH_SQL,
    KeywordSearchTimeoutError,
    SqlAlchemyKeywordSearchStore,
)
from backend.app.rag.serving_contracts import (
    EvidenceAccessClassification,
    RawChunkProvenance,
    RawServingVersionEnvelope,
    ServingEvidence,
)
from backend.app.rag.serving_generation import (
    RAG_COSINE_POLICY_VERSION,
    RAG_INDEX_POLICY_VERSION,
    RAG_LEXICAL_COMPAT_VERSION,
)
from backend.app.rag.source_observations import CanonicalSourceObservationResolver
from backend.app.rag.trusted_evidence import (
    ServingEvidenceResolver,
    TrustedEvidenceAuthorizer,
    TrustedServingEnvelopeResolver,
)


class _FakeStore:
    def __init__(
        self,
        candidates: tuple[ClassifiedRetrievalCandidate, ...] = (),
        *,
        failure: Exception | None = None,
    ) -> None:
        self.candidates = candidates
        self.failure = failure
        self.calls = 0

    def search(
        self,
        request: RetrievalRequest,
    ) -> tuple[ClassifiedRetrievalCandidate, ...]:
        del request
        self.calls += 1
        if self.failure is not None:
            raise self.failure
        return self.candidates


@dataclass(frozen=True, slots=True)
class _OracleApprovalLink:
    workspace_scope_id: str
    active: bool
    resolution_source: str
    child_source_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _OracleResourceRow:
    serving_document_id: str
    serving_kind: str
    project_key: str | None
    raw_source_id: int | None
    score: float
    links: tuple[_OracleApprovalLink, ...]


def _independent_resource_coarse_eligible(
    row: _OracleResourceRow,
    scope: SecurityScope,
) -> bool:
    allowed_projects = {
        value.removeprefix('project_key:') for value in scope.project_constraints
    }
    allowed_sources = {
        int(value.removeprefix('source_pk:')) for value in scope.source_constraints
    }
    if row.serving_kind == 'raw_chunk':
        return bool(
            not allowed_projects
            and (not allowed_sources or row.raw_source_id in allowed_sources)
        )
    if row.serving_kind != 'trusted_knowledge':
        return False
    if allowed_projects and row.project_key not in allowed_projects:
        return False
    active_links = tuple(link for link in row.links if link.active)
    if not active_links:
        return not allowed_sources
    return any(
        link.workspace_scope_id == scope.workspace_scope_id
        and link.resolution_source in {'human', 'auto_policy'}
        and bool(link.child_source_ids)
        and (
            not allowed_sources
            or all(source_id in allowed_sources for source_id in link.child_source_ids)
        )
        for link in active_links
    )


def _independent_top_50(
    rows: tuple[_OracleResourceRow, ...],
    scope: SecurityScope,
) -> tuple[_OracleResourceRow, ...]:
    eligible = tuple(
        row for row in rows if _independent_resource_coarse_eligible(row, scope)
    )
    return tuple(
        sorted(
            eligible,
            key=lambda row: (
                0 if row.serving_kind == 'trusted_knowledge' else 1,
                -row.score,
                row.serving_document_id,
            ),
        )[:50]
    )


def _settings() -> Settings:
    return Settings(
        agent_runtime_fingerprint_secret='keyword-test-secret',
        agent_runtime_fingerprint_key_version='keyword-test-v1',
    )


def _scope(**overrides: object) -> SecurityScope:
    values: dict[str, object] = {
        'contract_version': 'rag-security-scope:v1',
        'principal_subject': 'user-1',
        'workspace_scope_id': 'workspace-1',
        'resource_scope_mode': 'all_current_scope',
        'project_constraints': (),
        'source_constraints': (),
        'allowed_permission_levels': ('public',),
        'auth_policy_version': 'demo-auth:v1',
        'permission_policy_version': 'rag-permission-policy:v1',
    }
    values.update(overrides)
    return SecurityScope(**values)  # type: ignore[arg-type]


def _request(
    *,
    visible_limit: int = 5,
    fingerprint: str | None = None,
    query: str = 'alpha roadmap',
    scope: SecurityScope | None = None,
) -> RetrievalRequest:
    settings = _settings()
    resolved_scope = scope or _scope()
    return RetrievalRequest(
        retrieval_query_text=query,
        security_scope=resolved_scope,
        security_scope_fingerprint=(
            fingerprint
            if fingerprint is not None
            else security_scope_fingerprint(resolved_scope, settings=settings)
        ),
        query_embedding_result=None,
        candidate_scan_limit=50,
        visible_limit=visible_limit,  # type: ignore[arg-type]
        relevance_policy_version='rag-retrieval-policy:v2.0',
    )


def _seed_sqlite_raw_projection(db: Session) -> tuple[Source, DocumentChunk]:
    settings = _settings()
    body = 'Exact raw observation with %_Case and Unicode 한글'
    signature = 'a' * 64
    source = Source(
        source_type='gmail',
        source_id='gmail:keyword-task-6',
        source_url='https://mail.example.test/messages/keyword-task-6',
        title='Raw TITLE %_Case 한글',
        author='owner@example.test',
        permission_level='internal',
        raw_metadata={},
        server_content_signature_schema='server-source-content:v1',
        server_content_signature=signature,
    )
    db.add(source)
    db.flush()
    document = Document(
        source_id=source.id,
        title=source.title,
        current_version='v1',
    )
    db.add(document)
    db.flush()
    version = DocumentVersion(document_id=document.id, version='v1', body=body)
    db.add(version)
    db.flush()
    parser_run = DocumentParserRun(
        document_id=document.id,
        document_version_id=version.id,
        source_id=source.id,
        parser_name='server_gmail_source_event',
        parser_status='parsed',
        parser_status_reason=None,
        mime_type='message/rfc822',
        document_version_label='v1',
        revision_id='revision-1',
        content_signature=signature,
        server_content_signature_schema='server-source-content:v1',
        server_content_signature=signature,
        parser_policy_version=SERVER_PARSER_POLICY_VERSION,
        parser_version=SERVER_PARSER_VERSION,
        chunk_policy_version=SERVER_CHUNK_POLICY_VERSION,
        chunk_count=1,
    )
    db.add(parser_run)
    db.flush()
    chunk = DocumentChunk(
        version_id=version.id,
        source_id=source.id,
        parser_run_id=parser_run.id,
        chunk_index=0,
        text=body,
        source_snippet=body,
        permission_level='internal',
        metadata_={},
    )
    db.add(chunk)
    document.current_document_version_id = version.id
    db.flush()
    observation = CanonicalSourceObservationResolver(
        db=db,
        settings=settings,
    ).resolve_for_index(chunk.id)
    assert observation is not None
    evidence = observation.evidence
    db.add(
        RagServingCorpusGeneration(
            id=1,
            corpus_generation=1,
            vector_index_generation=0,
            embedding_model=settings.openai_embedding_model,
            embedding_dimensions=settings.openai_embedding_dimensions,
            index_policy_version=RAG_INDEX_POLICY_VERSION,
            pgvector_cosine_policy_version=RAG_COSINE_POLICY_VERSION,
            fingerprint_key_version=settings.agent_runtime_fingerprint_key_version,
            fingerprint_key_material_verifier=fingerprint_key_material_verifier(
                settings.agent_runtime_fingerprint_secret
            ),
        )
    )
    db.add(
        RagLexicalServingProjection(
            corpus_generation_id=1,
            corpus_generation=1,
            serving_document_id=evidence.serving_document_id,
            serving_kind=evidence.serving_kind,
            support_mode=evidence.support_mode,
            effective_permission=evidence.effective_permission,
            serving_identity_hmac=evidence.serving_identity_hmac,
            serving_version_fingerprint=evidence.serving_version_fingerprint,
            model_content_hmac=evidence.model_content_hmac,
            canonical_citation_projection_hmac=(
                evidence.canonical_citation_projection_hmac
            ),
            title_lower=evidence.title.lower(),
            searchable_lower=f'{evidence.title}\n{evidence.model_content}'.lower(),
            lexical_contract_version=RAG_LEXICAL_COMPAT_VERSION,
            fingerprint_key_version=settings.agent_runtime_fingerprint_key_version,
            fingerprint_key_material_verifier=fingerprint_key_material_verifier(
                settings.agent_runtime_fingerprint_secret
            ),
        )
    )
    db.commit()
    return source, chunk


def _seed_sqlite_trusted_projection(db: Session) -> DecisionRecord:
    settings = _settings()
    review = ReviewItem(
        item_type='decision_record',
        payload={'title': 'Trusted hidden decision'},
        source_links=['https://knowledge.example.test/reviews/trusted-hidden'],
        source_snippets=['Trusted hidden citation'],
        confidence_score=0.97,
        permission_level='internal',
        status='approved',
        resolution_source='human',
    )
    db.add(review)
    db.flush()
    decision = DecisionRecord(
        title='TrustedNeedle decision',
        decision_summary='Canonical trusted hidden fact.',
        source_links=list(review.source_links),
        source_snippets=list(review.source_snippets),
        confidence_score=0.97,
        permission_level='internal',
        review_status='approved',
        source_review_item_id=review.id,
    )
    db.add(decision)
    db.flush()
    envelope = TrustedServingEnvelopeResolver(
        db=db,
        settings=settings,
    ).resolve_for_index('decision_record', decision.id)
    assert envelope is not None
    evidence = envelope.evidence
    db.add(
        RagLexicalServingProjection(
            corpus_generation_id=1,
            corpus_generation=1,
            serving_document_id=evidence.serving_document_id,
            serving_kind=evidence.serving_kind,
            support_mode=evidence.support_mode,
            effective_permission=evidence.effective_permission,
            serving_identity_hmac=evidence.serving_identity_hmac,
            serving_version_fingerprint=evidence.serving_version_fingerprint,
            model_content_hmac=evidence.model_content_hmac,
            canonical_citation_projection_hmac=(
                evidence.canonical_citation_projection_hmac
            ),
            title_lower=evidence.title.lower(),
            searchable_lower=f'{evidence.title}\n{evidence.model_content}'.lower(),
            lexical_contract_version=RAG_LEXICAL_COMPAT_VERSION,
            fingerprint_key_version=settings.agent_runtime_fingerprint_key_version,
            fingerprint_key_material_verifier=fingerprint_key_material_verifier(
                settings.agent_runtime_fingerprint_secret
            ),
        )
    )
    db.commit()
    return decision


def _evidence(
    identifier: int,
    *,
    kind: str = 'raw_chunk',
    permission: str = 'public',
) -> ServingEvidence:
    document_id = (
        f'chunk:{identifier}'
        if kind == 'raw_chunk'
        else f'decision_record:{identifier}'
    )
    version = RawServingVersionEnvelope(
        serving_document_id=document_id,
        source_row_id=identifier,
        public_source_id=f'source-{identifier}',
        document_id=identifier,
        document_version_id=identifier,
        current_document_version_id=identifier,
        document_chunk_id=identifier,
        parser_run_id=identifier,
        external_revision=None,
        server_content_signature_schema='server-source-content:v1',
        server_content_signature='a' * 64,
        parser_policy_version='parser-policy:v1',
        parser_version='parser:v1',
        chunk_policy_version='chunk:v1',
        model_content_hmac='b' * 64,
        canonical_citation_projection_hmac='c' * 64,
        effective_permission='public',
    )
    provenance = RawChunkProvenance(branch='raw_chunk', raw_version=version)
    return ServingEvidence(
        serving_document_id=document_id,
        serving_kind=kind,  # type: ignore[arg-type]
        public_source_id=f'source-{identifier}',
        public_source_type='gmail',
        support_mode=(
            'source_observation' if kind == 'raw_chunk' else 'trusted_fact'
        ),
        model_content=f'content {identifier}',
        title=f'title {identifier}',
        effective_permission=permission,  # type: ignore[arg-type]
        serving_identity_hmac=f'{identifier:064x}'[-64:],
        serving_version_fingerprint='d' * 64,
        model_content_hmac='b' * 64,
        canonical_citation_projection_hmac='c' * 64,
        version_envelope=version,
        provenance=provenance,
    )


def _candidate(
    identifier: int,
    *,
    kind: str = 'raw_chunk',
    score: float = 1.0,
    visibility: str = 'visible',
    global_eligibility: str = 'eligible',
    resource_scope: str = 'in_scope',
) -> ClassifiedRetrievalCandidate:
    return ClassifiedRetrievalCandidate(
        evidence=_evidence(identifier, kind=kind),
        relevance_score=score,
        matched_terms=('alpha',),
        access=EvidenceAccessClassification(
            global_eligibility=global_eligibility,  # type: ignore[arg-type]
            resource_scope=resource_scope,  # type: ignore[arg-type]
            permission_visibility=visibility,  # type: ignore[arg-type]
        ),
    )


def test_keyword_retriever_is_actual_langchain_runnable_with_immutable_contracts() -> None:
    retriever = KeywordEvidenceRetriever(store=_FakeStore(), settings=_settings())

    assert isinstance(retriever, Runnable)
    request = _request()
    with pytest.raises(FrozenInstanceError):
        request.visible_limit = 8  # type: ignore[misc]

    result = retriever.invoke(request)
    assert result.visible == ()
    assert not hasattr(retriever, 'last_result')
    assert not hasattr(retriever, '_last_result')
    with pytest.raises(FrozenInstanceError):
        result.hidden_match_count = 1  # type: ignore[misc]


def test_scope_fingerprint_mismatch_stops_query_db_scorer_and_provider_sentinels() -> None:
    calls = {'query': 0, 'db': 0, 'scorer': 0, 'provider': 0}

    class SentinelStore:
        def search(self, request: RetrievalRequest):
            del request
            for name in calls:
                calls[name] += 1
            return ()

    retriever = KeywordEvidenceRetriever(store=SentinelStore(), settings=_settings())

    with pytest.raises(ValueError, match='serialized security scope fingerprint'):
        retriever.invoke(_request(fingerprint='0' * 64))
    assert calls == {'query': 0, 'db': 0, 'scorer': 0, 'provider': 0}


@pytest.mark.parametrize(
    ('question', 'expected'),
    (
        ('Alpha,beta.gamma', ('alpha', 'beta', 'gamma')),
        ('ab abc ABÇ', ('abc', 'abç')),
        ('abc abc', ('abc', 'abc')),
        ('100% a_b', ('100%', 'a_b')),
        ('x!y under_score', ('x!y', 'under_score')),
        ('\u2003Alpha\u00a0BETA', ('alpha', 'beta')),
    ),
)
def test_keyword_tokenizer_preserves_exact_v1_semantics(
    question: str,
    expected: tuple[str, ...],
) -> None:
    assert tokenize_rag_lexical_query(question) == expected


def test_keyword_scorer_preserves_duplicate_phrase_title_and_binary64_rounding() -> None:
    score, terms = score_rag_lexical_candidate(
        question='Alpha alpha body',
        title='ALPHA title',
        text='body Alpha alpha body',
    )

    assert terms == ('alpha', 'alpha', 'body')
    assert score == round(1.0 + 1.0 + min(2 * 0.15, 0.45), 6)
    assert score > 0
    assert score_rag_lexical_candidate(
        question='nomatch', title='title', text='body'
    ) == (0.0, ())


def test_direct_1001_plus_terms_keep_every_duplicate_and_order() -> None:
    question = ' '.join(f'term{index}' for index in range(1002))
    terms = tokenize_rag_lexical_query(question)

    assert len(terms) == 1002
    assert terms[0] == 'term0'
    assert terms[-1] == 'term1001'


def test_two_stage_window_is_top_50_before_permission_and_visible_five() -> None:
    candidates = tuple(
        _candidate(index, kind='trusted_knowledge', score=1.0)
        for index in range(1, 46)
    ) + tuple(
        _candidate(
            index,
            kind='trusted_knowledge',
            score=1.0,
            visibility='denied_known',
        )
        for index in range(46, 51)
    ) + tuple(
        _candidate(index, score=3.0) for index in range(51, 61)
    )
    result = KeywordEvidenceRetriever(
        store=_FakeStore(candidates), settings=_settings()
    ).invoke(_request())

    assert len(result.visible) == 5
    assert all(item.evidence.serving_kind == 'trusted_knowledge' for item in result.visible)
    assert result.hidden_match_count == 5
    assert result.trace.candidate_window_count == 50


def test_answer_visible_limit_is_eight_and_hidden_count_caps_at_twenty() -> None:
    candidates = tuple(
        _candidate(index, kind='trusted_knowledge', visibility='denied_known')
        for index in range(1, 26)
    ) + tuple(
        _candidate(index, kind='trusted_knowledge') for index in range(26, 40)
    )
    result = KeywordEvidenceRetriever(
        store=_FakeStore(candidates), settings=_settings()
    ).invoke(_request(visible_limit=8))

    assert len(result.visible) == 8
    assert result.hidden_match_count == 20
    assert result.hidden_count_capped is True


def test_unknown_permission_and_out_of_scope_are_excluded_from_window_and_count() -> None:
    candidates = (
        _candidate(1, visibility='unknown_permission'),
        _candidate(2, resource_scope='out_of_scope'),
        _candidate(3, global_eligibility='ineligible'),
        _candidate(4, visibility='denied_known'),
        _candidate(5),
    )
    result = KeywordEvidenceRetriever(
        store=_FakeStore(candidates), settings=_settings()
    ).invoke(_request())

    assert tuple(item.evidence.serving_document_id for item in result.visible) == (
        'chunk:5',
    )
    assert result.hidden_match_count == 1
    assert result.trace.candidate_window_count == 2


def test_order_is_trusted_tier_then_score_then_stable_serving_id() -> None:
    candidates = (
        _candidate(3, kind='raw_chunk', score=9.0),
        _candidate(9, kind='trusted_knowledge', score=0.1),
        _candidate(2, kind='trusted_knowledge', score=2.0),
        _candidate(1, kind='trusted_knowledge', score=2.0),
    )
    result = KeywordEvidenceRetriever(
        store=_FakeStore(candidates), settings=_settings()
    ).invoke(_request())

    assert tuple(item.evidence.serving_document_id for item in result.visible) == (
        'decision_record:1',
        'decision_record:2',
        'decision_record:9',
        'chunk:3',
    )


def test_denied_identity_never_appears_in_public_result_or_trace(caplog) -> None:
    denied_id = 987654321
    result = KeywordEvidenceRetriever(
        store=_FakeStore(
            (
                _candidate(denied_id, visibility='denied_known'),
                _candidate(1),
            )
        ),
        settings=_settings(),
    ).invoke(_request())

    public_bytes = f'{result!r}\n{result.trace!r}\n{caplog.text}'
    assert str(denied_id) not in public_bytes
    assert result.hidden_match_count == 1


def test_timeout_raises_without_partial_result_or_mutable_side_channel() -> None:
    store = _FakeStore(
        (_candidate(1),),
        failure=KeywordSearchTimeoutError('keyword retrieval timed out'),
    )
    retriever = KeywordEvidenceRetriever(store=store, settings=_settings())

    with pytest.raises(KeywordSearchTimeoutError):
        retriever.invoke(_request())
    assert not hasattr(retriever, '_last_result')


def test_keyword_request_rejects_embedding_before_store_activity() -> None:
    store = _FakeStore()
    request = _request()

    with pytest.raises(ValueError, match='keyword retrieval does not accept'):
        KeywordEvidenceRetriever(store=store, settings=_settings()).invoke(
            replace(request, query_embedding_result=object())  # type: ignore[arg-type]
        )
    assert store.calls == 0


def test_registry_requires_explicit_known_unique_backend_names() -> None:
    registry = RagRetrieverRegistry()
    keyword = KeywordEvidenceRetriever(store=_FakeStore(), settings=_settings())
    registry.register('keyword', keyword)

    assert registry.resolve('keyword') is keyword
    with pytest.raises(ValueError, match='already registered'):
        registry.register('keyword', keyword)
    with pytest.raises(ValueError, match='unknown retrieval backend'):
        registry.register('other', keyword)  # type: ignore[arg-type]
    with pytest.raises(KeyError, match='not registered'):
        registry.resolve('pgvector')
    with pytest.raises(ValueError, match='unknown retrieval backend'):
        registry.resolve('other')  # type: ignore[arg-type]


def test_postgresql_query_scores_before_limit_and_uses_literal_complete_superset() -> None:
    sql = ' '.join(POSTGRES_KEYWORD_SEARCH_SQL.split())

    coarse = sql.index(
        'strpos(projection.searchable_lower, coarse_term.term) > 0'
    )
    scorer = sql.index('rag_python_lexical_score_v1')
    relevance = sql.index('scored.score > 0')
    limit = sql.index('LIMIT 50')
    assert coarse < scorer < relevance < limit
    assert 'strpos(projection.searchable_lower, coarse_term.term) > 0' in sql
    assert 'LIKE' not in sql.upper()
    assert 'statement_timeout' not in sql
    assert "approval.knowledge_type IN ('decision', 'decision_record')" in sql
    assert (
        'child.canonical_source_id '
        '<> ALL(CAST(:source_id_texts AS text[]))'
    ) in sql
    assert ':all_scope' not in sql
    assert 'approval.security_scope_id = :workspace_scope_id' in sql
    assert 'approval.active IS TRUE' in sql
    assert sql.count(
        'FROM trusted_knowledge_approval_links AS approval'
    ) == 2
    assert (
        'AND NOT EXISTS ( SELECT 1 '
        'FROM trusted_knowledge_approval_links AS approval'
    ) in sql


@pytest.mark.parametrize(
    'scope',
    (
        _scope(allowed_permission_levels=('public',)),
        _scope(
            resource_scope_mode='constrained',
            project_constraints=('project_key:project-a',),
            allowed_permission_levels=('public',),
        ),
    ),
)
def test_independent_resource_oracle_excludes_foreign_workspace_before_top_50(
    scope: SecurityScope,
) -> None:
    foreign = tuple(
        _OracleResourceRow(
            serving_document_id=f'history_event:{index}',
            serving_kind='trusted_knowledge',
            project_key='project-a',
            raw_source_id=None,
            score=3.0,
            links=(
                _OracleApprovalLink(
                    workspace_scope_id='workspace-foreign',
                    active=True,
                    resolution_source='human',
                    child_source_ids=(1,),
                ),
            ),
        )
        for index in range(1, 61)
    )
    valid = tuple(
        _OracleResourceRow(
            serving_document_id=f'history_event:{index}',
            serving_kind='trusted_knowledge',
            project_key='project-a',
            raw_source_id=None,
            score=1.0,
            links=(
                _OracleApprovalLink(
                    workspace_scope_id=scope.workspace_scope_id,
                    active=True,
                    resolution_source='auto_policy',
                    child_source_ids=(1,),
                ),
            ),
        )
        for index in range(101, 152)
    )

    window = _independent_top_50((*foreign, *valid), scope)

    assert len(window) == 50
    assert all(row.serving_document_id.startswith('history_event:1') for row in window)
    assert not any(row in foreign for row in window)


def test_independent_resource_oracle_keeps_explicit_and_legacy_branches_disjoint() -> None:
    all_scope = _scope(allowed_permission_levels=('public',))
    source_scope = _scope(
        resource_scope_mode='constrained',
        source_constraints=('source_pk:7',),
        allowed_permission_levels=('public',),
    )
    foreign_explicit = _OracleResourceRow(
        serving_document_id='history_event:1',
        serving_kind='trusted_knowledge',
        project_key='project-a',
        raw_source_id=None,
        score=2.0,
        links=(
            _OracleApprovalLink(
                workspace_scope_id='workspace-foreign',
                active=True,
                resolution_source='human',
                child_source_ids=(7,),
            ),
        ),
    )
    current_explicit = replace(
        foreign_explicit,
        links=(
            replace(
                foreign_explicit.links[0],
                workspace_scope_id=all_scope.workspace_scope_id,
            ),
        ),
    )
    legacy = replace(foreign_explicit, serving_document_id='history_event:2', links=())

    assert not _independent_resource_coarse_eligible(foreign_explicit, all_scope)
    assert _independent_resource_coarse_eligible(current_explicit, all_scope)
    assert _independent_resource_coarse_eligible(legacy, all_scope)
    assert not _independent_resource_coarse_eligible(legacy, source_scope)
    assert not _independent_resource_coarse_eligible(
        replace(current_explicit, links=(replace(current_explicit.links[0], child_source_ids=(8,)),)),
        source_scope,
    )


def test_sqlite_oracle_uses_canonical_projection_and_permission_second_stage(
    db_session: Session,
) -> None:
    source, chunk = _seed_sqlite_raw_projection(db_session)
    settings = _settings()
    store = SqlAlchemyKeywordSearchStore(db=db_session, settings=settings)
    denied_scope = _scope(allowed_permission_levels=('public',))

    denied = KeywordEvidenceRetriever(store=store, settings=settings).invoke(
        _request(query='%_Case %_Case', scope=denied_scope)
    )
    assert denied.visible == ()
    assert denied.hidden_match_count == 1

    visible_scope = _scope(allowed_permission_levels=('public', 'internal'))
    visible = KeywordEvidenceRetriever(store=store, settings=settings).invoke(
        _request(query='%_Case %_Case', scope=visible_scope)
    )
    assert tuple(item.evidence.serving_document_id for item in visible.visible) == (
        f'chunk:{chunk.id}',
    )
    assert visible.visible[0].matched_terms == ('%_case', '%_case')
    assert visible.visible[0].evidence.public_source_id == source.source_id


def test_trusted_denied_permission_stays_in_scope_and_counts_hidden(
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seed_sqlite_raw_projection(db_session)
    _seed_sqlite_trusted_projection(db_session)
    settings = _settings()
    store = SqlAlchemyKeywordSearchStore(db=db_session, settings=settings)
    denied_scope = _scope(allowed_permission_levels=('public',))
    observed_scopes: list[SecurityScope] = []
    original_classify_resource = (
        TrustedEvidenceAuthorizer.classify_resource_access
    )

    def record_exact_scope(self, scope, envelope):
        observed_scopes.append(scope)
        return original_classify_resource(self, scope, envelope)

    def forbidden_projection(*args, **kwargs):
        del args, kwargs
        raise AssertionError('denied trusted evidence must not project citation bytes')

    monkeypatch.setattr(
        TrustedEvidenceAuthorizer,
        'classify_resource_access',
        record_exact_scope,
    )
    monkeypatch.setattr(
        ServingEvidenceResolver,
        'resolve_candidate',
        forbidden_projection,
    )

    result = KeywordEvidenceRetriever(store=store, settings=settings).invoke(
        _request(query='TrustedNeedle', scope=denied_scope)
    )

    assert result.visible == ()
    assert result.hidden_match_count == 1
    assert result.trace.candidate_window_count == 1
    assert observed_scopes
    assert all(scope is denied_scope for scope in observed_scopes)
    assert denied_scope.allowed_permission_levels == ('public',)


def test_sqlite_oracle_applies_exact_source_scope_before_candidate_window(
    db_session: Session,
) -> None:
    source, _ = _seed_sqlite_raw_projection(db_session)
    settings = _settings()
    store = SqlAlchemyKeywordSearchStore(db=db_session, settings=settings)
    allowed = _scope(
        resource_scope_mode='constrained',
        source_constraints=(f'source_pk:{source.id}',),
        allowed_permission_levels=('public', 'internal'),
    )
    denied = replace(allowed, source_constraints=(f'source_pk:{source.id + 1}',))

    allowed_result = KeywordEvidenceRetriever(store=store, settings=settings).invoke(
        _request(query='Unicode 한글', scope=allowed)
    )
    denied_result = KeywordEvidenceRetriever(store=store, settings=settings).invoke(
        _request(query='Unicode 한글', scope=denied)
    )
    assert len(allowed_result.visible) == 1
    assert denied_result.visible == ()
    assert denied_result.hidden_match_count == 0
    assert denied_result.trace.candidate_window_count == 0


def test_sqlite_direct_1001_plus_terms_have_no_term_drop_or_refusal(
    db_session: Session,
) -> None:
    _seed_sqlite_raw_projection(db_session)
    settings = _settings()
    store = SqlAlchemyKeywordSearchStore(db=db_session, settings=settings)
    scope = _scope(allowed_permission_levels=('public', 'internal'))
    query = ' '.join([*(f'absent{index}' for index in range(1001)), '%_Case'])

    result = KeywordEvidenceRetriever(store=store, settings=settings).invoke(
        _request(query=query, scope=scope)
    )
    assert len(tokenize_rag_lexical_query(query)) == 1002
    assert len(result.visible) == 1
    assert result.visible[0].matched_terms == ('%_case',)


def test_paid_carriers_are_strict_frozen_values_without_mutable_defaults() -> None:
    usage = StrictProviderUsage(input_tokens=3, output_tokens=0, total_tokens=3)
    budget = PreparedPaidCallBudget(
        component='query_embedding',
        estimated_input_tokens=3,
        maximum_output_tokens=0,
        reserved_cost_usd=Decimal('0.000001'),
        cost_policy_snapshot_hmac='a' * 64,
        estimator_input_hmac='b' * 64,
    )
    trace = SanitizedRetrievalTrace(
        candidate_window_count=0,
        visible_count=0,
        hidden_match_count=0,
        provider_attempt_count=0,
        latency_ms=0,
        fallback_category=None,
    )

    assert usage.total_tokens == usage.input_tokens + usage.output_tokens
    assert budget.reserved_cost_usd == Decimal('0.000001')
    assert trace.provider_attempt_count == 0
    with pytest.raises(FrozenInstanceError):
        usage.total_tokens = 4  # type: ignore[misc]
