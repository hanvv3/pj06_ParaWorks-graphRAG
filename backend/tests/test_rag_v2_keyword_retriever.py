from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
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
    POSTGRES_KEYWORD_BIND_SPECS,
    POSTGRES_KEYWORD_RESOURCE_PREDICATE,
    POSTGRES_KEYWORD_SEARCH_SQL,
    KeywordBooleanOperator,
    KeywordBooleanPredicate,
    KeywordColumn,
    KeywordComparisonOperator,
    KeywordComparisonPredicate,
    KeywordConstantPredicate,
    KeywordExistence,
    KeywordExistsPredicate,
    KeywordLiteral,
    KeywordParameter,
    KeywordRelation,
    KeywordRule,
    KeywordSearchTimeoutError,
    KeywordSqlTruth,
    PostgresKeywordBindTarget,
    PostgresKeywordRelationRow,
    PostgresKeywordResourceCandidate,
    SqlAlchemyKeywordSearchStore,
    build_postgres_keyword_bind_values,
    build_postgres_keyword_search_sql,
    evaluate_postgres_keyword_predicate,
    render_postgres_keyword_predicate,
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


def _production_predicate_top_50(
    rows: tuple[PostgresKeywordResourceCandidate, ...],
    scope: SecurityScope,
) -> tuple[PostgresKeywordResourceCandidate, ...]:
    request = _request(scope=scope)
    parameters = build_postgres_keyword_bind_values(
        request=request,
        terms=tokenize_rag_lexical_query(request.retrieval_query_text),
        settings=_settings(),
    )
    eligible = tuple(
        row
        for row in rows
        if evaluate_postgres_keyword_predicate(
            POSTGRES_KEYWORD_RESOURCE_PREDICATE,
            candidate=row,
            bind_values=parameters,
        )
        is KeywordSqlTruth.TRUE
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


def _relation_row(
    relation: str,
    **columns: str | int | bool | None,
) -> PostgresKeywordRelationRow:
    return PostgresKeywordRelationRow(
        relation=relation,
        columns=tuple(columns.items()),
    )


def _resource_row(
    serving_document_id: str,
    *,
    serving_kind: str = 'trusted_knowledge',
    project_key: str | None = 'project-a',
    raw_source_id: int | None = None,
    score: float = 1.0,
    links: tuple[tuple[str, bool, str, tuple[int, ...]], ...] = (),
) -> PostgresKeywordResourceCandidate:
    document_kind, identifier_text = serving_document_id.split(':', 1)
    identifier = int(identifier_text)
    relation_rows: list[PostgresKeywordRelationRow] = []
    if serving_kind == 'raw_chunk' and raw_source_id is not None:
        relation_rows.append(
            _relation_row('document_chunks', id=identifier, source_id=raw_source_id)
        )
    knowledge_relations = {
        'decision_record': 'decision_records',
        'history_event': 'history_events',
        'timeline_event': 'timeline_events',
        'todo': 'todos',
    }
    if serving_kind == 'trusted_knowledge' and project_key is not None:
        relation_rows.append(
            _relation_row(
                knowledge_relations[document_kind],
                id=identifier,
                project_key=project_key,
            )
        )
    for approval_id, (workspace, active, resolution, children) in enumerate(
        links,
        start=1,
    ):
        relation_rows.append(
            _relation_row(
                'trusted_knowledge_approval_links',
                id=approval_id,
                knowledge_type=document_kind,
                knowledge_id=identifier,
                active=active,
                security_scope_id=workspace,
                resolution_source=resolution,
            )
        )
        relation_rows.extend(
            _relation_row(
                'trusted_knowledge_evidence_links',
                approval_link_id=approval_id,
                canonical_source_id=str(source_id),
            )
            for source_id in children
        )
    return PostgresKeywordResourceCandidate(
        serving_document_id=serving_document_id,
        serving_kind=serving_kind,
        score=score,
        relation_rows=tuple(relation_rows),
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
    assert sql.count('CAST(:project_keys AS text[]) IS NOT NULL') == 2
    assert sql.count('CAST(:source_ids AS bigint[]) IS NOT NULL') == 1
    assert sql.count('CAST(:source_id_texts AS text[]) IS NOT NULL') == 2
    assert (
        'approval.security_scope_id = CAST(:workspace_scope_id AS text)'
        in sql
    )
    assert 'approval.active IS TRUE' in sql
    assert sql.count(
        'FROM trusted_knowledge_approval_links AS approval'
    ) == 2
    assert (
        'AND NOT EXISTS (SELECT 1 '
        'FROM trusted_knowledge_approval_links AS approval'
    ) in sql
    assert build_postgres_keyword_search_sql(
        POSTGRES_KEYWORD_RESOURCE_PREDICATE
    ) == POSTGRES_KEYWORD_SEARCH_SQL
    assert render_postgres_keyword_predicate(
        POSTGRES_KEYWORD_RESOURCE_PREDICATE
    ) in POSTGRES_KEYWORD_SEARCH_SQL


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
def test_production_resource_predicate_excludes_foreign_workspace_before_top_50(
    scope: SecurityScope,
) -> None:
    foreign = tuple(
        _resource_row(
            f'history_event:{index}',
            project_key='project-a',
            score=3.0,
            links=(('workspace-foreign', True, 'human', (1,)),),
        )
        for index in range(1, 61)
    )
    valid = tuple(
        _resource_row(
            f'history_event:{index}',
            project_key='project-a',
            score=1.0,
            links=((scope.workspace_scope_id, True, 'auto_policy', (1,)),),
        )
        for index in range(101, 152)
    )

    window = _production_predicate_top_50((*foreign, *valid), scope)

    assert len(window) == 50
    assert all(row.serving_document_id.startswith('history_event:1') for row in window)
    assert not any(row in foreign for row in window)


def test_production_resource_predicate_keeps_explicit_and_legacy_branches_disjoint() -> None:
    all_scope = _scope(allowed_permission_levels=('public',))
    project_scope = _scope(
        resource_scope_mode='constrained',
        project_constraints=('project_key:project-a',),
        allowed_permission_levels=('public',),
    )
    source_scope = _scope(
        resource_scope_mode='constrained',
        source_constraints=('source_pk:7',),
        allowed_permission_levels=('public',),
    )
    foreign_explicit = _resource_row(
        'history_event:1',
        project_key='project-a',
        score=2.0,
        links=(('workspace-foreign', True, 'human', (7,)),),
    )
    current_explicit = _resource_row(
        'history_event:1',
        project_key='project-a',
        score=2.0,
        links=((all_scope.workspace_scope_id, True, 'human', (7,)),),
    )
    legacy = _resource_row('history_event:2', project_key='project-a')

    def truth(row, scope):
        request = _request(scope=scope)
        binds = build_postgres_keyword_bind_values(
            request=request,
            terms=tokenize_rag_lexical_query(request.retrieval_query_text),
            settings=_settings(),
        )
        return evaluate_postgres_keyword_predicate(
            POSTGRES_KEYWORD_RESOURCE_PREDICATE,
            candidate=row,
            bind_values=binds,
        )

    assert truth(foreign_explicit, all_scope) is KeywordSqlTruth.FALSE
    assert truth(current_explicit, all_scope) is KeywordSqlTruth.TRUE
    assert truth(current_explicit, project_scope) is KeywordSqlTruth.TRUE
    assert (
        truth(
            _resource_row(
                'history_event:1',
                project_key='project-foreign',
                links=((all_scope.workspace_scope_id, True, 'human', (7,)),),
            ),
            project_scope,
        )
        is KeywordSqlTruth.FALSE
    )
    assert truth(legacy, all_scope) is KeywordSqlTruth.TRUE
    assert truth(legacy, source_scope) is KeywordSqlTruth.FALSE
    wrong_child = _resource_row(
        'history_event:1',
        links=((all_scope.workspace_scope_id, True, 'human', (8,)),),
    )
    assert truth(wrong_child, source_scope) is KeywordSqlTruth.FALSE


def test_production_resource_predicate_handles_null_empty_and_raw_without_bypass() -> None:
    raw = _resource_row(
        'chunk:1',
        serving_kind='raw_chunk',
        project_key=None,
        raw_source_id=7,
    )
    legacy = _resource_row('history_event:1')
    request = _request()
    empty = build_postgres_keyword_bind_values(
        request=request,
        terms=tokenize_rag_lexical_query(request.retrieval_query_text),
        settings=_settings(),
    )

    def truth(row, **overrides):
        return evaluate_postgres_keyword_predicate(
            POSTGRES_KEYWORD_RESOURCE_PREDICATE,
            candidate=row,
            bind_values={**empty, **overrides},
        )

    assert truth(raw) is KeywordSqlTruth.TRUE
    assert truth(legacy) is KeywordSqlTruth.TRUE
    assert truth(raw, source_ids=(7,)) is KeywordSqlTruth.TRUE
    assert truth(raw, source_ids=(8,)) is KeywordSqlTruth.FALSE
    assert truth(raw, project_keys=None) is KeywordSqlTruth.FALSE
    assert truth(raw, source_ids=None) is KeywordSqlTruth.FALSE
    assert truth(legacy, project_keys=None) is KeywordSqlTruth.FALSE
    assert truth(legacy, source_id_texts=None) is KeywordSqlTruth.FALSE


def _replace_predicate_rule(predicate, rule_name, mutation):
    replacements = 0

    def visit(node):
        nonlocal replacements
        if isinstance(node, KeywordRule):
            if node.name == rule_name:
                replacements += 1
                return replace(node, predicate=mutation(node.predicate))
            return replace(node, predicate=visit(node.predicate))
        if isinstance(node, KeywordBooleanPredicate):
            return replace(node, children=tuple(visit(child) for child in node.children))
        if isinstance(node, KeywordExistsPredicate):
            return replace(node, where=visit(node.where))
        return node

    result = visit(predicate)
    assert replacements == 1
    return result


def _mutate_comparison(operator: KeywordComparisonOperator):
    def mutate(node):
        assert isinstance(node, KeywordComparisonPredicate)
        return replace(node, operator=operator)

    return mutate


def _mutate_existence(polarity: KeywordExistence):
    def mutate(node):
        assert isinstance(node, KeywordExistsPredicate)
        return replace(node, polarity=polarity)

    return mutate


def _mutate_boolean(operator: KeywordBooleanOperator):
    def mutate(node):
        assert isinstance(node, KeywordBooleanPredicate)
        return replace(node, operator=operator)

    return mutate


def _constant_true(node):
    del node
    return KeywordConstantPredicate(KeywordSqlTruth.TRUE)


def _mutation_case(case_name: str):
    all_scope = _scope()
    source_scope = _scope(
        resource_scope_mode='constrained',
        source_constraints=('source_pk:7',),
    )
    project_scope = _scope(
        resource_scope_mode='constrained',
        project_constraints=('project_key:project-a',),
    )
    current = _resource_row(
        'history_event:1',
        links=(('workspace-1', True, 'human', (7,)),),
    )
    cases = {
        'foreign-workspace': (
            _resource_row(
                'history_event:1',
                links=(('workspace-foreign', True, 'human', (7,)),),
            ),
            all_scope,
            {},
            KeywordSqlTruth.FALSE,
        ),
        'current-explicit': (current, source_scope, {}, KeywordSqlTruth.TRUE),
        'current-project': (current, project_scope, {}, KeywordSqlTruth.TRUE),
        'inactive-explicit': (
            _resource_row(
                'history_event:1',
                links=(('workspace-1', False, 'human', (7,)),),
            ),
            source_scope,
            {},
            KeywordSqlTruth.FALSE,
        ),
        'wrong-resolution': (
            _resource_row(
                'history_event:1',
                links=(('workspace-1', True, 'model', (7,)),),
            ),
            source_scope,
            {},
            KeywordSqlTruth.FALSE,
        ),
        'legacy': (
            _resource_row('history_event:2'),
            all_scope,
            {},
            KeywordSqlTruth.TRUE,
        ),
        'raw-source': (
            _resource_row(
                'chunk:1',
                serving_kind='raw_chunk',
                project_key=None,
                raw_source_id=7,
            ),
            source_scope,
            {},
            KeywordSqlTruth.TRUE,
        ),
        'null-source-array': (
            current,
            source_scope,
            {'source_id_texts': None},
            KeywordSqlTruth.FALSE,
        ),
        'empty-source-array': (
            current,
            all_scope,
            {},
            KeywordSqlTruth.TRUE,
        ),
    }
    return cases[case_name]


@pytest.mark.parametrize(
    ('mutation_name', 'rule_name', 'mutation', 'case_name'),
    (
        ('workspace-bypass', 'explicit_workspace', _constant_true, 'foreign-workspace'),
        (
            'workspace-comparator',
            'explicit_workspace',
            _mutate_comparison(KeywordComparisonOperator.NE),
            'current-explicit',
        ),
        (
            'project-comparator',
            'history_project_membership',
            _mutate_comparison(KeywordComparisonOperator.NOT_EQUAL_ALL),
            'current-project',
        ),
        ('explicit-active', 'explicit_active', _constant_true, 'inactive-explicit'),
        (
            'explicit-resolution',
            'explicit_resolution',
            _constant_true,
            'wrong-resolution',
        ),
        (
            'explicit-existence',
            'explicit_link_exists',
            _mutate_existence(KeywordExistence.NOT_EXISTS),
            'current-explicit',
        ),
        (
            'legacy-absence',
            'legacy_active_link_absence',
            _mutate_existence(KeywordExistence.EXISTS),
            'legacy',
        ),
        (
            'every-child-polarity',
            'every_child_violation_absence',
            _mutate_existence(KeywordExistence.EXISTS),
            'current-explicit',
        ),
        (
            'raw-source-comparator',
            'raw_source_membership',
            _mutate_comparison(KeywordComparisonOperator.NOT_EQUAL_ALL),
            'raw-source',
        ),
        (
            'root-or-grouping',
            'root_resource_group',
            _mutate_boolean(KeywordBooleanOperator.AND),
            'current-explicit',
        ),
        (
            'null-guard',
            'explicit_sources_not_null',
            _constant_true,
            'null-source-array',
        ),
        (
            'empty-array',
            'explicit_source_empty',
            _mutate_comparison(KeywordComparisonOperator.NE),
            'empty-source-array',
        ),
    ),
)
def test_declarative_predicate_mutations_change_sql_and_fail_executed_matrix(
    mutation_name,
    rule_name,
    mutation,
    case_name,
) -> None:
    del mutation_name
    candidate, scope, overrides, expected = _mutation_case(case_name)
    request = _request(scope=scope)
    bind_values = build_postgres_keyword_bind_values(
        request=request,
        terms=tokenize_rag_lexical_query(request.retrieval_query_text),
        settings=_settings(),
    )
    bind_values = {**bind_values, **overrides}
    mutant = _replace_predicate_rule(
        POSTGRES_KEYWORD_RESOURCE_PREDICATE,
        rule_name,
        mutation,
    )

    assert (
        evaluate_postgres_keyword_predicate(
            POSTGRES_KEYWORD_RESOURCE_PREDICATE,
            candidate=candidate,
            bind_values=bind_values,
        )
        is expected
    )
    assert build_postgres_keyword_search_sql(mutant) != POSTGRES_KEYWORD_SEARCH_SQL
    assert (
        evaluate_postgres_keyword_predicate(
            mutant,
            candidate=candidate,
            bind_values=bind_values,
        )
        is not expected
    )


def test_postgresql_search_uses_single_typed_bind_spec_factory() -> None:
    class _Dialect:
        name = 'postgresql'

    class _Bind:
        dialect = _Dialect()

    class _EmptyResult:
        def mappings(self):
            return self

        def all(self):
            return []

    class _FakePostgresSession:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict[str, object]]] = []

        def get_bind(self):
            return _Bind()

        def execute(self, statement, parameters):
            self.calls.append((str(statement), dict(parameters)))
            return _EmptyResult()

    scope = _scope(
        resource_scope_mode='constrained',
        project_constraints=('project_key:project-a',),
        source_constraints=('source_pk:7',),
    )
    request = _request(query='  Alpha Roadmap  ', scope=scope)
    session = _FakePostgresSession()
    store = SqlAlchemyKeywordSearchStore(
        db=session,  # type: ignore[arg-type]
        settings=_settings(),
    )

    assert store.search(request) == ()
    assert tuple(
        (spec.name, spec.sql_type, spec.target)
        for spec in POSTGRES_KEYWORD_BIND_SPECS
    ) == (
        ('timeout_value', 'text', PostgresKeywordBindTarget.TIMEOUT),
        ('query_terms', 'text[]', PostgresKeywordBindTarget.SEARCH),
        ('phrase_lower', 'text', PostgresKeywordBindTarget.SEARCH),
        ('lexical_contract_version', 'text', PostgresKeywordBindTarget.SEARCH),
        ('fingerprint_key_version', 'text', PostgresKeywordBindTarget.SEARCH),
        ('key_material_verifier', 'text', PostgresKeywordBindTarget.SEARCH),
        ('project_keys', 'text[]', PostgresKeywordBindTarget.SEARCH),
        ('source_ids', 'bigint[]', PostgresKeywordBindTarget.SEARCH),
        ('source_id_texts', 'text[]', PostgresKeywordBindTarget.SEARCH),
        ('workspace_scope_id', 'text', PostgresKeywordBindTarget.SEARCH),
    )
    assert session.calls == [
        (
            "SELECT set_config('statement_timeout', CAST(:timeout_value AS text), true)",
            {'timeout_value': '5000ms'},
        ),
        (
            POSTGRES_KEYWORD_SEARCH_SQL,
            {
                'query_terms': ['alpha', 'roadmap'],
                'phrase_lower': 'alpha roadmap',
                'lexical_contract_version': 'rag-keyword-lexical-compat:v1',
                'fingerprint_key_version': 'keyword-test-v1',
                'key_material_verifier': (
                    '881dd0aac1d34d5412742640d9a75e64705a016327fb31cb19ed8853ff036f65'
                ),
                'project_keys': ['project-a'],
                'source_ids': [7],
                'source_id_texts': ['7'],
                'workspace_scope_id': 'workspace-1',
            },
        ),
    ]


def test_generic_evaluator_uses_sql_three_valued_logic_and_where_true_only() -> None:
    candidate = PostgresKeywordResourceCandidate(
        serving_document_id='history_event:1',
        serving_kind='trusted_knowledge',
        score=1.0,
        relation_rows=(
            _relation_row('nullable_rows', id=1, nullable=None),
        ),
    )
    assert render_postgres_keyword_predicate(
        KeywordConstantPredicate(KeywordSqlTruth.UNKNOWN)
    ) == 'NULL'
    unknown_comparisons = (
        KeywordComparisonPredicate(
            KeywordParameter('nullable'),
            KeywordComparisonOperator.EQ,
            KeywordLiteral(1),
        ),
        KeywordComparisonPredicate(
            KeywordLiteral(1),
            KeywordComparisonOperator.IN,
            KeywordLiteral((2, None)),
        ),
        KeywordComparisonPredicate(
            KeywordLiteral(1),
            KeywordComparisonOperator.EQUAL_ANY,
            KeywordLiteral((2, None)),
        ),
        KeywordComparisonPredicate(
            KeywordLiteral(1),
            KeywordComparisonOperator.NOT_EQUAL_ALL,
            KeywordLiteral((2, None)),
        ),
    )
    for comparison in unknown_comparisons:
        assert (
            evaluate_postgres_keyword_predicate(
                comparison,
                candidate=candidate,
                bind_values={'nullable': None},
            )
            is KeywordSqlTruth.UNKNOWN
        )

    for boolean_predicate in (
        KeywordBooleanPredicate(
            KeywordBooleanOperator.AND,
            (
                KeywordConstantPredicate(KeywordSqlTruth.TRUE),
                unknown_comparisons[0],
            ),
        ),
        KeywordBooleanPredicate(
            KeywordBooleanOperator.OR,
            (
                KeywordConstantPredicate(KeywordSqlTruth.FALSE),
                unknown_comparisons[0],
            ),
        ),
    ):
        assert (
            evaluate_postgres_keyword_predicate(
                boolean_predicate,
                candidate=candidate,
                bind_values={'nullable': None},
            )
            is KeywordSqlTruth.UNKNOWN
        )

    exists_with_unknown_where = KeywordExistsPredicate(
        relation=KeywordRelation('nullable_rows', 'nullable'),
        polarity=KeywordExistence.EXISTS,
        where=KeywordComparisonPredicate(
            KeywordColumn('nullable', 'nullable'),
            KeywordComparisonOperator.EQ,
            KeywordLiteral(1),
        ),
    )
    assert (
        evaluate_postgres_keyword_predicate(
            exists_with_unknown_where,
            candidate=candidate,
            bind_values={},
        )
        is KeywordSqlTruth.FALSE
    )


@pytest.mark.parametrize(
    ('left', 'right', 'any_expected', 'all_expected'),
    (
        (None, None, KeywordSqlTruth.UNKNOWN, KeywordSqlTruth.UNKNOWN),
        (None, (), KeywordSqlTruth.FALSE, KeywordSqlTruth.TRUE),
        (None, (1, 2), KeywordSqlTruth.UNKNOWN, KeywordSqlTruth.UNKNOWN),
        (None, (2, 3), KeywordSqlTruth.UNKNOWN, KeywordSqlTruth.UNKNOWN),
        (None, (1, None), KeywordSqlTruth.UNKNOWN, KeywordSqlTruth.UNKNOWN),
        (None, (2, None), KeywordSqlTruth.UNKNOWN, KeywordSqlTruth.UNKNOWN),
        (None, (None,), KeywordSqlTruth.UNKNOWN, KeywordSqlTruth.UNKNOWN),
        (1, None, KeywordSqlTruth.UNKNOWN, KeywordSqlTruth.UNKNOWN),
        (1, (), KeywordSqlTruth.FALSE, KeywordSqlTruth.TRUE),
        (1, (1, 2), KeywordSqlTruth.TRUE, KeywordSqlTruth.FALSE),
        (1, (2, 3), KeywordSqlTruth.FALSE, KeywordSqlTruth.TRUE),
        (1, (1, None), KeywordSqlTruth.TRUE, KeywordSqlTruth.FALSE),
        (1, (2, None), KeywordSqlTruth.UNKNOWN, KeywordSqlTruth.UNKNOWN),
        (1, (None,), KeywordSqlTruth.UNKNOWN, KeywordSqlTruth.UNKNOWN),
    ),
    ids=(
        'null-left-null-array',
        'null-left-empty-array',
        'null-left-no-null-match-values',
        'null-left-no-null-miss-values',
        'null-left-match-plus-null',
        'null-left-miss-plus-null',
        'null-left-all-null-elements',
        'value-left-null-array',
        'value-left-empty-array',
        'value-left-no-null-match',
        'value-left-no-null-miss',
        'value-left-match-plus-null',
        'value-left-miss-plus-null',
        'value-left-all-null-elements',
    ),
)
def test_quantified_comparisons_match_postgresql_vacuous_three_valued_truth_table(
    left,
    right,
    any_expected: KeywordSqlTruth,
    all_expected: KeywordSqlTruth,
) -> None:
    candidate = PostgresKeywordResourceCandidate(
        serving_document_id='history_event:1',
        serving_kind='trusted_knowledge',
        score=1.0,
        relation_rows=(_relation_row('single_row', id=1),),
    )
    for operator, expected in (
        (KeywordComparisonOperator.EQUAL_ANY, any_expected),
        (KeywordComparisonOperator.NOT_EQUAL_ALL, all_expected),
    ):
        comparison = KeywordComparisonPredicate(
            KeywordLiteral(left),
            operator,
            KeywordParameter('right_array'),
        )
        direct = evaluate_postgres_keyword_predicate(
            comparison,
            candidate=candidate,
            bind_values={'right_array': right},
        )
        where_membership = evaluate_postgres_keyword_predicate(
            KeywordExistsPredicate(
                relation=KeywordRelation('single_row', 'row'),
                polarity=KeywordExistence.EXISTS,
                where=comparison,
            ),
            candidate=candidate,
            bind_values={'right_array': right},
        )

        assert direct is expected
        assert where_membership is (
            KeywordSqlTruth.TRUE
            if expected is KeywordSqlTruth.TRUE
            else KeywordSqlTruth.FALSE
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
