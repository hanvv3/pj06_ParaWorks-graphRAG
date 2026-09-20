from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

from sqlalchemy import select, text
from sqlalchemy.exc import DBAPIError, SQLAlchemyError
from sqlalchemy.orm import Session

from backend.app.admin.auto_review_keys import fingerprint_key_material_verifier
from backend.app.agent_runtime.rag_v2_contracts import resolved_rag_backend
from backend.app.agent_runtime.rag_v2_identity import SecurityScope
from backend.app.core.config import Settings
from backend.app.core.demo_auth import DemoUser
from backend.app.models import RagLexicalServingProjection, RagServingCorpusGeneration
from backend.app.rag.embeddings import (
    OpenAIEmbeddingConfig,
    OpenAIEmbeddingModel,
    ValidatedQueryEmbeddingVector,
    validate_query_embedding_vector_carrier,
)
from backend.app.rag.lexical_projection import (
    score_rag_lexical_candidate,
    tokenize_rag_lexical_query,
)
from backend.app.rag.pgvector_retriever import PgVectorSearchRuntimeError
from backend.app.rag.pgvector_store import (
    PgVectorConfig,
    PgVectorServingCandidateRow,
    PgVectorStore,
)
from backend.app.rag.retrieval import (
    QUERY_EMBEDDING_DIMENSIONS,
    ClassifiedRetrievalCandidate,
    QueryEmbeddingCallResult,
    RetrievalRequest,
)
from backend.app.rag.serving_contracts import (
    EvidenceAccessClassification,
    TrustedServingEnvelope,
    known_permission,
)
from backend.app.rag.serving_generation import RAG_LEXICAL_COMPAT_VERSION
from backend.app.rag.source_observations import (
    CanonicalSourceObservationEligibilityService,
    CanonicalSourceObservationResolver,
)
from backend.app.rag.trusted_evidence import (
    ServingEvidenceResolver,
    TrustedEvidenceAuthorizer,
    TrustedServingEnvelopeResolver,
)


def build_pgvector_search_store(
    *,
    db: Session,
    settings: Settings,
    shared_query_embedding: QueryEmbeddingCallResult | None = None,
):
    if resolved_rag_backend(settings) != 'pgvector':
        return None
    if db.bind is None or db.bind.dialect.name != 'postgresql':
        return None
    if not settings.openai_api_key:
        return None
    store = PgVectorStore(
        session=db,
        config=PgVectorConfig(embedding_dimensions=settings.openai_embedding_dimensions),
    )
    embedding_model = (
        None
        if shared_query_embedding is not None
        else OpenAIEmbeddingModel(
            config=OpenAIEmbeddingConfig(
                api_key=settings.openai_api_key,
                model=settings.openai_embedding_model,
                dimensions=settings.openai_embedding_dimensions,
                timeout_seconds=settings.openai_embedding_timeout_seconds,
            )
        )
    )
    return PgVectorSearchAdapter(
        store=store,
        embedding_model=embedding_model,
        shared_query_embedding=shared_query_embedding,
    )


def build_rag_v2_pgvector_search_store(
    *,
    db: Session,
    settings: Settings,
):
    if db.get_bind().dialect.name != 'postgresql':
        return None
    store = PgVectorStore(
        session=db,
        config=PgVectorConfig(
            embedding_dimensions=settings.openai_embedding_dimensions
        ),
        settings=settings,
    )
    return SqlAlchemyPgVectorSearchStore(
        db=db,
        store=store,
        settings=settings,
    )


class SqlAlchemyPgVectorSearchStore:
    """D-only store: vector rows rank; canonical resolvers authorize/project."""

    def __init__(
        self,
        *,
        db: Session,
        store: PgVectorStore,
        settings: Settings,
    ) -> None:
        self._db = db
        self._store = store
        self._settings = settings

    def search(
        self,
        request: RetrievalRequest,
        vector: ValidatedQueryEmbeddingVector,
    ) -> tuple[ClassifiedRetrievalCandidate, ...]:
        validate_query_embedding_vector_carrier(
            vector,
            expected_dimensions=QUERY_EMBEDDING_DIMENSIONS,
        )
        try:
            rows = self._store.search_rag_v2(
                request=request,
                query_embedding=vector,
            )
            candidates: list[ClassifiedRetrievalCandidate] = []
            for row in rows:
                candidate = self._canonical_candidate(row, request=request)
                if candidate is None or not _is_stage_one_candidate(candidate.access):
                    raise PgVectorSearchRuntimeError(
                        'pgvector retrieval is unavailable'
                    )
                candidates.append(candidate)
            return tuple(candidates)
        except PgVectorSearchRuntimeError:
            raise
        except (DBAPIError, SQLAlchemyError, TypeError, ValueError):
            raise PgVectorSearchRuntimeError(
                'pgvector retrieval is unavailable'
            ) from None

    def _canonical_candidate(
        self,
        row: PgVectorServingCandidateRow,
        *,
        request: RetrievalRequest,
    ) -> ClassifiedRetrievalCandidate | None:
        resolved = _resolve_canonical_projection(
            self._db,
            self._settings,
            row,
            request=request,
        )
        if resolved is None:
            return None
        evidence, access = resolved
        return ClassifiedRetrievalCandidate(
            evidence=evidence,
            relevance_score=row.score,
            matched_terms=(),
            access=access,
        )


def _resolve_canonical_projection(
    db: Session,
    settings: Settings,
    row: Mapping[str, object] | PgVectorServingCandidateRow,
    *,
    request: RetrievalRequest,
) -> tuple[object, EvidenceAccessClassification] | None:
    document_id = str(_row_value(row, 'serving_document_id'))
    prefix, separator, raw_id = document_id.partition(':')
    if separator != ':' or not raw_id.isascii() or not raw_id.isdecimal():
        return None
    identifier = int(raw_id)
    if identifier <= 0 or str(identifier) != raw_id:
        return None
    if prefix == 'chunk':
        observation = CanonicalSourceObservationResolver(
            db=db,
            settings=settings,
        ).resolve_for_index(identifier)
        if observation is None:
            return None
        identity = observation.identity
        evidence = observation.evidence
        access = CanonicalSourceObservationEligibilityService().classify_access(
            request.security_scope,
            observation,
        )
    elif prefix in {'decision_record', 'history_event', 'timeline_event', 'todo'}:
        envelope = TrustedServingEnvelopeResolver(
            db=db,
            settings=settings,
        ).resolve_for_index(prefix, identifier)
        if envelope is None:
            return None
        identity = envelope.identity
        evidence = envelope.evidence
        access = _classify_trusted_stage_one_access(
            TrustedEvidenceAuthorizer(db=db, settings=settings),
            scope=request.security_scope,
            envelope=envelope,
        )
    else:
        return None
    if not _projection_matches(row, evidence=evidence):
        return None
    if access.permission_visibility == 'visible':
        visible = ServingEvidenceResolver(settings=settings).resolve_candidate(
            db=db,
            identity=identity,
            scope=request.security_scope,
        )
        if visible is None:
            return None
        evidence = visible
    return evidence, access


class PgVectorSearchAdapter:
    def __init__(
        self,
        *,
        store: PgVectorStore,
        embedding_model: OpenAIEmbeddingModel | None,
        shared_query_embedding: QueryEmbeddingCallResult | None = None,
    ) -> None:
        if embedding_model is None and shared_query_embedding is None:
            raise ValueError('pgvector embedding authority is required')
        self.store = store
        self.embedding_model = embedding_model
        self.shared_query_embedding = shared_query_embedding
        self._query_embeddings: dict[str, list[float]] = {}

    def search(self, *, query: str, user: DemoUser, limit: int = 5):
        if self.shared_query_embedding is not None:
            from backend.app.agents.rag_orchestrator_agent.v2_embedding import (
                share_query_embedding_result,
            )

            shared = share_query_embedding_result(
                self.shared_query_embedding,
                legacy_query_text=query,
                v2_query_text=query,
            )
            return self.store.search_with_embedding(
                query_embedding=shared.vector.coordinates,
                user=user,
                limit=limit,
            )
        query_embedding = self._query_embeddings.get(query)
        if query_embedding is None:
            if self.embedding_model is None:
                raise ValueError('pgvector embedding authority is unavailable')
            query_embedding = self.embedding_model.embed(query)
            self._query_embeddings[query] = query_embedding
        return self.store.search_with_embedding(
            query_embedding=query_embedding,
            user=user,
            limit=limit,
        )


class KeywordSqlTruth(StrEnum):
    TRUE = 'true'
    FALSE = 'false'
    UNKNOWN = 'unknown'


class KeywordBooleanOperator(StrEnum):
    AND = 'and'
    OR = 'or'


class KeywordComparisonOperator(StrEnum):
    EQ = 'eq'
    NE = 'ne'
    IN = 'in'
    EQUAL_ANY = 'equal_any'
    NOT_EQUAL_ALL = 'not_equal_all'
    IS_TRUE = 'is_true'
    IS_NOT_NULL = 'is_not_null'


class KeywordExistence(StrEnum):
    EXISTS = 'exists'
    NOT_EXISTS = 'not_exists'


class KeywordFunctionName(StrEnum):
    SPLIT_PART = 'split_part'
    CARDINALITY = 'cardinality'


class PostgresKeywordBindTarget(StrEnum):
    TIMEOUT = 'timeout'
    SEARCH = 'search'


class _KeywordBindSource(StrEnum):
    CONSTANT = 'constant'
    TERMS = 'terms'
    REQUEST = 'request'
    SETTINGS = 'settings'


class _KeywordBindTransform(StrEnum):
    TO_TUPLE = 'to_tuple'
    STRIP_LOWER = 'strip_lower'
    REMOVE_PREFIX_EACH = 'remove_prefix_each'
    INT_EACH = 'int_each'
    FINGERPRINT_VERIFIER = 'fingerprint_verifier'


@dataclass(frozen=True, slots=True)
class _KeywordBindTransformSpec:
    operation: _KeywordBindTransform
    argument: str | None = None


@dataclass(frozen=True, slots=True)
class PostgresKeywordBindSpec:
    name: str
    sql_type: str
    target: PostgresKeywordBindTarget
    source: _KeywordBindSource
    path: tuple[str, ...] = ()
    constant: object = None
    transforms: tuple[_KeywordBindTransformSpec, ...] = ()


POSTGRES_KEYWORD_BIND_SPECS = (
    PostgresKeywordBindSpec(
        name='timeout_value',
        sql_type='text',
        target=PostgresKeywordBindTarget.TIMEOUT,
        source=_KeywordBindSource.CONSTANT,
        constant='5000ms',
    ),
    PostgresKeywordBindSpec(
        name='query_terms',
        sql_type='text[]',
        target=PostgresKeywordBindTarget.SEARCH,
        source=_KeywordBindSource.TERMS,
        transforms=(
            _KeywordBindTransformSpec(_KeywordBindTransform.TO_TUPLE),
        ),
    ),
    PostgresKeywordBindSpec(
        name='phrase_lower',
        sql_type='text',
        target=PostgresKeywordBindTarget.SEARCH,
        source=_KeywordBindSource.REQUEST,
        path=('retrieval_query_text',),
        transforms=(
            _KeywordBindTransformSpec(_KeywordBindTransform.STRIP_LOWER),
        ),
    ),
    PostgresKeywordBindSpec(
        name='lexical_contract_version',
        sql_type='text',
        target=PostgresKeywordBindTarget.SEARCH,
        source=_KeywordBindSource.CONSTANT,
        constant=RAG_LEXICAL_COMPAT_VERSION,
    ),
    PostgresKeywordBindSpec(
        name='fingerprint_key_version',
        sql_type='text',
        target=PostgresKeywordBindTarget.SEARCH,
        source=_KeywordBindSource.SETTINGS,
        path=('agent_runtime_fingerprint_key_version',),
    ),
    PostgresKeywordBindSpec(
        name='key_material_verifier',
        sql_type='text',
        target=PostgresKeywordBindTarget.SEARCH,
        source=_KeywordBindSource.SETTINGS,
        path=('agent_runtime_fingerprint_secret',),
        transforms=(
            _KeywordBindTransformSpec(
                _KeywordBindTransform.FINGERPRINT_VERIFIER
            ),
        ),
    ),
    PostgresKeywordBindSpec(
        name='project_keys',
        sql_type='text[]',
        target=PostgresKeywordBindTarget.SEARCH,
        source=_KeywordBindSource.REQUEST,
        path=('security_scope', 'project_constraints'),
        transforms=(
            _KeywordBindTransformSpec(
                _KeywordBindTransform.REMOVE_PREFIX_EACH,
                'project_key:',
            ),
        ),
    ),
    PostgresKeywordBindSpec(
        name='source_ids',
        sql_type='bigint[]',
        target=PostgresKeywordBindTarget.SEARCH,
        source=_KeywordBindSource.REQUEST,
        path=('security_scope', 'source_constraints'),
        transforms=(
            _KeywordBindTransformSpec(
                _KeywordBindTransform.REMOVE_PREFIX_EACH,
                'source_pk:',
            ),
            _KeywordBindTransformSpec(_KeywordBindTransform.INT_EACH),
        ),
    ),
    PostgresKeywordBindSpec(
        name='source_id_texts',
        sql_type='text[]',
        target=PostgresKeywordBindTarget.SEARCH,
        source=_KeywordBindSource.REQUEST,
        path=('security_scope', 'source_constraints'),
        transforms=(
            _KeywordBindTransformSpec(
                _KeywordBindTransform.REMOVE_PREFIX_EACH,
                'source_pk:',
            ),
        ),
    ),
    PostgresKeywordBindSpec(
        name='workspace_scope_id',
        sql_type='text',
        target=PostgresKeywordBindTarget.SEARCH,
        source=_KeywordBindSource.REQUEST,
        path=('security_scope', 'workspace_scope_id'),
    ),
)


def _resolve_keyword_bind_path(root: object, path: tuple[str, ...]) -> object:
    value = root
    for attribute in path:
        value = getattr(value, attribute)
    return value


def _apply_keyword_bind_transform(
    value: object,
    transform: _KeywordBindTransformSpec,
) -> object:
    if transform.operation is _KeywordBindTransform.TO_TUPLE:
        return tuple(value)  # type: ignore[arg-type]
    if transform.operation is _KeywordBindTransform.STRIP_LOWER:
        return str(value).strip().lower()
    if transform.operation is _KeywordBindTransform.REMOVE_PREFIX_EACH:
        prefix = transform.argument or ''
        return tuple(str(item).removeprefix(prefix) for item in value)  # type: ignore[union-attr]
    if transform.operation is _KeywordBindTransform.INT_EACH:
        return tuple(int(item) for item in value)  # type: ignore[union-attr]
    if transform.operation is _KeywordBindTransform.FINGERPRINT_VERIFIER:
        return fingerprint_key_material_verifier(str(value))
    raise ValueError(f'unsupported keyword bind transform: {transform.operation}')


def build_postgres_keyword_bind_values(
    *,
    request: RetrievalRequest,
    terms: tuple[str, ...],
    settings: Settings,
) -> dict[str, object]:
    roots = {
        _KeywordBindSource.TERMS: terms,
        _KeywordBindSource.REQUEST: request,
        _KeywordBindSource.SETTINGS: settings,
    }
    values: dict[str, object] = {}
    for spec in POSTGRES_KEYWORD_BIND_SPECS:
        value = (
            spec.constant
            if spec.source is _KeywordBindSource.CONSTANT
            else _resolve_keyword_bind_path(roots[spec.source], spec.path)
        )
        for transform in spec.transforms:
            value = _apply_keyword_bind_transform(value, transform)
        values[spec.name] = value
    return values


@dataclass(frozen=True, slots=True)
class PostgresKeywordRelationRow:
    relation: str
    columns: tuple[tuple[str, str | int | bool | None], ...]


@dataclass(frozen=True, slots=True)
class PostgresKeywordResourceCandidate:
    serving_document_id: str
    serving_kind: str
    score: float
    relation_rows: tuple[PostgresKeywordRelationRow, ...]


@dataclass(frozen=True, slots=True)
class KeywordColumn:
    alias: str
    name: str


@dataclass(frozen=True, slots=True)
class KeywordParameter:
    name: str


@dataclass(frozen=True, slots=True)
class KeywordLiteral:
    value: object


@dataclass(frozen=True, slots=True)
class KeywordFunction:
    name: KeywordFunctionName
    arguments: tuple[object, ...]


@dataclass(frozen=True, slots=True)
class KeywordCast:
    expression: object
    sql_type: str


@dataclass(frozen=True, slots=True)
class KeywordConstantPredicate:
    truth: KeywordSqlTruth


@dataclass(frozen=True, slots=True)
class KeywordComparisonPredicate:
    left: object
    operator: KeywordComparisonOperator
    right: object | None = None


@dataclass(frozen=True, slots=True)
class KeywordBooleanPredicate:
    operator: KeywordBooleanOperator
    children: tuple[object, ...]


@dataclass(frozen=True, slots=True)
class KeywordRelation:
    table: str
    alias: str


@dataclass(frozen=True, slots=True)
class KeywordExistsPredicate:
    relation: KeywordRelation
    where: object
    polarity: KeywordExistence


@dataclass(frozen=True, slots=True)
class KeywordRule:
    name: str
    predicate: object


def _keyword_sql_and(values: tuple[KeywordSqlTruth, ...]) -> KeywordSqlTruth:
    if KeywordSqlTruth.FALSE in values:
        return KeywordSqlTruth.FALSE
    if KeywordSqlTruth.UNKNOWN in values:
        return KeywordSqlTruth.UNKNOWN
    return KeywordSqlTruth.TRUE


def _keyword_sql_or(values: tuple[KeywordSqlTruth, ...]) -> KeywordSqlTruth:
    if KeywordSqlTruth.TRUE in values:
        return KeywordSqlTruth.TRUE
    if KeywordSqlTruth.UNKNOWN in values:
        return KeywordSqlTruth.UNKNOWN
    return KeywordSqlTruth.FALSE


def _keyword_literal_sql(value: object) -> str:
    if value is None:
        return 'NULL'
    if value is True:
        return 'TRUE'
    if value is False:
        return 'FALSE'
    if isinstance(value, str):
        return "'" + value.replace("'", "''") + "'"
    if isinstance(value, tuple):
        return '(' + ', '.join(_keyword_literal_sql(item) for item in value) + ')'
    return str(value)


def _postgres_keyword_bind_spec(name: str) -> PostgresKeywordBindSpec:
    matches = tuple(spec for spec in POSTGRES_KEYWORD_BIND_SPECS if spec.name == name)
    if len(matches) != 1:
        raise ValueError(f'unknown PostgreSQL keyword bind: {name}')
    return matches[0]


def _render_postgres_keyword_expression(expression: object) -> str:
    if isinstance(expression, KeywordColumn):
        return f'{expression.alias}.{expression.name}'
    if isinstance(expression, KeywordParameter):
        spec = _postgres_keyword_bind_spec(expression.name)
        return f'CAST(:{spec.name} AS {spec.sql_type})'
    if isinstance(expression, KeywordLiteral):
        return _keyword_literal_sql(expression.value)
    if isinstance(expression, KeywordFunction):
        arguments = ', '.join(
            _render_postgres_keyword_expression(argument)
            for argument in expression.arguments
        )
        return f'{expression.name.value}({arguments})'
    if isinstance(expression, KeywordCast):
        return (
            f'CAST({_render_postgres_keyword_expression(expression.expression)} '
            f'AS {expression.sql_type})'
        )
    raise TypeError(f'unsupported keyword SQL expression: {type(expression)!r}')


def render_postgres_keyword_predicate(predicate: object) -> str:
    if isinstance(predicate, KeywordRule):
        return render_postgres_keyword_predicate(predicate.predicate)
    if isinstance(predicate, KeywordConstantPredicate):
        if predicate.truth is KeywordSqlTruth.UNKNOWN:
            return 'NULL'
        return predicate.truth.value.upper()
    if isinstance(predicate, KeywordBooleanPredicate):
        separator = f' {predicate.operator.value.upper()} '
        return '(' + separator.join(
            render_postgres_keyword_predicate(child)
            for child in predicate.children
        ) + ')'
    if isinstance(predicate, KeywordComparisonPredicate):
        left = _render_postgres_keyword_expression(predicate.left)
        if predicate.operator is KeywordComparisonOperator.IS_TRUE:
            return f'{left} IS TRUE'
        if predicate.operator is KeywordComparisonOperator.IS_NOT_NULL:
            return f'{left} IS NOT NULL'
        if predicate.right is None:
            raise ValueError(f'{predicate.operator.value} requires a right operand')
        right = _render_postgres_keyword_expression(predicate.right)
        operators = {
            KeywordComparisonOperator.EQ: '=',
            KeywordComparisonOperator.NE: '<>',
            KeywordComparisonOperator.IN: 'IN',
            KeywordComparisonOperator.EQUAL_ANY: '= ANY',
            KeywordComparisonOperator.NOT_EQUAL_ALL: '<> ALL',
        }
        operator = operators[predicate.operator]
        if predicate.operator in {
            KeywordComparisonOperator.EQUAL_ANY,
            KeywordComparisonOperator.NOT_EQUAL_ALL,
        }:
            return f'{left} {operator}({right})'
        return f'{left} {operator} {right}'
    if isinstance(predicate, KeywordExistsPredicate):
        prefix = (
            'EXISTS'
            if predicate.polarity is KeywordExistence.EXISTS
            else 'NOT EXISTS'
        )
        return (
            f'{prefix} (SELECT 1 FROM {predicate.relation.table} '
            f'AS {predicate.relation.alias} WHERE '
            f'{render_postgres_keyword_predicate(predicate.where)})'
        )
    raise TypeError(f'unsupported keyword SQL predicate: {type(predicate)!r}')


def _keyword_relation_value(
    row: PostgresKeywordRelationRow,
    column_name: str,
) -> object:
    matches = tuple(value for name, value in row.columns if name == column_name)
    if len(matches) != 1:
        return None
    return matches[0]


def _evaluate_keyword_expression(
    expression: object,
    *,
    candidate: PostgresKeywordResourceCandidate,
    bind_values: Mapping[str, object],
    relation_context: Mapping[str, PostgresKeywordRelationRow],
) -> object:
    if isinstance(expression, KeywordColumn):
        if expression.alias == 'projection':
            return getattr(candidate, expression.name, None)
        row = relation_context.get(expression.alias)
        return None if row is None else _keyword_relation_value(row, expression.name)
    if isinstance(expression, KeywordParameter):
        return bind_values.get(expression.name)
    if isinstance(expression, KeywordLiteral):
        return expression.value
    if isinstance(expression, KeywordFunction):
        arguments = tuple(
            _evaluate_keyword_expression(
                argument,
                candidate=candidate,
                bind_values=bind_values,
                relation_context=relation_context,
            )
            for argument in expression.arguments
        )
        if any(argument is None for argument in arguments):
            return None
        if expression.name is KeywordFunctionName.SPLIT_PART:
            value, delimiter, index = arguments
            parts = str(value).split(str(delimiter))
            position = int(index) - 1
            return parts[position] if 0 <= position < len(parts) else ''
        if expression.name is KeywordFunctionName.CARDINALITY:
            return len(arguments[0])  # type: ignore[arg-type]
        raise ValueError(f'unsupported keyword function: {expression.name}')
    if isinstance(expression, KeywordCast):
        value = _evaluate_keyword_expression(
            expression.expression,
            candidate=candidate,
            bind_values=bind_values,
            relation_context=relation_context,
        )
        if value is None:
            return None
        if expression.sql_type == 'bigint':
            return int(value)
        if expression.sql_type == 'text':
            return str(value)
        raise ValueError(f'unsupported keyword cast: {expression.sql_type}')
    raise TypeError(f'unsupported keyword expression: {type(expression)!r}')


def _evaluate_keyword_comparison(
    predicate: KeywordComparisonPredicate,
    *,
    candidate: PostgresKeywordResourceCandidate,
    bind_values: Mapping[str, object],
    relation_context: Mapping[str, PostgresKeywordRelationRow],
) -> KeywordSqlTruth:
    if predicate.operator in {
        KeywordComparisonOperator.IS_TRUE,
        KeywordComparisonOperator.IS_NOT_NULL,
    }:
        left = _evaluate_keyword_expression(
            predicate.left,
            candidate=candidate,
            bind_values=bind_values,
            relation_context=relation_context,
        )
        if predicate.operator is KeywordComparisonOperator.IS_TRUE:
            return KeywordSqlTruth.TRUE if left is True else KeywordSqlTruth.FALSE
        return KeywordSqlTruth.TRUE if left is not None else KeywordSqlTruth.FALSE
    if predicate.right is None:
        raise ValueError(f'{predicate.operator.value} requires a right operand')
    right = _evaluate_keyword_expression(
        predicate.right,
        candidate=candidate,
        bind_values=bind_values,
        relation_context=relation_context,
    )
    if predicate.operator in {
        KeywordComparisonOperator.EQUAL_ANY,
        KeywordComparisonOperator.NOT_EQUAL_ALL,
    }:
        if right is None:
            return KeywordSqlTruth.UNKNOWN
        if not isinstance(right, (list, tuple)):
            raise TypeError(f'{predicate.operator.value} requires an array operand')
        items = tuple(right)
        if not items:
            return (
                KeywordSqlTruth.FALSE
                if predicate.operator is KeywordComparisonOperator.EQUAL_ANY
                else KeywordSqlTruth.TRUE
            )
        left = _evaluate_keyword_expression(
            predicate.left,
            candidate=candidate,
            bind_values=bind_values,
            relation_context=relation_context,
        )
        comparisons = tuple(
            KeywordSqlTruth.UNKNOWN
            if left is None or item is None
            else (
                KeywordSqlTruth.TRUE
                if (
                    left == item
                    if predicate.operator is KeywordComparisonOperator.EQUAL_ANY
                    else left != item
                )
                else KeywordSqlTruth.FALSE
            )
            for item in items
        )
        if predicate.operator is KeywordComparisonOperator.EQUAL_ANY:
            return _keyword_sql_or(comparisons)
        return _keyword_sql_and(comparisons)
    left = _evaluate_keyword_expression(
        predicate.left,
        candidate=candidate,
        bind_values=bind_values,
        relation_context=relation_context,
    )
    if left is None or right is None:
        return KeywordSqlTruth.UNKNOWN
    if predicate.operator is KeywordComparisonOperator.EQ:
        result = left == right
    elif predicate.operator is KeywordComparisonOperator.NE:
        result = left != right
    elif predicate.operator is KeywordComparisonOperator.IN:
        items = tuple(right)  # type: ignore[arg-type]
        if any(item is not None and left == item for item in items):
            return KeywordSqlTruth.TRUE
        if any(item is None for item in items):
            return KeywordSqlTruth.UNKNOWN
        result = False
    else:
        raise ValueError(f'unsupported keyword comparison: {predicate.operator}')
    return KeywordSqlTruth.TRUE if result else KeywordSqlTruth.FALSE


def evaluate_postgres_keyword_predicate(
    predicate: object,
    *,
    candidate: PostgresKeywordResourceCandidate,
    bind_values: Mapping[str, object],
    _relation_context: Mapping[str, PostgresKeywordRelationRow] | None = None,
) -> KeywordSqlTruth:
    relation_context = _relation_context or {}
    if isinstance(predicate, KeywordRule):
        return evaluate_postgres_keyword_predicate(
            predicate.predicate,
            candidate=candidate,
            bind_values=bind_values,
            _relation_context=relation_context,
        )
    if isinstance(predicate, KeywordConstantPredicate):
        return predicate.truth
    if isinstance(predicate, KeywordComparisonPredicate):
        return _evaluate_keyword_comparison(
            predicate,
            candidate=candidate,
            bind_values=bind_values,
            relation_context=relation_context,
        )
    if isinstance(predicate, KeywordBooleanPredicate):
        values = tuple(
            evaluate_postgres_keyword_predicate(
                child,
                candidate=candidate,
                bind_values=bind_values,
                _relation_context=relation_context,
            )
            for child in predicate.children
        )
        if predicate.operator is KeywordBooleanOperator.AND:
            return _keyword_sql_and(values)
        return _keyword_sql_or(values)
    if isinstance(predicate, KeywordExistsPredicate):
        matched = any(
            evaluate_postgres_keyword_predicate(
                predicate.where,
                candidate=candidate,
                bind_values=bind_values,
                _relation_context={
                    **relation_context,
                    predicate.relation.alias: row,
                },
            )
            is KeywordSqlTruth.TRUE
            for row in candidate.relation_rows
            if row.relation == predicate.relation.table
        )
        if predicate.polarity is KeywordExistence.NOT_EXISTS:
            matched = not matched
        return KeywordSqlTruth.TRUE if matched else KeywordSqlTruth.FALSE
    raise TypeError(f'unsupported keyword predicate: {type(predicate)!r}')


def _keyword_and(*children: object) -> KeywordBooleanPredicate:
    return KeywordBooleanPredicate(KeywordBooleanOperator.AND, children)


def _keyword_or(*children: object) -> KeywordBooleanPredicate:
    return KeywordBooleanPredicate(KeywordBooleanOperator.OR, children)


def _keyword_rule(name: str, predicate: object) -> KeywordRule:
    return KeywordRule(name=name, predicate=predicate)


def _keyword_eq(left: object, right: object) -> KeywordComparisonPredicate:
    return KeywordComparisonPredicate(left, KeywordComparisonOperator.EQ, right)


def _keyword_split_part(expression: object, index: int) -> KeywordFunction:
    return KeywordFunction(
        KeywordFunctionName.SPLIT_PART,
        (expression, KeywordLiteral(':'), KeywordLiteral(index)),
    )


_PROJECTION_DOCUMENT_ID = KeywordColumn('projection', 'serving_document_id')
_PROJECTION_KIND = KeywordColumn('projection', 'serving_kind')
_DOCUMENT_KIND = _keyword_split_part(_PROJECTION_DOCUMENT_ID, 1)
_DOCUMENT_ROW_ID = KeywordCast(
    _keyword_split_part(_PROJECTION_DOCUMENT_ID, 2),
    'bigint',
)


def _keyword_project_branch(
    *,
    document_kind: str,
    table: str,
    rule_name: str,
) -> KeywordBooleanPredicate:
    relation = KeywordRelation(table, 'knowledge')
    return _keyword_and(
        _keyword_eq(_DOCUMENT_KIND, KeywordLiteral(document_kind)),
        KeywordExistsPredicate(
            relation=relation,
            polarity=KeywordExistence.EXISTS,
            where=_keyword_and(
                _keyword_eq(KeywordColumn('knowledge', 'id'), _DOCUMENT_ROW_ID),
                _keyword_rule(
                    rule_name,
                    KeywordComparisonPredicate(
                        KeywordColumn('knowledge', 'project_key'),
                        KeywordComparisonOperator.EQUAL_ANY,
                        KeywordParameter('project_keys'),
                    ),
                ),
            ),
        ),
    )


def _keyword_approval_target(*, active_rule_name: str) -> object:
    approval = 'approval'
    return _keyword_and(
        _keyword_or(
            _keyword_and(
                _keyword_eq(_DOCUMENT_KIND, KeywordLiteral('decision_record')),
                KeywordComparisonPredicate(
                    KeywordColumn(approval, 'knowledge_type'),
                    KeywordComparisonOperator.IN,
                    KeywordLiteral(('decision', 'decision_record')),
                ),
            ),
            _keyword_and(
                KeywordComparisonPredicate(
                    _DOCUMENT_KIND,
                    KeywordComparisonOperator.NE,
                    KeywordLiteral('decision_record'),
                ),
                _keyword_eq(
                    KeywordColumn(approval, 'knowledge_type'),
                    _DOCUMENT_KIND,
                ),
            ),
        ),
        _keyword_eq(KeywordColumn(approval, 'knowledge_id'), _DOCUMENT_ROW_ID),
        _keyword_rule(
            active_rule_name,
            KeywordComparisonPredicate(
                KeywordColumn(approval, 'active'),
                KeywordComparisonOperator.IS_TRUE,
            ),
        ),
    )


_RAW_RESOURCE_PREDICATE = _keyword_and(
    _keyword_eq(_PROJECTION_KIND, KeywordLiteral('raw_chunk')),
    KeywordComparisonPredicate(
        KeywordParameter('project_keys'),
        KeywordComparisonOperator.IS_NOT_NULL,
    ),
    KeywordComparisonPredicate(
        KeywordParameter('source_ids'),
        KeywordComparisonOperator.IS_NOT_NULL,
    ),
    _keyword_eq(
        KeywordFunction(
            KeywordFunctionName.CARDINALITY,
            (KeywordParameter('project_keys'),),
        ),
        KeywordLiteral(0),
    ),
    _keyword_or(
        _keyword_eq(
            KeywordFunction(
                KeywordFunctionName.CARDINALITY,
                (KeywordParameter('source_ids'),),
            ),
            KeywordLiteral(0),
        ),
        KeywordExistsPredicate(
            relation=KeywordRelation('document_chunks', 'chunk'),
            polarity=KeywordExistence.EXISTS,
            where=_keyword_and(
                _keyword_eq(KeywordColumn('chunk', 'id'), _DOCUMENT_ROW_ID),
                _keyword_rule(
                    'raw_source_membership',
                    KeywordComparisonPredicate(
                        KeywordColumn('chunk', 'source_id'),
                        KeywordComparisonOperator.EQUAL_ANY,
                        KeywordParameter('source_ids'),
                    ),
                ),
            ),
        ),
    ),
)


_TRUSTED_PROJECT_PREDICATE = _keyword_and(
    KeywordComparisonPredicate(
        KeywordParameter('project_keys'),
        KeywordComparisonOperator.IS_NOT_NULL,
    ),
    _keyword_or(
        _keyword_eq(
            KeywordFunction(
                KeywordFunctionName.CARDINALITY,
                (KeywordParameter('project_keys'),),
            ),
            KeywordLiteral(0),
        ),
        _keyword_project_branch(
            document_kind='decision_record',
            table='decision_records',
            rule_name='decision_project_membership',
        ),
        _keyword_project_branch(
            document_kind='history_event',
            table='history_events',
            rule_name='history_project_membership',
        ),
        _keyword_project_branch(
            document_kind='timeline_event',
            table='timeline_events',
            rule_name='timeline_project_membership',
        ),
        _keyword_project_branch(
            document_kind='todo',
            table='todos',
            rule_name='todo_project_membership',
        ),
    ),
)


_EXPLICIT_LINK_PREDICATE = _keyword_rule(
    'explicit_link_exists',
    KeywordExistsPredicate(
        relation=KeywordRelation(
            'trusted_knowledge_approval_links',
            'approval',
        ),
        polarity=KeywordExistence.EXISTS,
        where=_keyword_and(
            _keyword_approval_target(active_rule_name='explicit_active'),
            _keyword_rule(
                'explicit_workspace',
                _keyword_eq(
                    KeywordColumn('approval', 'security_scope_id'),
                    KeywordParameter('workspace_scope_id'),
                ),
            ),
            _keyword_rule(
                'explicit_resolution',
                KeywordComparisonPredicate(
                    KeywordColumn('approval', 'resolution_source'),
                    KeywordComparisonOperator.IN,
                    KeywordLiteral(('human', 'auto_policy')),
                ),
            ),
            KeywordExistsPredicate(
                relation=KeywordRelation(
                    'trusted_knowledge_evidence_links',
                    'child',
                ),
                polarity=KeywordExistence.EXISTS,
                where=_keyword_eq(
                    KeywordColumn('child', 'approval_link_id'),
                    KeywordColumn('approval', 'id'),
                ),
            ),
            _keyword_rule(
                'explicit_sources_not_null',
                KeywordComparisonPredicate(
                    KeywordParameter('source_id_texts'),
                    KeywordComparisonOperator.IS_NOT_NULL,
                ),
            ),
            _keyword_or(
                _keyword_rule(
                    'explicit_source_empty',
                    _keyword_eq(
                        KeywordFunction(
                            KeywordFunctionName.CARDINALITY,
                            (KeywordParameter('source_id_texts'),),
                        ),
                        KeywordLiteral(0),
                    ),
                ),
                _keyword_rule(
                    'every_child_violation_absence',
                    KeywordExistsPredicate(
                        relation=KeywordRelation(
                            'trusted_knowledge_evidence_links',
                            'child',
                        ),
                        polarity=KeywordExistence.NOT_EXISTS,
                        where=_keyword_and(
                            _keyword_eq(
                                KeywordColumn('child', 'approval_link_id'),
                                KeywordColumn('approval', 'id'),
                            ),
                            KeywordComparisonPredicate(
                                KeywordColumn('child', 'canonical_source_id'),
                                KeywordComparisonOperator.NOT_EQUAL_ALL,
                                KeywordParameter('source_id_texts'),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    ),
)


_LEGACY_LINK_PREDICATE = _keyword_and(
    KeywordComparisonPredicate(
        KeywordParameter('source_id_texts'),
        KeywordComparisonOperator.IS_NOT_NULL,
    ),
    _keyword_eq(
        KeywordFunction(
            KeywordFunctionName.CARDINALITY,
            (KeywordParameter('source_id_texts'),),
        ),
        KeywordLiteral(0),
    ),
    _keyword_rule(
        'legacy_active_link_absence',
        KeywordExistsPredicate(
            relation=KeywordRelation(
                'trusted_knowledge_approval_links',
                'approval',
            ),
            polarity=KeywordExistence.NOT_EXISTS,
            where=_keyword_approval_target(active_rule_name='legacy_active'),
        ),
    ),
)


POSTGRES_KEYWORD_RESOURCE_PREDICATE = _keyword_rule(
    'root_resource_group',
    _keyword_or(
        _RAW_RESOURCE_PREDICATE,
        _keyword_and(
            _keyword_eq(
                _PROJECTION_KIND,
                KeywordLiteral('trusted_knowledge'),
            ),
            _TRUSTED_PROJECT_PREDICATE,
            _keyword_or(_EXPLICIT_LINK_PREDICATE, _LEGACY_LINK_PREDICATE),
        ),
    ),
)


_POSTGRES_KEYWORD_SEARCH_SQL_TEMPLATE = """
WITH coarse_candidates AS (
  SELECT
    projection.serving_document_id,
    projection.serving_kind,
    projection.support_mode,
    projection.effective_permission,
    projection.serving_identity_hmac,
    projection.serving_version_fingerprint,
    projection.model_content_hmac,
    projection.canonical_citation_projection_hmac,
    projection.title_lower,
    projection.searchable_lower
  FROM rag_lexical_serving_projections AS projection
  JOIN rag_serving_corpus_generations AS generation
    ON generation.id = projection.corpus_generation_id
   AND generation.corpus_generation = projection.corpus_generation
  WHERE projection.corpus_generation_id = 1
    AND projection.lexical_contract_version = __BIND_lexical_contract_version__
    AND projection.fingerprint_key_version = __BIND_fingerprint_key_version__
    AND projection.fingerprint_key_material_verifier = __BIND_key_material_verifier__
    AND projection.effective_permission IN ('public', 'internal', 'restricted')
    AND EXISTS (
      SELECT 1
      FROM unnest(__BIND_query_terms__) AS coarse_term(term)
      WHERE strpos(projection.searchable_lower, coarse_term.term) > 0
    )
    AND (__RESOURCE_PREDICATE__)
), scored_candidates AS (
  SELECT
    projection.*,
    scored.score,
    scored.matched_terms
  FROM coarse_candidates AS projection
  CROSS JOIN LATERAL rag_python_lexical_score_v1(
    projection.title_lower,
    projection.searchable_lower,
    __BIND_query_terms__,
    __BIND_phrase_lower__
  ) AS scored
), relevant_candidates AS (
  SELECT *
  FROM scored_candidates AS scored
  WHERE scored.score > 0
  ORDER BY
    CASE WHEN scored.serving_kind = 'trusted_knowledge' THEN 0 ELSE 1 END,
    scored.score DESC,
    scored.serving_document_id
  LIMIT 50
)
SELECT * FROM relevant_candidates
ORDER BY
  CASE WHEN serving_kind = 'trusted_knowledge' THEN 0 ELSE 1 END,
  score DESC,
  serving_document_id
"""


def build_postgres_keyword_search_sql(
    predicate: object,
) -> str:
    marker = '__RESOURCE_PREDICATE__'
    if _POSTGRES_KEYWORD_SEARCH_SQL_TEMPLATE.count(marker) != 1:
        raise RuntimeError('PostgreSQL keyword resource predicate marker is invalid')
    sql = _POSTGRES_KEYWORD_SEARCH_SQL_TEMPLATE.replace(
        marker,
        render_postgres_keyword_predicate(predicate),
    )
    for spec in POSTGRES_KEYWORD_BIND_SPECS:
        bind_marker = f'__BIND_{spec.name}__'
        occurrences = sql.count(bind_marker)
        if spec.target is PostgresKeywordBindTarget.SEARCH and occurrences:
            sql = sql.replace(
                bind_marker,
                _render_postgres_keyword_expression(KeywordParameter(spec.name)),
            )
    if '__BIND_' in sql:
        raise RuntimeError('PostgreSQL keyword bind marker is invalid')
    return sql


POSTGRES_KEYWORD_SEARCH_SQL = build_postgres_keyword_search_sql(
    POSTGRES_KEYWORD_RESOURCE_PREDICATE
)
POSTGRES_KEYWORD_TIMEOUT_SQL = (
    "SELECT set_config('statement_timeout', "
    + _render_postgres_keyword_expression(KeywordParameter('timeout_value'))
    + ', true)'
)

_SQLITE_ORACLE_MAX_PROJECTIONS = 10_000


class KeywordSearchUnavailableError(RuntimeError):
    pass


class KeywordSearchTimeoutError(KeywordSearchUnavailableError):
    pass


class SqlAlchemyKeywordSearchStore:
    """Canonical keyword candidate store for PostgreSQL and bounded SQLite smoke."""

    def __init__(self, *, db: Session, settings: Settings) -> None:
        self._db = db
        self._settings = settings

    def search(
        self,
        request: RetrievalRequest,
    ) -> tuple[ClassifiedRetrievalCandidate, ...]:
        terms = tokenize_rag_lexical_query(request.retrieval_query_text)
        if not terms:
            return ()
        try:
            if self._db.get_bind().dialect.name == 'postgresql':
                return self._postgresql_search(request, terms=terms)
            return self._sqlite_search(request, terms=terms)
        except DBAPIError as exc:
            if _is_statement_timeout(exc):
                raise KeywordSearchTimeoutError('keyword retrieval timed out') from None
            raise KeywordSearchUnavailableError('keyword retrieval is unavailable') from None
        except SQLAlchemyError:
            raise KeywordSearchUnavailableError('keyword retrieval is unavailable') from None

    def _postgresql_search(
        self,
        request: RetrievalRequest,
        *,
        terms: tuple[str, ...],
    ) -> tuple[ClassifiedRetrievalCandidate, ...]:
        bind_values = build_postgres_keyword_bind_values(
            request=request,
            terms=terms,
            settings=self._settings,
        )
        timeout_values = {
            spec.name: bind_values[spec.name]
            for spec in POSTGRES_KEYWORD_BIND_SPECS
            if spec.target is PostgresKeywordBindTarget.TIMEOUT
        }
        search_values = {
            spec.name: (list(bind_values[spec.name]) if spec.sql_type.endswith('[]') else bind_values[spec.name])
            for spec in POSTGRES_KEYWORD_BIND_SPECS
            if spec.target is PostgresKeywordBindTarget.SEARCH
        }
        self._db.execute(
            text(POSTGRES_KEYWORD_TIMEOUT_SQL),
            timeout_values,
        )
        rows = tuple(
            self._db.execute(
                text(POSTGRES_KEYWORD_SEARCH_SQL),
                search_values,
            )
            .mappings()
            .all()
        )
        candidates: list[ClassifiedRetrievalCandidate] = []
        for row in rows:
            candidate = self._canonical_candidate(
                row,
                request=request,
                supplied_score=float(row['score']),
                supplied_terms=tuple(row['matched_terms']),
            )
            if candidate is None:
                raise KeywordSearchUnavailableError('keyword retrieval is unavailable')
            if not _is_stage_one_candidate(candidate.access):
                raise KeywordSearchUnavailableError('keyword retrieval is unavailable')
            candidates.append(candidate)
        return tuple(candidates)

    def _sqlite_search(
        self,
        request: RetrievalRequest,
        *,
        terms: tuple[str, ...],
    ) -> tuple[ClassifiedRetrievalCandidate, ...]:
        generation = self._db.get(RagServingCorpusGeneration, 1)
        if generation is None:
            return ()
        rows = tuple(
            self._db.scalars(
                select(RagLexicalServingProjection)
                .where(
                    RagLexicalServingProjection.corpus_generation
                    == generation.corpus_generation,
                    RagLexicalServingProjection.lexical_contract_version
                    == RAG_LEXICAL_COMPAT_VERSION,
                    RagLexicalServingProjection.fingerprint_key_version
                    == self._settings.agent_runtime_fingerprint_key_version,
                    RagLexicalServingProjection.fingerprint_key_material_verifier
                    == fingerprint_key_material_verifier(
                        self._settings.agent_runtime_fingerprint_secret
                    ),
                    RagLexicalServingProjection.effective_permission.in_(
                        ('public', 'internal', 'restricted')
                    ),
                )
                .order_by(RagLexicalServingProjection.serving_document_id)
                .limit(_SQLITE_ORACLE_MAX_PROJECTIONS + 1)
            ).all()
        )
        if len(rows) > _SQLITE_ORACLE_MAX_PROJECTIONS:
            raise KeywordSearchUnavailableError('keyword retrieval is unavailable')
        scored: list[ClassifiedRetrievalCandidate] = []
        for row in rows:
            score, matched_terms = score_rag_lexical_candidate(
                question=request.retrieval_query_text,
                title=row.title_lower,
                text=_projection_model_text(row),
            )
            if score <= 0:
                continue
            candidate = self._canonical_candidate(
                row,
                request=request,
                supplied_score=score,
                supplied_terms=matched_terms,
            )
            if candidate is not None and _is_stage_one_candidate(candidate.access):
                scored.append(candidate)
        return tuple(sorted(scored, key=_candidate_order)[:50])

    def _canonical_candidate(
        self,
        row: Mapping[str, object] | RagLexicalServingProjection,
        *,
        request: RetrievalRequest,
        supplied_score: float,
        supplied_terms: tuple[str, ...],
    ) -> ClassifiedRetrievalCandidate | None:
        document_id = str(_row_value(row, 'serving_document_id'))
        prefix, separator, raw_id = document_id.partition(':')
        if separator != ':' or not raw_id.isascii() or not raw_id.isdecimal():
            return None
        identifier = int(raw_id)
        if identifier <= 0 or str(identifier) != raw_id:
            return None
        if prefix == 'chunk':
            observation = CanonicalSourceObservationResolver(
                db=self._db,
                settings=self._settings,
            ).resolve_for_index(identifier)
            if observation is None:
                return None
            identity = observation.identity
            evidence = observation.evidence
            access = CanonicalSourceObservationEligibilityService().classify_access(
                request.security_scope,
                observation,
            )
        elif prefix in {'decision_record', 'history_event', 'timeline_event', 'todo'}:
            envelope = TrustedServingEnvelopeResolver(
                db=self._db,
                settings=self._settings,
            ).resolve_for_index(prefix, identifier)
            if envelope is None:
                return None
            identity = envelope.identity
            evidence = envelope.evidence
            access = _classify_trusted_stage_one_access(
                TrustedEvidenceAuthorizer(
                    db=self._db,
                    settings=self._settings,
                ),
                scope=request.security_scope,
                envelope=envelope,
            )
        else:
            return None
        if not _projection_matches(row, evidence=evidence):
            return None
        oracle_score, oracle_terms = score_rag_lexical_candidate(
            question=request.retrieval_query_text,
            title=evidence.title,
            text=evidence.model_content,
        )
        if supplied_score != oracle_score or supplied_terms != oracle_terms:
            return None
        if access.permission_visibility == 'visible':
            visible = ServingEvidenceResolver(settings=self._settings).resolve_candidate(
                db=self._db,
                identity=identity,
                scope=request.security_scope,
            )
            if visible is None:
                return None
            evidence = visible
        return ClassifiedRetrievalCandidate(
            evidence=evidence,
            relevance_score=oracle_score,
            matched_terms=oracle_terms,
            access=access,
        )


def _projection_model_text(row: RagLexicalServingProjection) -> str:
    prefix = f'{row.title_lower}\n'
    if not row.searchable_lower.startswith(prefix):
        return ''
    return row.searchable_lower[len(prefix) :]


def _projection_matches(
    row: Mapping[str, object] | RagLexicalServingProjection,
    *,
    evidence: object,
) -> bool:
    return bool(
        _row_value(row, 'serving_document_id') == evidence.serving_document_id
        and _row_value(row, 'serving_kind') == evidence.serving_kind
        and _row_value(row, 'support_mode') == evidence.support_mode
        and _row_value(row, 'effective_permission') == evidence.effective_permission
        and _row_value(row, 'serving_identity_hmac')
        == evidence.serving_identity_hmac
        and _row_value(row, 'serving_version_fingerprint')
        == evidence.serving_version_fingerprint
        and _row_value(row, 'model_content_hmac') == evidence.model_content_hmac
        and _row_value(row, 'canonical_citation_projection_hmac')
        == evidence.canonical_citation_projection_hmac
        and _row_value(row, 'title_lower') == evidence.title.lower()
        and _row_value(row, 'searchable_lower')
        == f'{evidence.title}\n{evidence.model_content}'.lower()
    )


def _row_value(
    row: Mapping[str, object] | RagLexicalServingProjection,
    key: str,
) -> object:
    if isinstance(row, Mapping):
        return row[key]
    return getattr(row, key)


def _candidate_order(candidate: ClassifiedRetrievalCandidate) -> tuple[object, ...]:
    return (
        0 if candidate.evidence.serving_kind == 'trusted_knowledge' else 1,
        -candidate.relevance_score,
        candidate.evidence.serving_document_id,
    )


def _classify_trusted_stage_one_access(
    authorizer: TrustedEvidenceAuthorizer,
    *,
    scope: SecurityScope,
    envelope: TrustedServingEnvelope,
) -> EvidenceAccessClassification:
    resource_classification = authorizer.classify_resource_access(scope, envelope)
    permission = known_permission(envelope.identity.effective_permission)
    if permission is None:
        visibility = 'unknown_permission'
    elif permission in scope.allowed_permission_levels:
        visibility = 'visible'
    else:
        visibility = 'denied_known'
    return EvidenceAccessClassification(
        global_eligibility=resource_classification.global_eligibility,
        resource_scope=resource_classification.resource_scope,
        permission_visibility=visibility,
    )


def _is_stage_one_candidate(access: EvidenceAccessClassification) -> bool:
    return bool(
        access.global_eligibility == 'eligible'
        and access.resource_scope == 'in_scope'
        and access.permission_visibility in {'visible', 'denied_known'}
    )


def _is_statement_timeout(exc: DBAPIError) -> bool:
    original = exc.orig
    sqlstate = getattr(original, 'sqlstate', None) or getattr(original, 'pgcode', None)
    return sqlstate == '57014'
