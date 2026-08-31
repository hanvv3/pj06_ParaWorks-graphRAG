from __future__ import annotations

from datetime import UTC, datetime

import pytest
from sqlalchemy import create_engine, inspect
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

import backend.app.models as models
from backend.app.db.base import Base

CORPUS_COLUMNS = {
    'id',
    'corpus_generation',
    'vector_index_generation',
    'embedding_model',
    'embedding_dimensions',
    'index_policy_version',
    'pgvector_cosine_policy_version',
    'fingerprint_key_version',
    'fingerprint_key_material_verifier',
    'updated_at',
}

LEXICAL_COLUMNS = {
    'id',
    'corpus_generation_id',
    'corpus_generation',
    'serving_document_id',
    'serving_kind',
    'support_mode',
    'effective_permission',
    'serving_identity_hmac',
    'serving_version_fingerprint',
    'model_content_hmac',
    'canonical_citation_projection_hmac',
    'title_lower',
    'searchable_lower',
    'lexical_contract_version',
    'fingerprint_key_version',
    'fingerprint_key_material_verifier',
    'created_at',
    'updated_at',
}

VECTOR_D_COLUMNS = {
    'corpus_generation_id',
    'serving_kind',
    'support_mode',
    'effective_permission',
    'serving_identity_hmac',
    'serving_version_fingerprint',
    'model_content_hmac',
    'canonical_citation_projection_hmac',
    'index_policy_version',
    'pgvector_cosine_policy_version',
    'cosine_indexable',
    'vector_index_generation',
    'vector_index_state_hmac',
    'fingerprint_key_version',
    'fingerprint_key_material_verifier',
}


def _valid_corpus(**overrides):
    values = {
        'id': 1,
        'corpus_generation': 0,
        'vector_index_generation': 0,
        'embedding_model': 'text-embedding-3-small',
        'embedding_dimensions': 1536,
        'index_policy_version': 'rag-v2-serving-index:v1',
        'pgvector_cosine_policy_version': 'pgvector-cosine-indexable:v1',
        'fingerprint_key_version': 'runtime-key-v1',
        'fingerprint_key_material_verifier': 'a' * 64,
    }
    values.update(overrides)
    return models.RagServingCorpusGeneration(**values)


def _valid_projection(**overrides):
    values = {
        'corpus_generation_id': 1,
        'corpus_generation': 0,
        'serving_document_id': 'chunk:1',
        'serving_kind': 'raw_chunk',
        'support_mode': 'source_observation',
        'effective_permission': 'internal',
        'serving_identity_hmac': 'b' * 64,
        'serving_version_fingerprint': 'c' * 64,
        'model_content_hmac': 'd' * 64,
        'canonical_citation_projection_hmac': 'e' * 64,
        'title_lower': 'migration plan',
        'searchable_lower': 'migration plan\nmove to postgresql',
        'lexical_contract_version': 'rag-keyword-lexical-compat:v1',
        'fingerprint_key_version': 'runtime-key-v1',
        'fingerprint_key_material_verifier': 'a' * 64,
    }
    values.update(overrides)
    return models.RagLexicalServingProjection(**values)


def _valid_d_vector(**overrides):
    values = {
        'document_id': 'chunk:1',
        'embedding_model': 'text-embedding-3-small',
        'embedding_dimensions': 1536,
        'content_hash': 'f' * 64,
        'status': 'indexed',
        'corpus_generation_id': 1,
        'serving_kind': 'raw_chunk',
        'support_mode': 'source_observation',
        'effective_permission': 'internal',
        'serving_identity_hmac': 'b' * 64,
        'serving_version_fingerprint': 'c' * 64,
        'model_content_hmac': 'd' * 64,
        'canonical_citation_projection_hmac': 'e' * 64,
        'index_policy_version': 'rag-v2-serving-index:v1',
        'pgvector_cosine_policy_version': 'pgvector-cosine-indexable:v1',
        'cosine_indexable': True,
        'vector_index_generation': 0,
        'vector_index_state_hmac': '1' * 64,
        'fingerprint_key_version': 'runtime-key-v1',
        'fingerprint_key_material_verifier': 'a' * 64,
    }
    values.update(overrides)
    return models.VectorIndexState(**values)


def test_rag_serving_models_export_exact_additive_schema() -> None:
    assert hasattr(models, 'RagServingCorpusGeneration')
    assert hasattr(models, 'RagLexicalServingProjection')

    corpus = Base.metadata.tables['rag_serving_corpus_generations']
    lexical = Base.metadata.tables['rag_lexical_serving_projections']
    vector = Base.metadata.tables['vector_index_states']

    assert set(corpus.c.keys()) == CORPUS_COLUMNS
    assert set(lexical.c.keys()) == LEXICAL_COLUMNS
    assert set(vector.c.keys()) >= VECTOR_D_COLUMNS
    assert all(vector.c[name].nullable for name in VECTOR_D_COLUMNS)


def test_sqlite_enforces_singleton_generations_and_lexical_identity() -> None:
    engine = create_engine('sqlite://')
    Base.metadata.create_all(engine)

    with Session(engine) as db:
        db.add(_valid_corpus())
        db.commit()
        db.add(_valid_projection())
        db.commit()

        db.add(_valid_projection(serving_identity_hmac='2' * 64))
        with pytest.raises(IntegrityError):
            db.commit()
        db.rollback()

        db.add(
            _valid_projection(
                serving_document_id='chunk:2',
                serving_version_fingerprint='3' * 64,
                model_content_hmac='4' * 64,
                canonical_citation_projection_hmac='5' * 64,
            )
        )
        with pytest.raises(IntegrityError):
            db.commit()
        db.rollback()

        db.add(
            _valid_projection(
                serving_document_id='chunk:01',
                serving_identity_hmac='6' * 64,
                serving_version_fingerprint='7' * 64,
                model_content_hmac='8' * 64,
                canonical_citation_projection_hmac='9' * 64,
            )
        )
        with pytest.raises(IntegrityError):
            db.commit()
        db.rollback()

        db.add(
            _valid_projection(
                serving_document_id='chunk:3',
                serving_identity_hmac='6' * 64,
                serving_version_fingerprint='7' * 64,
                model_content_hmac='8' * 63,
                canonical_citation_projection_hmac='9' * 64,
            )
        )
        with pytest.raises(IntegrityError):
            db.commit()
        db.rollback()

        db.add(
            _valid_projection(
                serving_document_id='chunk:4',
                serving_identity_hmac='6' * 64,
                serving_version_fingerprint='7' * 64,
                model_content_hmac='8' * 64,
                canonical_citation_projection_hmac='9' * 64,
                effective_permission='unknown',
            )
        )
        with pytest.raises(IntegrityError):
            db.commit()
        db.rollback()

        db.add(_valid_corpus(id=2))
        with pytest.raises(IntegrityError):
            db.commit()
        db.rollback()

        corpus = db.get(models.RagServingCorpusGeneration, 1)
        assert corpus is not None
        corpus.corpus_generation = -1
        with pytest.raises(IntegrityError):
            db.commit()


def test_vector_state_keeps_historical_rows_unbound_and_requires_complete_d_provenance() -> (
    None
):
    engine = create_engine('sqlite://')
    Base.metadata.create_all(engine)

    with Session(engine) as db:
        db.add(_valid_corpus())
        db.add(
            models.VectorIndexState(
                document_id='legacy:1',
                embedding_model='legacy:8',
                embedding_dimensions=8,
                content_hash='0' * 64,
                status='indexed',
                indexed_at=datetime.now(UTC),
            )
        )
        db.commit()

        legacy = (
            db.query(models.VectorIndexState).filter_by(document_id='legacy:1').one()
        )
        assert all(getattr(legacy, name) is None for name in VECTOR_D_COLUMNS)

        db.add(
            models.VectorIndexState(
                document_id='partial:1',
                embedding_model='text-embedding-3-small',
                embedding_dimensions=1536,
                content_hash='1' * 64,
                status='indexed',
                serving_kind='raw_chunk',
            )
        )
        with pytest.raises(IntegrityError):
            db.commit()
        db.rollback()

        db.add(_valid_d_vector())
        db.commit()
        assert (
            db.query(models.VectorIndexState)
            .filter_by(document_id='chunk:1')
            .one()
            .cosine_indexable
        )


def test_metadata_declares_serving_foreign_keys_indexes_and_named_checks() -> None:
    engine = create_engine('sqlite://')
    Base.metadata.create_all(engine)
    schema = inspect(engine)

    lexical_indexes = {
        item['name'] for item in schema.get_indexes('rag_lexical_serving_projections')
    }
    assert lexical_indexes == {
        'ix_rag_lexical_serving_projection_permission',
        'ix_rag_lexical_serving_projection_searchable_lower',
        'ix_rag_lexical_serving_projection_support_mode',
    }
    lexical_fks = {
        item['name']
        for item in schema.get_foreign_keys('rag_lexical_serving_projections')
    }
    vector_fks = {
        item['name'] for item in schema.get_foreign_keys('vector_index_states')
    }
    assert lexical_fks == {'fk_rag_lexical_serving_projection_corpus'}
    assert vector_fks == {'fk_vector_index_state_rag_serving_corpus'}

    corpus_checks = {
        item['name']
        for item in schema.get_check_constraints('rag_serving_corpus_generations')
    }
    lexical_checks = {
        item['name']
        for item in schema.get_check_constraints('rag_lexical_serving_projections')
    }
    vector_checks = {
        item['name'] for item in schema.get_check_constraints('vector_index_states')
    }
    assert {
        'ck_rag_serving_corpus_singleton',
        'ck_rag_serving_corpus_generations_nonnegative',
        'ck_rag_serving_corpus_policy_identity',
        'ck_rag_serving_corpus_key_identity',
    } <= corpus_checks
    assert {
        'ck_rag_lexical_serving_identity',
        'ck_rag_lexical_serving_hmacs',
        'ck_rag_lexical_serving_contract',
        'ck_rag_lexical_serving_permission_support',
        'ck_rag_lexical_serving_generation',
    } <= lexical_checks
    assert {
        'ck_vector_index_state_d_provenance',
        'ck_vector_index_state_d_policy',
        'ck_vector_index_state_d_hmacs',
    } <= vector_checks
