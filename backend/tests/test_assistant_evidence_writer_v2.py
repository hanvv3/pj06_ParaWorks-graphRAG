from __future__ import annotations

from types import SimpleNamespace

import pytest

from backend.app.assistant.evidence_persistence import (
    AssistantEvidenceWriter,
    AssistantMessageProjection,
)
from backend.app.rag.evidence_projection import (
    CanonicalEvidenceProjector,
    V1EvidenceProjection,
)
from backend.app.rag.retrieval import rank_evidence_slots
from backend.tests.test_rag_v2_projection import (
    _candidate,
    _fence,
    _projection,
    _scope,
    _settings,
    _Transaction,
)


class _Session:
    def __init__(self) -> None:
        self.added: list[object] = []
        self.flush_count = 0
        self.commit_count = 0

    def add(self, value: object) -> None:
        self.added.append(value)

    def add_all(self, values) -> None:
        self.added.extend(values)

    def flush(self, _values=None) -> None:
        self.flush_count += 1
        for value in self.added:
            if getattr(value, 'id', None) is None:
                value.id = 41

    def commit(self) -> None:
        self.commit_count += 1


def _empty_projection() -> V1EvidenceProjection:
    return V1EvidenceProjection((), (), (), (), (), 'a' * 64)


def test_v2_writer_preserves_exact_utf8_bytes_and_never_commits() -> None:
    db = _Session()
    writer = AssistantEvidenceWriter(
        fingerprint_secret=b'test-finalization-secret',
        fingerprint_key_version='test-v1',
    )
    projection = AssistantMessageProjection(
        content='  정확한 답변\n',
        metadata={'status': 'supported'},
        evidence=_empty_projection(),
        permission_level=None,
        permission_notice='evidence_unavailable',
        hidden_match_count=0,
        assembled_answer_hmac=None,
        canned_message_identity='rag-canned-evidence-unavailable:v1',
        result_hmac='b' * 64,
        write_mode='rag_v2_exact',
        model_influence=(),
    )

    message = writer.append_final(
        db=db,
        conversation=SimpleNamespace(id=7, user_id='owner', updated_at=None),
        projection=projection,
        pending=SimpleNamespace(parent_agent_run_id=9),
    )

    assert message.content.encode() == '  정확한 답변\n'.encode()
    assert message.content_write_mode == 'rag_v2_exact'
    assert message.content_origin == 'rag_canned'
    assert message.linked_agent_run_id == 9
    assert message.rag_result_hmac == 'b' * 64
    assert db.flush_count >= 1
    assert db.commit_count == 0


def test_v2_writer_rejects_blank_and_assembled_canned_identity_ambiguity() -> None:
    base = {
        'metadata': {},
        'evidence': _empty_projection(),
        'permission_level': None,
        'permission_notice': None,
        'hidden_match_count': 0,
        'result_hmac': 'b' * 64,
        'write_mode': 'rag_v2_exact',
        'model_influence': (),
    }
    with pytest.raises(ValueError, match='nonblank'):
        AssistantMessageProjection(
            content=' \n ', assembled_answer_hmac=None,
            canned_message_identity='rag-canned-no-evidence:v1', **base
        )
    with pytest.raises(ValueError, match='exactly one'):
        AssistantMessageProjection(
            content='x', assembled_answer_hmac='c' * 64,
            canned_message_identity='rag-canned-no-evidence:v1', **base
        )


def test_public_writer_has_no_mode_switch() -> None:
    import inspect

    signature = inspect.signature(AssistantEvidenceWriter.append_final)
    assert 'write_mode' not in signature.parameters


def test_v2_projection_rejects_transient_query_or_scope_metadata() -> None:
    with pytest.raises(ValueError, match='transient'):
        AssistantMessageProjection(
            content='answer',
            metadata={
                'retrieval_query_text': 'secret query',
                'principal_subject': 'private owner',
            },
            evidence=_empty_projection(),
            permission_level=None,
            permission_notice=None,
            hidden_match_count=0,
            assembled_answer_hmac=None,
            canned_message_identity='rag-canned-no-evidence:v1',
            result_hmac='b' * 64,
            write_mode='rag_v2_exact',
            model_influence=(),
        )


def test_v2_writer_persists_all_model_influence_roles_with_one_whole_set() -> None:
    rows = (_projection(1), _projection(2))
    transaction = _Transaction()
    projector = CanonicalEvidenceProjector(
        db=transaction,
        settings=_settings(),
        resolver=type(
            '_Resolver',
            (),
            {
                'resolve_projection_candidate_strict': lambda self, *, db, identity, scope: next(
                    (
                        row
                        for row in rows
                        if row.identity == identity
                    ),
                    None,
                )
            },
        )(),
    )
    slots = rank_evidence_slots(
        (_candidate(rows[0], 0.9), _candidate(rows[1], 0.8))
    )
    prepared = projector.prepare_model_influence(
        slots,
        scope=_scope(),
        prepared_corpus_generation=7,
        prepared_index_generation=4,
        prepared_readiness_hmac='a' * 64,
        rendered_input_hmac='1' * 64,
    )
    fence = _fence(hidden_hmac='a' * 64)
    dependencies = projector.finalize_model_influence_dependencies(
        prepared, ('E1',), scope=_scope(), fence=fence
    )
    evidence = projector.project_selected(
        slots, selected_slot_ids=('E1',), scope=_scope(), fence=fence
    )
    db = _Session()
    writer = AssistantEvidenceWriter(
        fingerprint_secret=b'task-nine-fingerprint-secret',
        fingerprint_key_version='task-nine-v1',
        settings=_settings(),
    )
    message = writer.append_final(
        db=db,
        conversation=SimpleNamespace(id=7, user_id='owner', updated_at=None),
        projection=AssistantMessageProjection(
            content=' exact ',
            metadata={},
            evidence=evidence,
            permission_level='internal',
            permission_notice=None,
            hidden_match_count=0,
            assembled_answer_hmac='c' * 64,
            canned_message_identity=None,
            result_hmac='b' * 64,
            write_mode='rag_v2_exact',
            model_influence=dependencies,
        ),
        pending=SimpleNamespace(parent_agent_run_id=9),
    )
    dependency_rows = [
        value
        for value in db.added
        if value.__class__.__name__ == 'AssistantMessageEvidenceDependency'
    ]
    assert [value.dependency_role for value in dependency_rows] == [
        'selected_citation',
        'unselected_model_influence',
    ]
    assert all(
        value.dependency_set_hmac == message.dependency_set_hmac
        for value in dependency_rows
    )
    assert message.serving_dependency_count == 2
