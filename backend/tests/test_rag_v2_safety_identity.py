from __future__ import annotations

from dataclasses import replace

import pytest

from backend.app.agent_runtime.rag_runtime_contracts import (
    admission_source_window,
    terminal_source_window,
)
from backend.app.agent_runtime.rag_safety_identity import (
    FinalProductIdentityInput,
    final_product_identity,
)


def test_source_window_mapping_is_closed_and_phase_specific():
    assert admission_source_window(
        mode='enforce', surface='assistant', backend='pgvector'
    ) == 'rag-v2:admission:enforce:assistant:pgvector'
    assert terminal_source_window(
        stage='product', surface='ask', backend='keyword'
    ) == 'rag-v2:ask:keyword'
    assert terminal_source_window(
        stage='shadow', surface='search', backend='pgvector'
    ) == 'rag-v2:shadow:search:pgvector'
    assert terminal_source_window(
        stage='final_error', surface='assistant', backend='keyword'
    ) == 'rag-v2:final-error:assistant:keyword'
    with pytest.raises(ValueError):
        admission_source_window(
            mode='shadow', surface='ask', backend='keyword'
        )


def test_typed_final_product_identity_rejects_forgery_and_has_mutation_golden():
    material = FinalProductIdentityInput(
        admission_cache_identity_hmac='1' * 64,
        rag_result_hmac='2' * 64,
        surface='ask',
    )
    identity = final_product_identity(material, secret=b'identity-secret')
    assert identity == '392b11114d5c93121843c0dd2b314cb7140af77f3dac7a1f79a128e4c8e21906'
    assert final_product_identity(
        replace(material, surface='search'), secret=b'identity-secret'
    ) != identity
    with pytest.raises((TypeError, ValueError)):
        final_product_identity({'surface': 'ask'}, secret=b'identity-secret')
