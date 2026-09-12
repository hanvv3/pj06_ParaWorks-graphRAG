"""Independent, fixed-wire golden vectors for the approved legacy-v3 union."""

import base64
import hashlib
import hmac
import json
import zlib

import pytest

from backend.app.agent_runtime.fingerprints import keyed_fingerprint

SECRET = b'legacy-v3-independent-golden-secret'
POLICY = 'assistant-evidence:v1'


def _exact(value):
    """Test-owned literal UTF-8 encoding; never calls the production builder."""
    if isinstance(value, str):
        encoded = value.encode('utf-8')
        return {'byte_length': len(encoded), 'utf8_hex': encoded.hex()}
    if value is None or type(value) in {bool, int}:
        return value
    if isinstance(value, list):
        return [_exact(item) for item in value]
    return {key: _exact(item) for key, item in value.items()}


def _snapshot(**overrides):
    values = {
        'document_chunk_id': None,
        'document_version_id': None,
        'source_id': None,
        'parser_run_id': None,
        'current_document_version_id': None,
        'server_content_signature_schema': None,
        'server_content_signature': None,
        'parser_policy_version': None,
        'parser_version': None,
        'chunk_policy_version': None,
        'knowledge_type': None,
        'knowledge_id': None,
        'approval_link_id': None,
        'legacy_human_base': False,
        'legacy_source_review_item_id': None,
    }
    values.update(overrides)
    return values


def _identity(*, document, kind, role, provenance):
    return {
        'serving_document_id_bytes': _exact(document),
        'dependency_kind': kind,
        'dependency_role': role,
        'effective_permission': 'internal',
        'serving_version_fingerprint': '1',
        'model_content_hmac': '2',
        'canonical_citation_projection_hmac': '3',
        'legacy_public_source_id_bytes': _exact(document),
        'legacy_source_links_bytes': [_exact('https://e.test/근거')],
        'legacy_source_snippets_bytes': [_exact('정확한 근거')],
        'provenance': _exact(provenance),
    }


def golden_cases():
    raw = _identity(
        document='chunk:7',
        kind='raw_chunk',
        role='selected_citation',
        provenance={
            'snapshot': _snapshot(
                document_chunk_id=7,
                document_version_id=5,
                source_id=3,
                parser_run_id=11,
                current_document_version_id=5,
                server_content_signature_schema='source-signature:v1',
                server_content_signature='a',
                parser_policy_version='parser-policy:v2',
                parser_version='parser:v4',
                chunk_policy_version='chunk:v3',
            ),
            'approval': None,
            'review': None,
            'evidence_links': [],
        },
    )
    approval = {
        'id': 17,
        'knowledge_type': 'history_event',
        'knowledge_id': 9,
        'review_item_id': 13,
        'security_scope_id': 'workspace:default',
        'promotion_effect_kind': 'trusted_knowledge',
        'resolution_source': 'human',
        'claim_fingerprint': '4',
        'permission_level': 'internal',
        'fingerprint_key_version': 'c5-fingerprint:v1',
        'fingerprint_key_material_verifier': '5',
        'active': True,
    }
    review = {
        'id': 13,
        'status': 'approved',
        'permission_level': 'internal',
        'resolution_source': 'human',
        'candidate_contract_version': 'review-candidate:v2',
        'source_links': ['https://e.test/근거'],
        'source_snippets': ['정확한 근거'],
    }
    link = {
        'id': 23,
        'approval_link_id': 17,
        'canonical_source_kind': 'history_event',
        'canonical_source_id': 'history_event:9',
        'canonical_version_or_signature': '6',
        'evidence_hash': '7',
        'fingerprint_key_version': 'c5-fingerprint:v1',
        'fingerprint_key_material_verifier': '5',
    }
    explicit = _identity(
        document='history_event:9',
        kind='trusted_knowledge',
        role='selected_citation',
        provenance={
            'snapshot': _snapshot(
                knowledge_type='history_event',
                knowledge_id=9,
                approval_link_id=17,
            ),
            'approval': approval,
            'review': review,
            'evidence_links': [link],
        },
    )
    legacy_human = _identity(
        document='timeline_event:4',
        kind='trusted_knowledge',
        role='selected_citation',
        provenance={
            'snapshot': _snapshot(
                knowledge_type='timeline_event',
                knowledge_id=4,
                legacy_human_base=True,
                legacy_source_review_item_id=31,
            ),
            'approval': None,
            'review': {**review, 'id': 31},
            'evidence_links': [],
        },
    )
    unbound = _identity(
        document='decision:8',
        kind='legacy_unbound',
        role='legacy_evidence_influence',
        provenance={
            'snapshot': _snapshot(),
            'approval': None,
            'review': None,
            'evidence_links': [],
        },
    )
    email = _exact(
        {
            'action_type': 'email_draft',
            'evidence_derived': True,
            'email_draft': {
                'to': ['first@example.com', 'second@example.com'],
                'subject': '검토 요청',
                'body': '근거를 확인해 주세요.',
            },
        }
    )
    selected_child = {
        'approval_link_id': 17,
        'approval_provenance_hmac': '8',
        'candidate_ordinal': 1,
        'canonical_citation_projection_hmac': '3',
        'dependency_kind': 'trusted_knowledge',
        'dependency_role': 'selected_citation',
        'dependency_serving_scope': 'legacy_v1_only',
        'effective_permission': 'internal',
        'evidence_link_ids': [23, 29],
        'evidence_link_set_hmac': '9',
        'legacy_dependency_identity_hmac': 'a',
        'model_content_hmac': '2',
        'raw_document_chunk_id': None,
        'selected_v1_citation_projection_hmac': 'b',
        'serving_identity_hmac': None,
        'serving_version_fingerprint': None,
        'support_mode': None,
        'trusted_knowledge_id': 9,
    }
    ordered_set = {
        'assistant_message_content_hmac': 'c',
        'content_origin': 'legacy_evidence',
        'content_origin_hmac': 'd',
        'content_write_mode': 'legacy_trimmed',
        'dependencies': [
            {'candidate_ordinal': 0, 'dependency_child_hmac': 'e'},
            {'candidate_ordinal': 1, 'dependency_child_hmac': 'f'},
        ],
        'dependency_count': 2,
        'dependency_serving_scope': 'legacy_v1_only',
        'evidence_contract_version': 'assistant-evidence:v1',
        'linked_agent_run_id': None,
        'model_influence_set_hmac': None,
        'parent_selected_evidence_projection_hmac': '0',
        'rag_result_hmac': None,
    }
    return (
        ('raw', 'assistant-legacy-dependency-snapshot:v2', raw),
        ('explicit_trusted', 'assistant-legacy-dependency-snapshot:v2', explicit),
        ('legacy_human', 'assistant-legacy-dependency-snapshot:v2', legacy_human),
        ('genuine_unbound', 'assistant-legacy-dependency-snapshot:v2', unbound),
        ('uncited_email', 'assistant-legacy-action-payload:v1', email),
        ('selected_citation', 'assistant-dependency-child-hmac:v3', selected_child),
        ('ordered_multi_child', 'assistant-dependency-set-hmac:v3', ordered_set),
    )


EXPECTED = {
    'raw': (
        'eNqdVO2O4yAMfBd+b6UmNNDkVU4rRIjTcCUQAeleteq7n8lHt81e76RTJASOPWN7DJ+kcb3UllRkkF5+OH8O1Rmu0OxabU/gB69trC4ZeSODM1pdxQV80C5FyBB0iNLGHVx0A1bB7BhUB738o6OBk1TXXQMD2BRx3QUrh9A55Mgx9CLNCKT6JEpaZ7WSRigdZUQcMXj3E9S07XqpEJdixBeUOGvboBXLEKob7fn5r3cGkUkAgyDQ3HHRC9o2AV9ADOB7HZa0sXLwVhr0mPMWw1hjD0Rwo1cgdCPqa4SQ8k0bYcCeYkcq/kbG2B5FB78QhlF25AUDVlNJObnd0RYYo+05rEg/NlB59ox15Af89pxSmbd5y4oc+IEVnPJD3oKsea1wzeo9ub1viYLVwwDxJVfGnrhAyX1ZQFOW01qUKt9/Y+hdA6iRw1bZuOqSlES1LmAlDkVqjxzSGVtZ2dEY7PgyMHPxmApCeTTCx+qxzsVj9OSNbV99JpXFdi43VR1fisEZpUkONXqfsm+cGvu0WaAmpgKHaLXPfMnKH6wb77N1HwaaEzwk+mWL1wFW66JON/bSiloG/NFKE2Cr29wYoSP0D5h4XwP4f1W/0ZTvWcZzTnFi8rzBU8sUKxnl5dSOPLVjAfbjXFKW3U2vSMqXHBPqIaFiPIbfJyXok5Vx9PA942e9sr8Fi/mp+Y6xyYiylmM2KH6BVVOsmOMIZHiTkrWYspyJ1ptNKnpbiPEd/BqN/77zK9I6LQ/PK/oj+e03j4XrMg==',
        '7148191a4c4384b1f5ed8cfc02d98d9bf48d4abf38cab3db9729b914a9297924',
    ),
    'explicit_trusted': (
        'eNrVVmGPmzAM/S98vkqFlOToX5mmKARTskKCksCtOt1/nwOFttC7blr3YaqKILGfrednJ+9RYRqhdLSPWmHFm7FHtz/CCYpNqfQBbGuV9vs+jl6i1tRKnngP1ikTPIRzynmh/QZ6VYCWMBo6WUEj7hrWcBDytCmgBR08ThunResqgzESdO1F3UG0f4+k0EYrKWoulRcecXhrzQ+Qw2vVCIm4BD0uUPyodIGr3nbOQ8GP2rzVUBzg1sqaGiNEDmoEQ7MJH62gLEOAHngLtlHunD4yAFaLGi3G/Hnb5cgFd6azErgqeH7y4ELe4YXXoA++ivZx+hJ1vnzlFfxEHPpKM0bYjpYsYVla0pRRmlJgOyJIFn3M8GfcWumjm6C/LbCTeIHNdvjbMkJEUiaInSAuRsB4SQkiZ7nEZ5xvo4/vy0BOq7YF/2msmN7EAim2WQpFlg3PNJPJdhWhMQVg8Qxyp/1UsFBiLGMPWqBaAl+iDd/IbXgfuI/2WEB4iWQtVMOvVLim9yYrsgsMXtlz1DFvBNZOoYxQjapUYB+hpPdQZikvfdltEQhJk4JSmlGgDMlPsCBJ+AolZpTEAVwVo+OsTx5WsusFf2phHYz8pppCkIuA0b+HeoX2egs25BgyxsxjKgcIaxoztNvYF+f++poC1GDCBtVhLjvMKkfAkjEqw3cgJUBbcKbuBuxRgivYZeewlBaYGIzevYI3rjw0A3WBGAeys8qjpKVpR0YfJMoCcTTHVLcITGhKBA1JUxpjMBloxFjTZBubcWiNSbLDEp+qeRlY81T4+3mwAr1fgj8RxgXyLGpuLHfqoIXvLDzqDhpdc1IJVz3yYP9RVyYkTK5RXudDqFAF5jrMMYvz6fOY2a26koH1DJ8MY5MgXVRXhv84tMYQN5mnAXlKw/5dV10fOs8+bhbnzHNOGLxR+M494inGJEO5y6EJxqae7hzXB9Cim6sOv5Y3Ht3VNe511oYzrTCya8LL1EjBeTSZt0acOxtrn396HJzP+6prhOa5cAhXitrB8iawHK1jang1dGA/oeO8aTu99liY4hKuzJeCq6nz9T4fb5Sz2WXAhoWhouiJ0+FSk+fdySboqWI3N5IIZ8fHL7x0yCY=',
        '6a3f1daf715fbbdb89a502d130d629e141856ed4c4bcbbc0ac9619936188a959',
    ),
    'legacy_human': (
        'eNqtVduOozAM/Reep1K5haG/slpFbmJKtiFBSWC2Gs2/rwO0pXRGsw+jSjQ49rHjc2LeE2k7UCY5JD04eLPu7A9nvKDcNcqc0PVOmXAY0+Ql6a1W4sJHdF7ZGAHeKx/AhB2OSqIRODt60WIHnzpqPIG47CT2aGLEZecN9L61lCOj0BH0gMnhPRFgrFECNBcqQCAc3jv7B8W0bDsQhJtTxB2Kn5WRZA1u8AElPxv7plGe8NHLWU0ZEo+awMjtik9e2DQxwYi8R9cpv5RPHUBnQJPHXD/vhyP1gns7OIFcSX68BPSx7rjgGs0ptMkhZS/JEJpX3uJfwqkKVjPJSiboH1lZNqysGL1jVeSQF8nHLcGCrJU5+yv4rw16lj6gs9eqoN++ynPImoywM8KlDHlVZA3CsToKeqbHffLxe5vIG9X3GL7MtTkJCtjXJcq6np5lLbL9U4bOSiT6LHXPhCtlkWQickQDpJfYMejjO3X3YAatiYRFS/PhqRSCcmTEt0UXUkmg0iKwAxHuOtvWXD92P5u6XdOzyiTLWUocREYKlsY+5VCxPIscKFJRTs29i4AwR9RPGV4fCajZ3PCMgFMmIpRDb/UwaXbu9BNGuSWxJIlQaTF6rYKf5n9D/M9QTpc8DP67PqVU5J661EziJ9nHoy5jYK2I6eA8sjErQ7QDvW/H0LI3OBdlJq0YurhY9lfht60Z55ON55jbDJlsxdoQLv0zm2nx3zd+ddvboQPDj+AJkKYXbm/nLH+uAnb8Jk5wHt0X3Vg23bA+y2LcuJKJLLdr6tXJEIcOv9vn85S/uV1H4WyYKKVI+obcKfnJOXkFv1K2+loRQkqa+geCXFsb',
        '9dc13cfcbbef087ce732c95ff6469e95614618bafdfcb8849530397fe760e362',
    ),
    'genuine_unbound': (
        'eNqlVFuOozAQvIu/J1KABBKushpZjWmCN8a2/CCLRtx9mwB5zmg/Vghkuquq7WrbX6w2HUjNSmbBwcW4sy/POGC9aaQ+obNO6lD2'
        'Cftg1igpBt6j89JMDPBe+gA6bLCXNWqBM9CLFjv4FqjwBGLY1GhRT4xh4zVY3xqqkRK1BxWRlV9MgDZaClBcyACBdLh15jeK67Dt'
        'QJBuRoy7FD9LXVN0rsGjrkykwBPEGYV3yDptLnVDdWlEaGyaqUqP3KLrpF/WQDag06AIsbBtrMgQ7k10k0TNqyGgnyY/DbhCfQot'
        'K5PtB4uhOfAW/5BOvsv3eZYfC3rzJscMsgMbb6KLmpL67FfBXy+KafKseCh29GyLLIO0SZt8n2JBVYqs2KUNQlVUgr5JtWXj52sh'
        'r6W1GH6sleRPtVDA9rjH+ni8fvdHkW7fKnSmRuqbIcd0WHs1dZc62KOGyWdyCez0T46WOipFxq/duC6epkJSjoJ4WRHrXnlkX9Hk'
        '/ooRbaT/17265KJz05RqI2I3DZb8A/2WmnW+SbxzztpcFNYn/DYWBotrdPG+jR1oXoGnRAPK42tX5mVzGbB70KQT6tH9sLYl6aJ+'
        'Z7xAKUSRW4O8PGkI0eG/8nw+2DfYuvHnwDjOTLo27gb/76lYBVfTHy4lYiVsHP8CuNG3LQ==',
        'ddbce2829ece51df0c13127b977b2fccb722457977b515cb758dd4b6b405216f',
    ),
    'uncited_email': (
        'eNp9kdtugzAMht8l10XiEBzCq0wTchzTZmWAQmBDVd99SbubttNuIsf+/fl0EXb6RDeKVszo8Wvy56U988426914ZD97N4Z2K8RBzNPgaO829oubUgYui1sCjiHjzVkeie/ChU78iX8KBz4i7RlSiJFsxn2Y0N6zNhxWFu1F3INd2Ofb1+yBu4HHYziJtigOYg190534O4KhBgsFaKC6B6nKaIOS4noQsQE3dNZjH26Qye4vsCp/gDEaZSi+hcnZYG2ozNlqrWsmbU0T7drI6COsoo4aGX2kUcuSU8llNR9M4bXn6rlM2URwA5gnWAIwmRJtgoRJtG/PAPU4NIBWpaqUlDnUqolDW5UDQV0yVNBDAj0jmgeEquLmkpThH8h72uPvZTvL3m1sRRv8ytfrD4kPuD0=',
        'd7ae652ba07a0ace1ea3562b8165d5eb34758badfd4cb2165ed751fd9f5c0fe4',
    ),
    'selected_citation': (
        'eNp9UstygzAM/Bef4ZDk0MKvdDoe1VaCipE9tiHDZPLvlRPIq52eMPKudrXWSVk/ALFqVYAIRx/71PY4o633xAeMIRLndtqoSgXvyMx6wpjIFwakRCkD5xonssgGr8BkOhzgT6DFgFygc206crbuBjDttBPWBG5E1Z4UhBC9/GlH3Guyqt28Vfdq+SCDqOlClvbvwjbAlixk1D5aYnDCulQ9kxGWoQxZ3BT6N5rLcaEX8bsv3ROLpMpxTBmt7tkfHdoDPqOid2JWJXTSTGBr/2dUwjhJjDoZHwrc4QFKhBvt2c2Cxf2+mJlQB4wDpSUwyRxjGUIQS7RrGkm1H9tdtW0+X+8S5nWkRoiL1oObAs6U5xUFghq8RUnHiyDf6Fu5kGXQ1ptxKHXTjctT8OhcdR9bJvkn2a+yDEsEL+Jrn+vlsir6YedukDEEH7MuRtfar7e5WGvO5x9VQgg6',
        '947cf511ab8ba8a667652d6530942ef3cb8c78de5e310a50dba21599dcc62770',
    ),
    'ordered_multi_child': (
        'eNqNUtFqwzAM/Bc/N9Bub/6VMYywlVSrIwfZSQkl/z65XdqUdbDHSHen810uJqQeiI01Awick5yyPeGMoWmJO5RBiIudDmZnhhTJz25CyZQqA3KmXIBLgxMFZI83YPZH7OElMOCAXKFzk7E0xx68nd6VM0Ec0djLA+t6zBk6dD5xQf2uYBXzil5HSai7mo/YgXpbffyCrOSw2ZyFCro+BXwIFKG+x4q6OyXMxn5cjAcOFEApSQIxRGP3G9js/JFiWA+hWXYvOYe/Oa1ZPp+3aeRi7NvTMKNMWo3LPg0b59PBJY6zOl9DuAYn4Ms/KovEJwxO49ZgZGRHwVgeY9yZGlB0xK0WVFW1tx/Dt73+N5WUMaIvqnE/P0j60pEeXh+410sCnRPMY9yqLMs388zu6g==',
        '299b51c1d580f1c986b6b1d6ccf2686f0714db17f9b93db5716c5a5ddf3d66a3',
    ),
}


@pytest.mark.parametrize(('name', 'schema', 'value'), golden_cases())
def test_independent_legacy_v3_canonical_bytes_and_hmac(name, schema, value):
    canonical = json.dumps(
        {
            'domain': 'paraworks:keyed-fingerprint:v1',
            'policy_version': POLICY,
            'schema_version': schema,
            'value': value,
        },
        ensure_ascii=False,
        allow_nan=False,
        separators=(',', ':'),
        sort_keys=True,
    ).encode('utf-8')
    expected_base64, expected_hmac = EXPECTED[name]
    assert canonical == zlib.decompress(base64.b64decode(expected_base64))
    assert hmac.new(SECRET, canonical, hashlib.sha256).hexdigest() == expected_hmac
    assert (
        keyed_fingerprint(
            value,
            secret=SECRET,
            schema_version=schema,
            policy_version=POLICY,
        )
        == expected_hmac
    )
