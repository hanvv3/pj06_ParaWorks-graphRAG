from __future__ import annotations

import math
import struct
from collections.abc import Sequence
from typing import TypeAlias

from backend.app.rag.serving_generation import RAG_COSINE_POLICY_VERSION

CanonicalFloat32Vector: TypeAlias = tuple[float, ...]


class CosineIndexableVectorValidator:
    policy_version = RAG_COSINE_POLICY_VERSION

    def validate(
        self,
        vector: Sequence[object],
        *,
        expected_dimensions: int,
    ) -> CanonicalFloat32Vector:
        if (
            type(expected_dimensions) is not int
            or expected_dimensions <= 0
            or len(vector) != expected_dimensions
        ):
            raise ValueError('vector must be an exact cosine-indexable float32 vector')
        canonical: list[float] = []
        try:
            for coordinate in vector:
                if type(coordinate) not in {int, float}:
                    raise ValueError
                numeric = float(coordinate)
                if not math.isfinite(numeric):
                    raise ValueError
                float32 = struct.unpack('>f', struct.pack('>f', numeric))[0]
                if not math.isfinite(float32):
                    raise ValueError
                canonical.append(float32)
        except (OverflowError, TypeError, ValueError, struct.error):
            raise ValueError(
                'vector must be an exact cosine-indexable float32 vector'
            ) from None
        if not any(coordinate != 0.0 for coordinate in canonical):
            raise ValueError('vector must be an exact cosine-indexable float32 vector')
        return tuple(canonical)

    def validate_batch(
        self,
        vectors: Sequence[Sequence[object]],
        *,
        expected_dimensions: int,
    ) -> tuple[CanonicalFloat32Vector, ...]:
        return tuple(
            self.validate(vector, expected_dimensions=expected_dimensions)
            for vector in vectors
        )
