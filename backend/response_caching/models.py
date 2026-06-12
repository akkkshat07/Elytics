from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional
import hashlib
import json

@dataclass
class SemanticSignature:
    semantic_variant: str = ''
    operation_type: str = 'analytical'
    modifiers: Dict[str, Any] = field(default_factory=dict)
    query_type: str = 'analytical'
    primary_entity: str = 'item'
    aggregation: str = 'none'
    grouping_dimensions: List[str] = field(default_factory=list)
    filter_types: List[str] = field(default_factory=list)
    query_routing_type: str = 'data_analyst'

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SemanticSignature':
        return cls(**data)

    def to_cache_key(self) -> str:
        l1 = f'{self.semantic_variant}_{self.operation_type}'
        l2 = f'{self.primary_entity}_{self.aggregation}'
        l3 = '_'.join(sorted(self.grouping_dimensions)) if self.grouping_dimensions else 'none'
        l4 = '_'.join(sorted(self.filter_types)) if self.filter_types else 'none'
        return f'{l1}|{l2}|{l3}|{l4}'

    def similarity_score(self, other: 'SemanticSignature') -> float:
        if not isinstance(other, SemanticSignature):
            return 0.0
        score = 0.0
        if self.semantic_variant == other.semantic_variant:
            score += 0.3
        elif self.semantic_variant and other.semantic_variant:
            if self.semantic_variant in other.semantic_variant or other.semantic_variant in self.semantic_variant:
                score += 0.15
        if self.operation_type == other.operation_type:
            score += 0.25
        elif self.operation_type and other.operation_type:
            if self.operation_type in other.operation_type or other.operation_type in self.operation_type:
                score += 0.12
        if self.primary_entity == other.primary_entity:
            score += 0.2
        elif self.primary_entity and other.primary_entity:
            if self.primary_entity in other.primary_entity or other.primary_entity in self.primary_entity:
                score += 0.1
        if self.aggregation == other.aggregation:
            score += 0.15
        elif self.aggregation and other.aggregation:
            if self.aggregation in other.aggregation or other.aggregation in self.aggregation:
                score += 0.07
        if set(self.grouping_dimensions) == set(other.grouping_dimensions):
            score += 0.1
        elif self.grouping_dimensions and other.grouping_dimensions:
            overlap = len(set(self.grouping_dimensions) & set(other.grouping_dimensions))
            total = max(len(self.grouping_dimensions), len(other.grouping_dimensions))
            if total > 0:
                score += 0.1 * (overlap / total)
        return min(1.0, score)

    def is_exact_match(self, other: 'SemanticSignature') -> bool:
        return self.semantic_variant == other.semantic_variant and self.operation_type == other.operation_type and (self.aggregation == other.aggregation) and (set(self.grouping_dimensions) == set(other.grouping_dimensions))

@dataclass
class GlobalCacheKey:
    semantic_signature: SemanticSignature
    embedding_hash: str

    def to_string(self) -> str:
        signature_key = self.semantic_signature.to_cache_key()
        return f'{signature_key}_{self.embedding_hash[:12]}'

    @classmethod
    def from_embedding(cls, semantic_signature: SemanticSignature, embedding: List[float]) -> 'GlobalCacheKey':
        embedding_str = json.dumps(embedding[:128] if len(embedding) > 128 else embedding, sort_keys=True)
        embedding_hash = hashlib.sha256(embedding_str.encode()).hexdigest()
        return cls(semantic_signature=semantic_signature, embedding_hash=embedding_hash)

@dataclass
class TenantCacheKey:
    global_key: GlobalCacheKey
    client_id: str

    def to_string(self) -> str:
        return f'{self.client_id}_{self.global_key.to_string()}'

    @classmethod
    def create(cls, client_id: str, semantic_signature: SemanticSignature, embedding: List[float]) -> 'TenantCacheKey':
        global_key = GlobalCacheKey.from_embedding(semantic_signature, embedding)
        return cls(global_key=global_key, client_id=client_id)