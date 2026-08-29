def canonical_knowledge_text(knowledge_type: str, target: object) -> str:
    canonical_type = (
        'decision_record' if knowledge_type == 'decision' else knowledge_type
    )
    try:
        fields = {
            'decision_record': ('title', 'decision_summary'),
            'history_event': ('title', 'reason'),
            'timeline_event': ('title', 'result_summary'),
            'todo': ('title', 'priority', 'priority_reason'),
        }[canonical_type]
    except KeyError:
        raise ValueError('trusted knowledge type is unsupported') from None
    return '\n'.join(str(getattr(target, field)) for field in fields)
