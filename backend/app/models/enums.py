from enum import StrEnum


class SourceType(StrEnum):
    drive = 'drive'
    gmail = 'gmail'
    slack = 'slack'
    calendar = 'calendar'


class ReviewStatus(StrEnum):
    pending_review = 'pending_review'
    approved = 'approved'
    rejected = 'rejected'
    needs_more_evidence = 'needs_more_evidence'
    revoked = 'revoked'


class AutoReviewRevocationReason(StrEnum):
    business_withdrawal = 'business_withdrawal'
    incorrect_content = 'incorrect_content'
    permission_violation = 'permission_violation'
    wrong_source_version = 'wrong_source_version'
    policy_violation = 'policy_violation'


class KnowledgeType(StrEnum):
    decision_record = 'decision_record'
    history_event = 'history_event'
    timeline_event = 'timeline_event'
    todo = 'todo'
