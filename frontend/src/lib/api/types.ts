export type SyncJob = {
  job_id: string;
  connector_type: string;
  status: string;
  message: string;
  progress_pct: number;
};

export type DashboardResponse = {
  source_counts: Record<string, number>;
  pending_review_count: number;
  recent_jobs: SyncJob[];
  pending_items: {
    id: number;
    title: string;
    item_type: string;
    category: string;
    confidence_score: number;
    review_url?: string;
  }[];
  today_todos: {
    id: number;
    title: string;
    assignee: string;
    due_date: string;
    category: string;
    priority?: string;
    completed_at?: string | null;
  }[];
  today_events?: {
    id: number;
    title: string;
    start: string;
    end: string;
    location: string;
    organizer: string;
    attendee_summary: string;
    source_url: string;
    permission_level: string;
  }[];
  calendar_events?: {
    id: number;
    title: string;
    start: string;
    end: string;
    location: string;
    organizer: string;
    attendee_summary: string;
    source_url: string;
    permission_level: string;
  }[];
  assigned_projects?: {
    project_key: string;
    name: string;
    summary: string;
    evidence_count: number;
    activity_count: number;
    pending_review_count: number;
    latest_timestamp: string;
    permission_level: string;
  }[];
  recent_decisions: {
    id: number;
    title: string;
    summary: string;
    created_at: string;
  }[];
  recent_timeline: {
    id: number;
    title: string;
    summary: string;
    created_at: string;
    confidence_score: number;
    source_links: string[];
  }[];
};

export type DemoUser = {
  id: string;
  email: string;
  role: "employee" | "reviewer" | "manager" | "admin" | string;
  permission_levels: string[];
  name: string;
  title: string;
  department: string;
  status?: string;
  avatar_url?: string | null;
};

export type AuthUserResponse = {
  user: DemoUser;
};

export type AuthUsersResponse = {
  users: DemoUser[];
};

export type GoogleLoginUrlResponse = {
  configured: boolean;
  login_url?: string | null;
  state?: string | null;
  required_scopes: string[];
  redirect_uri?: string | null;
  missing_config?: string[];
};

export type AuditLog = {
  id: number;
  actor_id: string;
  actor_email: string;
  actor_role: string;
  action: string;
  target_type: string;
  target_id?: string | null;
  status: string;
  metadata: Record<string, unknown>;
  created_at: string;
};

export type AuditLogsResponse = {
  logs: AuditLog[];
};

export type AgentRunSummaryItem = {
  id: number;
  agent_name: string;
  prompt_version: string;
  status: string;
  source_window: string;
  cache_key: string;
  model_name: string;
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
  token_usage?: {
    input_tokens: number;
    output_tokens: number;
    total_tokens: number;
  };
  estimated_cost_usd: number;
  permission_level: string;
  metadata: Record<string, unknown>;
  selection_strategy?: string | null;
  evidence_summary?: AgentRunEvidenceSummaryItem[];
  started_at: string;
  completed_at?: string | null;
};

export type AgentRunEvidenceSummaryItem = {
  rank: number;
  source_id: string;
  source_url?: string | null;
  timestamp?: string | null;
  author?: string | null;
  permission_level?: string | null;
  channel_id?: string | null;
  importance_score?: number;
  snippet: string;
};

export type AgentRunsResponse = {
  total_runs: number;
  total_tokens: number;
  estimated_cost_usd: number;
  recent_runs: AgentRunSummaryItem[];
};

export type AgentRunAgentSummary = {
  agent_name: string;
  run_count: number;
  total_tokens: number;
  estimated_cost_usd: number;
  average_tokens_per_run: number;
  latest_run_id: number;
  latest_status: string;
};

export type AgentRunSummaryResponse = {
  totals: {
    total_runs: number;
    total_tokens: number;
    estimated_cost_usd: number;
    average_tokens_per_run: number;
    average_cost_per_run: number;
    cache_hits: number;
    cache_hit_rate: number;
  };
  by_status: Record<string, number>;
  by_agent: AgentRunAgentSummary[];
};

export type OrchestrationStatusResponse = {
  workflow_name: string;
  backend: string;
  node_names: string[];
  graph_mermaid: string;
  cost_policy: {
    delta_sync: boolean;
    source_hash_skip: boolean;
    evidence_cache_reuse?: boolean;
    evidence_token_budget: boolean;
    per_run_budget_usd?: number;
    budget_actions?: string[];
    paid_llm_calls_in_status_api: boolean;
    requires_explicit_run: boolean;
    hitl_checkpointing?: boolean;
    checkpoint_store?: string;
    trusted_knowledge_requires_approval?: boolean;
  };
};

export type OrchestrationDryRunResponse = {
  workflow_name: string;
  backend: string;
  objective: string;
  inputs: Record<string, unknown>;
  completed_nodes: string[];
  outputs: Record<string, string>;
  token_cost_usd: number;
  cost_policy: OrchestrationStatusResponse["cost_policy"];
};

export type RagIndexingJobSummary = {
  job_id: string;
  connector_type: string;
  status: string;
  message: string;
  failure_reason?: string | null;
  progress_pct: number;
  indexed_count: number;
  skipped_count: number;
  saved_embedding_calls: number;
  updated_at: string;
};

export type EmbeddingBudgetDecision = {
  embedding_model: string;
  changed_document_count: number;
  estimated_input_tokens: number;
  estimated_cost_usd: number;
  budget_limit_usd?: number | null;
  budget_status: string;
  action: string;
  reason: string;
};

export type RagReindexResponse = {
  dry_run: boolean;
  indexed_count: number;
  skipped_count: number;
  saved_embedding_calls: number;
  embedding_request_count: number;
  embedding_prompt_tokens: number;
  embedding_total_tokens: number;
  embedding_dimensions: number;
  document_ids: string[];
  skipped_document_ids: string[];
  incremental: boolean;
  storage_backend: string;
  embedding_budget: EmbeddingBudgetDecision;
  parser_status_counts: Record<string, number>;
};

export type RagIndexingSummaryResponse = {
  state_counts: Record<string, number>;
  latest_jobs: RagIndexingJobSummary[];
  cost_policy: {
    embedding_model: string;
    embedding_input_cost_per_1m_tokens: number;
    max_estimated_embedding_cost_usd?: number | null;
    preflight_budget_gate: boolean;
    incremental_hash_skip: boolean;
  };
};

export type KnowledgeItem = {
  id: number;
  title: string;
  summary: string;
  priority?: string;
  source_links: string[];
  source_snippets: string[];
  confidence_score: number;
  permission_level: string;
  review_status: string;
  created_at: string;
};

export type KnowledgeResponse = {
  counts: {
    decisions: number;
    history_events: number;
    timeline_events: number;
    todos: number;
  };
  decisions: KnowledgeItem[];
  history_events: KnowledgeItem[];
  timeline_events: KnowledgeItem[];
  todos: KnowledgeItem[];
};

export type KnowledgeMapNode = {
  id: string;
  type: "decision" | "history_event" | "timeline_event" | "todo" | "evidence_source" | string;
  label: string;
  summary?: string;
  source_url?: string;
  permission_level: string;
  confidence_score?: number;
  review_status?: string;
  created_at?: string;
  source_count?: number;
  href?: string;
  connected_memory_count?: number;
  snippet_count?: number;
};

export type KnowledgeMapEdge = {
  source: string;
  target: string;
  relationship: string;
  permission_level: string;
};

export type KnowledgeMapResponse = {
  counts: {
    memory_nodes: number;
    evidence_nodes: number;
    edges: number;
    permission_levels: Record<string, number>;
  };
  nodes: KnowledgeMapNode[];
  edges: KnowledgeMapEdge[];
  cost_policy: {
    paid_llm_calls: boolean;
    embedding_calls: boolean;
    sync_jobs_triggered: boolean;
    strategy: string;
  };
};

export type NotificationItem = {
  id: string;
  category: "review" | "agent_run" | string;
  severity: "info" | "warning" | "error" | string;
  title: string;
  message: string;
  action_href: string;
  source_count: number;
  created_at?: string | null;
};

export type NotificationsResponse = {
  counts: {
    total: number;
    review: number;
    agent_runs: number;
  };
  notifications: NotificationItem[];
};

export type ReviewStatus =
  | "pending_review"
  | "approved"
  | "rejected"
  | "needs_more_evidence";

export type ReviewItemUpdate = {
  payload?: Record<string, unknown>;
  source_links?: string[];
  source_snippets?: string[];
  confidence_score?: number;
  permission_level?: string;
};

export type ReviewEvidenceRequest = {
  note?: string;
};

export type ReviewSourceEvidence = {
  index: number;
  rank: number;
  source_id?: string | null;
  source_url?: string | null;
  source_type?: string | null;
  source_snippet: string;
  permission_level: string;
  confidence_score: number;
  importance_score: number;
  timestamp?: string | null;
  author?: string | null;
  agent_run_id?: number | null;
  parser_status?: string | null;
  section_path?: string | null;
  evidence_reason?: string | null;
  calendar_id?: string | null;
  calendar_name?: string | null;
  calendar_start?: string | null;
  calendar_end?: string | null;
  calendar_location?: string | null;
  calendar_organizer?: string | null;
  calendar_attendee_summary?: string | null;
  event_context_key?: string | null;
};

export type ReviewAgentRunDetails = {
  model_name?: string | null;
  prompt_version?: string | null;
  estimated_cost_usd?: number | null;
  total_tokens?: number | null;
};

export type ReviewItem = {
  id: number;
  item_type: string;
  payload: Record<string, unknown>;
  source_links: string[];
  source_snippets: string[];
  source_evidence?: ReviewSourceEvidence[];
  agent_run_id?: number | null;
  agent_run_details?: ReviewAgentRunDetails | null;
  confidence_score: number;
  permission_level: string;
  status: ReviewStatus;
  reviewer_id?: string | null;
};

export type ReviewPromotionResult = {
  target_type: string;
  created_record_ids: number[];
  created_timeline_event_ids: number[];
  project_key?: string | null;
  next_routes: string[];
};

export type ReviewApprovalResponse = ReviewItem & {
  promotion_result?: ReviewPromotionResult;
  replayed: boolean;
  promotion: ReviewTransitionPromotion | null;
};

export type ReviewTransitionPromotion = {
  target_type: string;
  created_record_ids: number[];
  created_timeline_event_ids: number[];
};

export type ReviewPromotionPreview = {
  target_type: string;
  can_approve: boolean;
  missing_required_fields: string[];
  normalized_payload: Record<string, string>;
};

export type ReviewGroup = {
  group_id: string;
  title: string;
  item_type: string;
  status: ReviewStatus;
  permission_level: string;
  items: ReviewItem[];
  total_count: number;
  avg_confidence: number;
};

export type ReviewResponse = {
  groups: ReviewGroup[];
  items: ReviewItem[];
  total_count?: number;
  limit?: number;
  offset?: number;
  has_more?: boolean;
  include_previews?: boolean;
};

export type ReviewBulkActionResponse = {
  action: "approve" | "reject" | string;
  approved_count: number;
  rejected_count: number;
  failed_items: { id: number; detail: string }[];
  skipped_items: { id: number; detail: string }[];
  approved_item_ids: number[];
  rejected_item_ids: number[];
  replayed_items: ReviewTransitionItem[];
};

export type ReviewTransitionItem = {
  item_id: number;
  status: ReviewStatus;
  replayed: boolean;
  promotion: ReviewTransitionPromotion | null;
};

export type ReviewWorkflowSourceRef = {
  source_type: "gmail" | "gmail_attachment" | "drive" | "calendar";
  source_id: string;
  version_or_signature: string;
};

export type ReviewWorkflowRunRequest = {
  source_refs: ReviewWorkflowSourceRef[];
  agent_names: string[];
  client_request_id?: string;
};

export type ReviewWorkflowErrorCode =
  | "invalid_input"
  | "not_found"
  | "idempotency_key_reused"
  | "evidence_changed"
  | "permission_denied"
  | "checkpoint_unavailable"
  | "checkpoint_failed"
  | "review_unresolved"
  | "runtime_version_unavailable"
  | "model_unavailable"
  | "budget_exceeded"
  | "concurrent_resume"
  | "invalid_state_transition";

export type ReviewWorkflowCheckpointMode = "disabled" | "memory" | "postgres";

export type ReviewWorkflowDiagnostic = {
  enabled: boolean;
  available: boolean;
  checkpoint_mode: ReviewWorkflowCheckpointMode;
  durable: boolean;
  graph_version: string;
  default_agent_names: string[];
  error_code: ReviewWorkflowErrorCode | null;
};

export type ReviewWorkflowBudgetStatus = "within_budget" | "cached" | "no_input" | "over_budget";

export type ReviewWorkflowDryRun = {
  workflow_name: "company-memory-review";
  graph_version: "company-memory-review-v2.0";
  source_count: number;
  agent_names: string[];
  selection_policy_version: "company-memory-review-selection:v1";
  estimated_input_tokens: number;
  estimated_output_tokens: number;
  estimated_cost_usd: number;
  budget_limit_usd: number | null;
  budget_status: ReviewWorkflowBudgetStatus;
  cache_hit: boolean;
  requires_explicit_run: true;
};

export type ReviewWorkflowLifecycleStatus =
  | "created"
  | "drafting"
  | "checkpoint_pending"
  | "awaiting_human_review"
  | "resuming"
  | "completed"
  | "needs_more_evidence"
  | "checkpoint_failed"
  | "failed"
  | "cancelled";

export type ReviewWorkflowStatusCounts = Partial<Record<ReviewStatus, number>>;

export type ReviewWorkflowStatus = {
  thread_id: string;
  status: ReviewWorkflowLifecycleStatus;
  review_item_count: number;
  review_status_counts: ReviewWorkflowStatusCounts;
  durable: boolean;
  graph_version: string;
  review_resolution_ready: boolean;
  checkpoint_resumable: boolean;
  resume_allowed: boolean;
  retry_allowed: boolean;
  created_at: string;
  updated_at: string;
  error_code: ReviewWorkflowErrorCode | null;
  resume_error_code: ReviewWorkflowErrorCode | null;
};

export type SearchResult = {
  id: number;
  source_id: string;
  text: string;
  source_snippet: string;
  source_url?: string | null;
  source_type?: string | null;
  permission_level: string;
  relevance_score: number;
  matched_terms: string[];
  citation: RagCitation;
  parser_status?: string | null;
  parser_status_reason?: string | null;
  revision_id?: string | null;
};

export type SearchResponse = {
  retrieval_backend: "deterministic_lexical" | "pgvector" | string;
  cost_policy: {
    embedding_query_call: boolean;
    paid_llm_call: boolean;
    requires_pgvector_flag: boolean;
  };
  results: SearchResult[];
  hidden_match_count: number;
  permission_notice?: string;
};

export type DocumentSummary = {
  id: number;
  source_id: string;
  title: string;
  current_version: string;
  revision_id?: string | null;
  parser_name?: string | null;
  parser_status?: string | null;
  parser_status_reason?: string | null;
  chunk_count?: number | null;
};

export type DocumentVersionSummary = {
  id: number;
  version: string;
  revision_id?: string | null;
  parser_name?: string | null;
  parser_status?: string | null;
  parser_status_reason?: string | null;
  chunk_count?: number | null;
};

export type RagCitation = {
  source_id: string;
  source_url: string;
  source_type?: string | null;
  permission_level: string;
  source_snippet: string;
  relevance_score: number;
  matched_terms: string[];
};

export type AskResponse = {
  agent_name: string;
  prompt_version: string;
  question: string;
  answer: string;
  source_ids: string[];
  source_links: string[];
  source_snippets: string[];
  citations: RagCitation[];
  permission_level: string;
  hidden_match_count: number;
  permission_notice?: string | null;
  agent_run_id?: number | null;
  cache_key: string;
  model_name: string;
  estimated_cost_usd: number;
  token_usage: {
    input_tokens: number;
    output_tokens: number;
    total_tokens: number;
  };
};

export type AssistantConversation = {
  id: number;
  title: string;
  summary?: string | null;
  created_at: string;
  updated_at: string;
};

export type AssistantMessage = {
  id: number;
  conversation_id: number;
  role: "user" | "assistant" | string;
  content: string;
  citations: RagCitation[];
  source_ids: string[];
  source_links: string[];
  source_snippets: string[];
  permission_level?: string | null;
  hidden_match_count: number;
  permission_notice?: string | null;
  agent_run_id?: number | null;
  metadata: Record<string, unknown>;
  created_at: string;
};

export type AssistantConversationsResponse = {
  conversations: AssistantConversation[];
};

export type AssistantConversationCreatedResponse = {
  conversation: AssistantConversation;
};

export type AssistantMessagesResponse = {
  conversation: AssistantConversation;
  messages: AssistantMessage[];
};

export type AssistantTurnResponse = {
  conversation: AssistantConversation;
  user_message: AssistantMessage;
  assistant_message: AssistantMessage;
};

export type AssistantEmailSendResponse = {
  message: AssistantMessage;
  status: string;
  gmail_message_id?: string | null;
};

export type ProjectEvidence = {
  id: string;
  source_id: string;
  source_type: string;
  title: string;
  source_url: string;
  source_snippet: string;
  permission_level: string;
  timestamp: string;
  task_summary: string;
  evidence_reason: string;
};

export type ProjectTimelineItem = {
  id: string;
  item_type: string;
  title: string;
  summary: string;
  source_links: string[];
  source_snippets: string[];
  confidence_score: number;
  permission_level: string;
  review_status: string;
  created_at: string;
  occurred_at?: string;
  evidence_reason: string;
  project_key?: string | null;
  completed_at?: string | null;
  completed_by?: string | null;
};

export type ProjectMemory = {
  project_key: string;
  name: string;
  summary: string;
  source_types: string[];
  evidence_count: number;
  permission_level: string;
  latest_timestamp: string;
  pending_review_count: number;
  evidence: ProjectEvidence[];
  timeline_items: ProjectTimelineItem[];
  activity_items: ProjectTimelineItem[];
};

export type ProjectsResponse = {
  project_count: number;
  hidden_project_count: number;
  hidden_evidence_count?: number;
  projects: ProjectMemory[];
};

export type IntegrationSyncResponse = {
  job_id: string;
  connector_type: string;
  status: string;
  created_review_items: number;
  pending_review_count: number;
  fetched_events: number;
  skipped_events: number;
  parser_status_counts?: Record<string, number>;
  changed_source_ids?: string[];
  changed_source_refs?: ReviewWorkflowSourceRef[];
  agent_generated_items?: number;
  project_assignment_items?: number;
};

export type IntegrationManifest = {
  type: string;
  display_name: string;
  mode: string;
  status: string;
  auth_type: string;
  required_scopes: string[];
  sync_strategy: string;
  cost_policy: string;
};

export type OAuthInstallUrlResponse = {
  connector_type: string;
  configured: boolean;
  install_url?: string | null;
  state?: string | null;
  required_scopes: string[];
};

export type SlackOAuthInstallUrlResponse = OAuthInstallUrlResponse;

export type IntegrationConnection = {
  connector_type: string;
  workspace_id: string;
  workspace_name: string;
  status: string;
  credential_status?: "available" | "missing";
  masked_bot_token: string;
  scopes: string[];
};

export type AgentReviewResponse = {
  agent_name: string;
  status: string;
  created_review_items: number;
  preflight?: AgentLlmPreflight;
};

export type SlackAgentReviewResponse = AgentReviewResponse;

export type AgentLlmPreflight = {
  action: "run" | "skip" | "blocked" | "use_cache" | string;
  reason: string;
  budget_status: string;
  model_name?: string | null;
  provider_order: string[];
  available_providers: string[];
  estimated_input_tokens: number;
  estimated_output_tokens: number;
  estimated_total_tokens: number;
  estimated_cost_usd: number;
  budget_limit_usd?: number | null;
  evidence_message_count: number;
  max_evidence_messages: number;
  source_window?: string;
  requires_paid_confirmation: boolean;
};

export type SlackLlmPreflight = AgentLlmPreflight;

export type MailDocumentLlmPreflight = AgentLlmPreflight;

export type SlackRuntimeStatus = {
  connector_type: "slack";
  mode: "mock" | "live";
  configured_channel_ids: string[];
  selected_channel_ids: string[];
  channel_options: {
    id: string;
    name: string;
    is_selected: boolean;
    is_configured: boolean;
  }[];
  connection_status: string;
  credential_status: "available" | "missing";
  latest_sync?: {
    job_id: string;
    status: string;
    message: string;
    progress_pct: number;
    created_at?: string | null;
    updated_at?: string | null;
  } | null;
  latest_sync_summary?: {
    fetched_events: number;
    created_review_items: number;
    skipped_events: number;
  } | null;
  last_error?: {
    code: string;
    message: string;
    action_hint: string;
  } | null;
  agent_bridge: {
    slack_source_count: number;
    pending_review_count: number;
    ready_for_agent_test: boolean;
  };
  cost_policy: {
    status_lookup_triggers_sync: boolean;
    status_lookup_triggers_llm: boolean;
    thread_reply_fetch_is_incremental: boolean;
  };
};

export type GoogleRuntimeStatus = {
  connector_type: "gmail" | "drive" | "calendar";
  mode: "mock" | "live";
  connection_status: string;
  credential_status: "available" | "missing";
  account_name?: string | null;
  latest_sync?: {
    job_id: string;
    status: string;
    message: string;
    progress_pct: number;
    created_at?: string | null;
    updated_at?: string | null;
  } | null;
  cost_policy: {
    status_lookup_triggers_sync: boolean;
    status_lookup_triggers_llm: boolean;
  };
};

export type MessageChannel = {
  id: string;
  name: string;
  description: string;
  unread_count: number;
};

export type Message = {
  id: string;
  channel_id: string;
  author_name: string;
  author_role: string;
  body: string;
  created_at: string;
};

export type MessageChannelsResponse = {
  channels: MessageChannel[];
};

export type ChannelMessagesResponse = {
  channel: MessageChannel;
  messages: Message[];
};
