export type ReviewProjectOption = {
  project_key: string;
  name: string;
};

export type ReviewWorkflowQuery =
  | { kind: "global" }
  | { kind: "invalid" }
  | { kind: "workflow"; workflowThreadId: string };

const REVIEW_WORKFLOW_THREAD_ID_MAX_LENGTH = 64;

export function parseReviewWorkflowQuery(rawWorkflowThreadId: string | null): ReviewWorkflowQuery {
  if (rawWorkflowThreadId === null) return { kind: "global" };

  if (!rawWorkflowThreadId.trim() || Array.from(rawWorkflowThreadId).length > REVIEW_WORKFLOW_THREAD_ID_MAX_LENGTH) {
    return { kind: "invalid" };
  }

  return { kind: "workflow", workflowThreadId: rawWorkflowThreadId };
}

export function deriveReviewBulkProjectState({
  isRenderedContextCurrent,
  loading,
  invalidWorkflowContext,
  bulkProjectKey,
  definedProjects,
}: {
  isRenderedContextCurrent: boolean;
  loading: boolean;
  invalidWorkflowContext: boolean;
  bulkProjectKey: string;
  definedProjects: ReviewProjectOption[];
}) {
  const ready = isRenderedContextCurrent && !loading && !invalidWorkflowContext;
  return {
    value: ready ? bulkProjectKey : "",
    projects: ready ? definedProjects : [],
    disabled: !ready,
  };
}
