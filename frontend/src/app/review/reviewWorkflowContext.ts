export type ReviewProjectOption = {
  project_key: string;
  name: string;
};

export type ReviewWorkflowQuery =
  | { kind: "global" }
  | { kind: "invalid" }
  | { kind: "workflow"; workflowThreadId: string };

const SERVER_ISSUED_REVIEW_WORKFLOW_ID = /^[0-9a-f]{32}$/;

export function parseReviewWorkflowQuery(rawWorkflowThreadId: string | null): ReviewWorkflowQuery {
  if (rawWorkflowThreadId === null) return { kind: "global" };

  if (!SERVER_ISSUED_REVIEW_WORKFLOW_ID.test(rawWorkflowThreadId)) {
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
