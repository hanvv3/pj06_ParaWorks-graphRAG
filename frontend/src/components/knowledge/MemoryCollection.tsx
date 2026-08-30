import { CheckCircle2, ExternalLink, ShieldCheck } from "lucide-react";
import type { KnowledgeItem } from "@/lib/api/types";
import { AutoReviewBadge } from "@/components/review/AutoReviewBadge";

type MemoryCollectionProps = {
  eyebrow: string;
  title: string;
  description: string;
  items: KnowledgeItem[];
  emptyText: string;
  metricLabel: string;
};

export function MemoryCollection({
  eyebrow,
  title,
  description,
  items,
  emptyText,
  metricLabel,
}: MemoryCollectionProps) {
  return (
    <div className="reference-dashboard space-y-4">
      <div className="page-heading reference-heading">
        <div>
          <p className="text-[13px] font-bold text-[var(--primary-dark)]">{eyebrow}</p>
          <h1>{title}</h1>
          <p>{description}</p>
        </div>
        <div className="panel inline-flex h-fit w-fit items-center gap-2 px-4 py-3 text-[13px] font-bold">
          <ShieldCheck className="h-4 w-4 text-[var(--primary)]" aria-hidden="true" />
          {items.length.toLocaleString()} {metricLabel}
        </div>
      </div>

      <section className="grid gap-4 lg:grid-cols-2">
        {items.map((item) => (
          <MemoryCard key={item.id} item={item} />
        ))}
        {items.length === 0 ? <EmptyState text={emptyText} /> : null}
      </section>
    </div>
  );
}

export function MemoryCard({ item }: { item: KnowledgeItem }) {
  const firstLink = item.source_links[0];
  const firstSnippet = item.source_snippets[0];

  return (
    <article className="panel reference-panel">
      <div className="flex flex-wrap items-center gap-2">
        <span className="badge green">
          <CheckCircle2 className="mr-1 h-3.5 w-3.5" aria-hidden="true" />
          {statusLabel(item.review_status)}
        </span>
        <span className="badge blue">{permissionLabel(item.permission_level)}</span>
        <AutoReviewBadge resolutionSource={item.resolution_source} />
        {item.priority ? <span className="priority-badge warning">{item.priority}</span> : null}
      </div>

      <h2 className="mt-4 text-[16px] font-extrabold leading-6 text-ink">{item.title}</h2>
      <p className="mt-2 text-[13px] leading-6 text-muted">{item.summary}</p>

      <div className="mt-4 rounded-lg border border-line bg-surface-soft p-4">
        <p className="text-[12px] font-bold text-muted">신뢰도 {(item.confidence_score * 100).toFixed(0)}%</p>
        {firstSnippet ? (
          <p className="mt-2 border-l-2 border-[#cbd5e1] pl-3 text-[12px] leading-5 text-muted">
            {firstSnippet}
          </p>
        ) : (
          <p className="mt-2 text-[12px] text-muted">표시할 근거 스니펫이 없습니다.</p>
        )}
      </div>

      {firstLink ? (
        <a
          href={firstLink}
          target="_blank"
          rel="noreferrer"
          className="mt-4 inline-flex items-center gap-1 text-[12px] font-bold text-[var(--primary-dark)] underline-offset-4 hover:underline"
        >
          <ExternalLink className="h-3.5 w-3.5" aria-hidden="true" />
          근거 열기
        </a>
      ) : null}
    </article>
  );
}

export function EmptyState({ text }: { text: string }) {
  return <div className="panel reference-panel px-5 py-10 text-[13px] text-muted">{text}</div>;
}

function statusLabel(status: string) {
  if (status === "approved") {
    return "승인됨";
  }
  if (status === "pending_review") {
    return "검토 대기";
  }
  if (status === "needs_more_evidence") {
    return "근거 요청";
  }
  if (status === "rejected") {
    return "반려됨";
  }
  return status;
}

function permissionLabel(permissionLevel: string) {
  if (permissionLevel === "restricted") {
    return "제한";
  }
  if (permissionLevel === "internal") {
    return "내부";
  }
  if (permissionLevel === "public") {
    return "공개";
  }
  return permissionLevel;
}
